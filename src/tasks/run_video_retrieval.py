import os
import time
import math
import numpy as np
from os.path import join, exists
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torchvision.utils as tutils
from torch.utils.data.distributed import DistributedSampler

from apex import amp
import torch.distributed as dist
from apex.parallel import DistributedDataParallel
from transformers import CLIPTokenizerFast

from src.modeling.VidCLIP import VidCLIP

from src.datasets.dataset_video_retrieval import (
    HDVILAVideoRetrievalDataset, VideoRetrievalCollator)
from src.datasets.dataloader import InfiniteIterator, PrefetchLoader

from src.configs.config import shared_configs
from src.utils.misc import set_random_seed, NoOp, zero_none_grad
from src.utils.logger import LOGGER, TB_LOGGER, add_log_to_file, RunningMeter
from src.utils.basic_utils import (
    load_jsonl, load_json, save_json, get_rounded_percentage)
from src.utils.basic_utils import flat_list_of_lists
from src.utils.load_save import (ModelSaver,
                                 BestModelSaver,
                                 save_training_meta,
                                 load_state_dict_with_mismatch)
from src.utils.load_save import E2E_TrainingRestorer as TrainingRestorer
from src.utils.distributed import AllGather
from src.optimization.sched import get_lr_sched
from src.optimization.utils import setup_e2e_optimizer
from src.optimization.loss import build_loss_func
from src.utils.distributed import all_gather_list
from src.utils.metrics import cal_cossim, compute_metrics, compute_metrics_multi, np_softmax

dist.init_process_group(backend='nccl')
allgather = AllGather.apply

def init_device(args, local_rank):
    # global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    LOGGER.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    # if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
    #     raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
    #         args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def mk_video_ret_dataloader(dataset_name, vis_format, anno_path, vis_dir, cfg, tokenizer, mode):
    """"""
    is_train = mode == "train"
    dataset = HDVILAVideoRetrievalDataset(
        cfg=cfg,
        vis_dir=vis_dir,
        anno_path=anno_path,
        vis_format=vis_format,
        mode=mode
    )
    LOGGER.info(f"[{dataset_name}] is_train {is_train} "
                f"dataset size {len(dataset)}, ")

    batch_size = cfg.train_batch_size if is_train else cfg.test_batch_size
    sampler = DistributedSampler(
        dataset, num_replicas=torch.cuda.device_count(), rank=dist.get_rank(),
        shuffle=is_train)
    vret_collator = VideoRetrievalCollator(
        tokenizer=tokenizer, max_length=cfg.max_txt_len, is_train=is_train)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=cfg.n_workers,
                            pin_memory=cfg.pin_mem,
                            collate_fn=vret_collator.collate_batch)
    return dataloader



def setup_dataloaders(cfg, tokenizer):
    LOGGER.info("Init. train_loader and val_loader...")

    db = cfg.train_datasets
    train_loader = mk_video_ret_dataloader(
        dataset_name=db.name, vis_format=db.vis_format,
        anno_path=db.txt, vis_dir=db.vis,
        cfg=cfg, tokenizer=tokenizer, mode="train"
    )

    val_loaders = {}
    for db in cfg.val_datasets:
        val_loaders[db.name] = mk_video_ret_dataloader(
            dataset_name=db.name, vis_format=db.vis_format,
            anno_path=db.txt, vis_dir=db.vis,
            cfg=cfg, tokenizer=tokenizer, mode="val"
        )

    inference_loaders = {}
    for db in cfg.inference_datasets:
        inference_loaders[db.name] = mk_video_ret_dataloader(
            dataset_name=db.name, vis_format=db.vis_format,
            anno_path=db.txt, vis_dir=db.vis,
            cfg=cfg, tokenizer=tokenizer, mode="test"
        )
    return train_loader, val_loaders, inference_loaders


def setup_model(cfg, device=None):
    LOGGER.info("Setup model...")
    
    model = VidCLIP(cfg)

    if cfg.e2e_weights_path:
        LOGGER.info(f"Loading e2e weights from {cfg.e2e_weights_path}") # path/to/CLIP-ViP-B/32/checkpoint
        load_state_dict_with_mismatch(model, cfg.e2e_weights_path)
    
    if hasattr(cfg, "overload_logit_scale"):
        model.overload_logit_scale(cfg.overload_logit_scale)
    
    model.to(device)

    LOGGER.info("Setup model done!")
    return model

@torch.no_grad()
def validate(model, val_loaders, cfg, device, cur_best_t2v=None):

    model.eval()

    st = time.time()

    for param in model.module.text_proxy.parameters():
        # Check if parameter dtype is Half (float16)
        if param.dtype == torch.float16:
            param.data = param.data.to(torch.float32)

    if cur_best_t2v == None:
        best_t2v = 0.0
    else:
        best_t2v = cur_best_t2v

    for loader_name, val_loader in val_loaders.items():
        LOGGER.info(f"Loop val_loader {loader_name}.")
        valid_len = len(val_loader.dataset)
        text_feats = []
        vis_feats = []
        vis_patch_feats = []

        for val_step, batch in enumerate(val_loader):
            feats = model(False, None, **batch)
            vis_feat = allgather(feats['vis_features'], cfg)
            text_feat = allgather(feats['text_features'], cfg)
            vis_patch_feat = allgather(feats['vis_patch_features'], cfg)

            text_feats.append(text_feat) # a batch consists of n gpus' sub_batch
            vis_feats.append(vis_feat)
            vis_patch_feats.append(vis_patch_feat)

        sim_proxy = []

        for idx1, b_text in enumerate(text_feats):
            each_row_proxy = []
            for idx2, b_ in enumerate(zip(vis_patch_feats, vis_feats)):
                b_vis_patch, b_vis = b_
                if hasattr(model, 'module'): # we do not implement this without nvidia apex
                    text_proxies = model.module.text_proxy(b_text, b_vis_patch, b_vis_patch) # (a,b,dim): each text of a generates b proxies with b videos
                    proxy_logits = model.module.sim_proxy(text_proxies, b_vis, is_train=False)
                    each_row_proxy.append(proxy_logits.cpu().detach().numpy()) # +(a,b)

            each_row_proxy = np.concatenate(tuple(each_row_proxy), axis=-1) # n*(a,b) -> (a,n*b)
            each_row_proxy = each_row_proxy[:, :valid_len] # -> (a,valid_len)
            sim_proxy.append(each_row_proxy)

        sim_proxy = np.concatenate(tuple(sim_proxy), axis=0) # n*(a,valid_len) -> (n*a,valid_len)
        sim_proxy = sim_proxy[:valid_len] # -> (valid_len,valid_len)

        text_feats = torch.cat(text_feats, dim=0)
        vis_feats = torch.cat(vis_feats, dim=0)
        text_feats = text_feats[:valid_len] # (valid_len,dim)
        vis_feats = vis_feats[:valid_len] # (valid_len,dim)

        sim_matrix_ = cal_cossim(text_feats.cpu().numpy(), vis_feats.cpu().numpy())

        for i in np.arange(0, 1.0, 0.1):
            LOGGER.info(f'==================  inference proxy logits weight : {i} =====================')
            sim_matrix = sim_matrix_ + sim_proxy * i
            for type in ["simple", "DSL"]:
                LOGGER.info(f"Evaluate under setting: {type}.")
                val_log = {f'valid/{loader_name}_t2v_recall_1': 0,
                    f'valid/{loader_name}_t2v_recall_5': 0,
                    f'valid/{loader_name}_t2v_recall_10': 0,
                    f'valid/{loader_name}_t2v_recall_median': 0,
                    f'valid/{loader_name}_t2v_recall_mean': 0,
                    # f'valid/{loader_name}_v2t_recall_1': 0,
                    # f'valid/{loader_name}_v2t_recall_5': 0,
                    # f'valid/{loader_name}_v2t_recall_10': 0,
                    # f'valid/{loader_name}_v2t_recall_median': 0,
                    # f'valid/{loader_name}_v2t_recall_mean': 0}
                           }

                if type == "DSL":
                    sim_matrix = sim_matrix * np_softmax(sim_matrix*100, axis=0)

                v2tr1,v2tr5,v2tr10,v2tmedr,v2tmeanr = compute_metrics(sim_matrix.T)
                t2vr1,t2vr5,t2vr10,t2vmedr,t2vmeanr = compute_metrics(sim_matrix)

                if type == "simple" and t2vr1 > best_t2v:
                    best_t2v = t2vr1


                val_log.update({f'valid/{loader_name}_t2v_recall_1': t2vr1,
                            f'valid/{loader_name}_t2v_recall_5': t2vr5,
                            f'valid/{loader_name}_t2v_recall_10': t2vr10,
                            f'valid/{loader_name}_t2v_recall_median': t2vmedr,
                            f'valid/{loader_name}_t2v_recall_mean': t2vmeanr,
                            f'valid/{loader_name}_v2t_recall_1': v2tr1,
                            f'valid/{loader_name}_v2t_recall_5': v2tr5,
                            f'valid/{loader_name}_v2t_recall_10': v2tr10,
                            f'valid/{loader_name}_v2t_recall_median': v2tmedr,
                            f'valid/{loader_name}_v2t_recall_mean': v2tmeanr
                            })

                LOGGER.info(f"validation finished in {int(time.time() - st)} seconds, "
                        f"validated on {vis_feats.shape[0]} videos \n"
                        f"{loader_name} t2v recall@1: {val_log['valid/%s_t2v_recall_1'%(loader_name)] * 100:.4f} "
                        f"{loader_name} t2v recall@5: {val_log['valid/%s_t2v_recall_5'%(loader_name)] * 100:.4f} "
                        f"{loader_name} t2v recall@10: {val_log['valid/%s_t2v_recall_10'%(loader_name)] * 100:.4f} "
                        f"{loader_name} t2v recall_med: {val_log['valid/%s_t2v_recall_median'%(loader_name)] :.1f} "
                        f"{loader_name} t2v recall_mean: {val_log['valid/%s_t2v_recall_mean'%(loader_name)] :.1f} "
                        f"{loader_name} v2t recall@1: {val_log['valid/%s_v2t_recall_1'%(loader_name)] * 100:.4f} "
                        f"{loader_name} v2t recall@5: {val_log['valid/%s_v2t_recall_5'%(loader_name)] * 100:.4f} "
                        f"{loader_name} v2t recall@10: {val_log['valid/%s_v2t_recall_10'%(loader_name)] * 100:.4f} "
                        f"{loader_name} v2t recall_med: {val_log['valid/%s_v2t_recall_median'%(loader_name)] :.1f} "
                        f"{loader_name} v2t recall_mean: {val_log['valid/%s_v2t_recall_mean'%(loader_name)] :.1f} "
                        )
        TB_LOGGER.log_scalar_dict(val_log)

    for param in model.module.text_proxy.parameters():
        # Check if parameter dtype is Half (float16)
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.float16)

    model.train()
    return val_log, best_t2v

def start_training():
    cfg = shared_configs.get_pretraining_args()
    blob_mount(cfg)
    set_random_seed(cfg.seed)

    device, n_gpu = init_device(cfg, cfg.local_rank)
    cfg.n_gpu = n_gpu
    cfg.world_size = n_gpu
    torch.cuda.set_device(cfg.local_rank)
    if dist.get_rank() != 0:
        LOGGER.disabled = True
    LOGGER.info(f"device: {device} n_gpu: {n_gpu}, "
                f"rank: {dist.get_rank()}, 16-bits training: {cfg.fp16}")

    if dist.get_rank() != 0:
        LOGGER.disabled = True

    model = setup_model(cfg, device=device)
    # model.train()

    optimizer = setup_e2e_optimizer(model, cfg)

    model, optimizer = amp.initialize(
        model, optimizer, enabled=cfg.fp16, opt_level=cfg.amp_level,
        keep_batchnorm_fp32=True if cfg.amp_level=='O2' else None)
    model = DistributedDataParallel(model)

    # prepare data
    tokenizer = CLIPTokenizerFast.from_pretrained(cfg.clip_weights)
    train_loader, val_loaders, inference_loaders = setup_dataloaders(cfg, tokenizer)

    img_norm = None
    train_loader = PrefetchLoader(train_loader, img_norm)
    val_loaders = {k: PrefetchLoader(v, img_norm)
                for k, v in val_loaders.items()}
    inference_loaders = {k: PrefetchLoader(v, img_norm)
                for k, v in inference_loaders.items()}

    # compute the number of steps and update cfg
    total_train_batch_size = int(
        n_gpu * cfg.train_batch_size *
        cfg.gradient_accumulation_steps * cfg.max_n_example_per_group)

    total_n_examples = len(train_loader.dataset) * cfg.max_n_example_per_group 
    print('total_n_examples', total_n_examples)

    cfg.num_train_steps = int(math.ceil(
        1. * cfg.num_train_epochs * total_n_examples / total_train_batch_size))

    cfg.valid_steps = int(math.ceil( #
        1. * cfg.num_train_steps / cfg.num_valid /
        cfg.min_valid_steps)) * cfg.min_valid_steps
    actual_num_valid = int(math.floor(
        1. * cfg.num_train_steps / cfg.valid_steps)) + 1

    n_steps_in_epoch = int(math.ceil(1. * total_n_examples / total_train_batch_size))

    # restore
    restorer = TrainingRestorer(cfg, model, optimizer)
    global_step = restorer.global_step
    TB_LOGGER.global_step = global_step
    if dist.get_rank() == 0:
        LOGGER.info("Saving training meta...")
        save_training_meta(cfg)
        LOGGER.info("Saving training done...")
        if cfg.if_tb_log:
            TB_LOGGER.create(join(cfg.output_dir, 'log'))
        # pbar = tqdm(total=cfg.num_train_steps)
        if cfg.if_model_saver:
            model_saver = ModelSaver(join(cfg.output_dir, "ckpt"))
            best_model_saver = BestModelSaver(join(cfg.output_dir, "ckpt"))
        else:
            model_saver = NoOp()
            restorer = NoOp()
            best_model_saver = NoOp()
            
        if cfg.if_log2file:
            add_log_to_file(join(cfg.output_dir, "log", "log.txt"))
    else:
        LOGGER.disabled = True
        # pbar = NoOp()
        model_saver = NoOp()
        restorer = NoOp()
        best_model_saver = NoOp()

    if global_step > 0:
        pass # pbar.update(global_step)

    LOGGER.info(cfg)
    LOGGER.info("Starting training...")
    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info(f"  Single-GPU Non-Accumulated batch size = {cfg.train_batch_size}")
    LOGGER.info(f"  max_n_example_per_group = {cfg.max_n_example_per_group}")
    LOGGER.info(f"  Accumulate steps = {cfg.gradient_accumulation_steps}")
    LOGGER.info(f"  Total batch size = #GPUs * Single-GPU batch size * "
                f"max_n_example_per_group * Accumulate steps [Image] = {total_train_batch_size}")
    LOGGER.info(f"  Total #epochs = {cfg.num_train_epochs}")
    LOGGER.info(f"  Total #steps = {cfg.num_train_steps}")
    LOGGER.info(f"  Validate and Save every {cfg.valid_steps} steps, in total {actual_num_valid} times")
    LOGGER.info(f"  Only Validate every {cfg.only_valid_steps} steps")

    # quick hack for amp delay_unscale bug
    # with optimizer.skip_synchronize():
    #     optimizer.zero_grad()
    #     if global_step == 0:
    #         optimizer.step()
    # with model.no_sync():
    #     optimizer.zero_grad()
    #     if global_step == 0:
    #         optimizer.step()

    running_loss = RunningMeter('train_loss', smooth=0)

    LOGGER.info(f'Step zero: start inference')
    validate(model, inference_loaders, cfg, device)

    loss_func = build_loss_func(cfg.loss_config)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    for step, batch in enumerate(InfiniteIterator(train_loader)):
        model.train()
        outputs = model(True, step, **batch)
        if cfg.loss_config.if_gather: # 1
            vis_feat = allgather(outputs['vis_features'], cfg)
            text_feat = allgather(outputs['text_features'], cfg)
            text_proxy = outputs['text_proxy']
            proxy_logits, pos_logits, contrast_logits = None, None, None
            if cfg.loss_config.loss_name in ["NCELearnableTempLoss", "NCELearnableTempDSLLoss"]:
                if hasattr(model, 'module'):
                    proxy_logits, pos_logits, contrast_logits = model.module.sim_proxy(text_proxy, vis_feat, text_feat)
                    logit_scale = model.module.clipmodel.logit_scale
                else:
                    proxy_logits, pos_logits, contrast_logits = model.sim_proxy(text_proxy, vis_feat, text_feat)
                    logit_scale = model.clipmodel.logit_scale
                loss = loss_func(vis_feat, text_feat, logit_scale, proxy_logits=proxy_logits, pos_logits=pos_logits, contrast_logits=contrast_logits)
            else:
                loss = loss_func(vis_feat, text_feat)
        else:
            loss = outputs['loss']

        if hasattr(model, 'module'):
            torch.clamp_(model.module.clipmodel.logit_scale.data, 0, np.log(200))
            logit_scale_ = model.module.clipmodel.logit_scale.data
        else:
            torch.clamp_(model.clipmodel.logit_scale.data, 0, np.log(200))
            logit_scale_ = model.clipmodel.logit_scale.data

        if step % 10 == 0:
            lr_ = optimizer.param_groups[0]['lr']
            LOGGER.info(f'Step {global_step}: loss {loss} lr {lr_} logit_scale {logit_scale_}')

        running_loss(loss.item())

        delay_unscale = (step + 1) % cfg.gradient_accumulation_steps != 0
        with amp.scale_loss(
                loss, optimizer, delay_unscale=delay_unscale
                ) as scaled_loss:
            scaled_loss.backward()
            # zero_none_grad(model)
            # optimizer.synchronize()

        # optimizer
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            global_step += 1
            TB_LOGGER.log_scalar_dict({'vtc_loss': running_loss.val})
            n_epoch = int(1.* cfg.gradient_accumulation_steps *
                          global_step / n_steps_in_epoch)
            # learning rate scheduling transformer
            lr_this_step = get_lr_sched(
                global_step, cfg.decay, cfg.learning_rate,
                cfg.num_train_steps, warmup_ratio=cfg.warmup_ratio,
                decay_epochs=cfg.step_decay_epochs, multi_step_epoch=n_epoch)

            for pg_n, param_group in enumerate(
                    optimizer.param_groups):
                if pg_n in [0, 1]:
                    param_group['lr'] = (
                        cfg.lr_mul * lr_this_step)
                elif pg_n in [2, 3]:
                    param_group['lr'] = lr_this_step
                
            TB_LOGGER.add_scalar(
                "train/lr", lr_this_step,
                global_step)

            # update model params
            if cfg.grad_norm != -1:
                grad_norm = clip_grad_norm_(
                    amp.master_params(optimizer), cfg.grad_norm)
                TB_LOGGER.add_scalar("train/grad_norm", grad_norm, global_step)
            TB_LOGGER.step()

            # Check if there is None grad
            # none_grads = [
            #     p[0] for p in model.named_parameters()
            #     if p[1].requires_grad and p[1].grad is None]
            #
            # assert len(none_grads) == 0, f"{none_grads}"

            optimizer.step()
            optimizer.zero_grad()


            restorer.step()

            # checkpoint
            if global_step % cfg.valid_steps == 0:
                LOGGER.info(f'Step {global_step}: start validation and Save')
                _, t2vr1 = validate(model, inference_loaders, cfg, device, best_model_saver.get_bestr1())
                model_saver.save(step=global_step, model=model)
                if dist.get_rank() == 0 and cfg.if_model_saver and t2vr1 > best_model_saver.get_bestr1():
                    best_model_saver.save(step=global_step, model=model)
                    best_model_saver.bestr1 = t2vr1
                result_t2vr1 = best_model_saver.get_bestr1()
                LOGGER.info(f"===================== Current Best t2v recall@1: {result_t2vr1} ===============================\n")

            else:
                if global_step % cfg.only_valid_steps == 0:
                    LOGGER.info(f'Step {global_step}: start inference')
                    _, t2vr1 = validate(model, inference_loaders, cfg, device, best_model_saver.get_bestr1())
                    if dist.get_rank() == 0 and cfg.if_model_saver and t2vr1 > best_model_saver.get_bestr1():
                        best_model_saver.save(step=global_step, model=model)
                        best_model_saver.bestr1 = t2vr1
                    result_t2vr1 = best_model_saver.get_bestr1()
                    LOGGER.info(
                        f"===================== Current Best t2v recall@1: {result_t2vr1} ===============================\n")

        if global_step >= cfg.num_train_steps:
            break

    ender.record()
    torch.cuda.synchronize()
    elapsed_time = starter.elapsed_time(ender)
    print(f"Training time: {elapsed_time} ms")

    if global_step % cfg.valid_steps != 0:
        LOGGER.info(f'Step {global_step}: start validation')
        _, t2vr1 = validate(model, inference_loaders, cfg, device)

        model_saver.save(step=global_step, model=model)
        if dist.get_rank() == 0 and cfg.if_model_saver and t2vr1 > best_model_saver.bestr1:
            best_model_saver.save(step=global_step, model=model)
            best_model_saver.bestr1 = t2vr1

def blob_mount(cfg):
    keys = ["e2e_weights_path",
            "output_dir"]
    for key in keys:
        if cfg[key] is not None:
            cfg[key] = os.path.join(cfg.blob_mount_dir, cfg[key])

    db = cfg.train_datasets
    db.txt = os.path.join(cfg.blob_mount_dir, db.txt)
    db.vis = os.path.join(cfg.blob_mount_dir, db.vis)

    for db in cfg.val_datasets:
        db.txt = os.path.join(cfg.blob_mount_dir, db.txt)
        db.vis = os.path.join(cfg.blob_mount_dir, db.vis)

    for db in cfg.inference_datasets:
        db.txt = os.path.join(cfg.blob_mount_dir, db.txt)
        db.vis = os.path.join(cfg.blob_mount_dir, db.vis)



if __name__ == '__main__':
    start_training()
