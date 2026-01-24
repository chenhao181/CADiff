import os
import pickle
import random
import time
from copy import deepcopy

import torch
from tqdm import tqdm

from data_loader import Dataset
from logger import OUTPUT_ROOT, get_args, logger
from models import CADiff
from utils import print_results, set_seed, valid_model


def main():
    args, parser = get_args()
    if args.dataset == 'books':
        print('!!!!!!!!!!!!!!!!!!!!!!!')
        print('Args reset for books')
        args.ri = 500
        args.bs = 512
        print('!!!!!!!!!!!!!!!!!!!!!!!')

    set_seed(args.seed)
    logger.set_log_file(args, parser)
    logger.print(args)

    device = torch.device(args.device)
    args.device = device

    dataset = Dataset(
        args.dataset,
        device=device,
        max_history_length=args.max_history_length,
        min_history_length=args.min_history_length
    )

    dataset.calc_user_cat_hist()
    logger.print(dataset)

    model = CADiff(dataset, device, args)
    logger.print(model)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    bs = args.bs
    best_score = 0
    best_epoch = 0
    patience = 0

    best_model = None
    best_test_res = None
    best_test_time = None

    train_data = dataset.seq_train_data
    start_time = time.time()

    for epoch in range(1, args.ne + 1):
        random.shuffle(train_data)
        logger.print(f'[Epoch {epoch}]')

        tqdm_batch = tqdm(
            range(0, len(train_data), bs) if args.iter == 0 else
            range(0, min(len(train_data), bs * args.iter), bs),
            bar_format='{l_bar}{r_bar}',
            desc='[Training]'
        )

        total_L_rec = 0
        total_L_user = 0
        total_L_mse = 0
        step = 0

        model.train()
        for i in tqdm_batch:
            step += 1
            batch_data = train_data[i:i + bs]
            uids, histories, lengths, pos_iids = zip(*batch_data)

            L_user, L_rec, L_mse = model.forward_bpr(
                uids, histories, lengths, pos_iids
            )

            loss = L_rec + L_user + L_mse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_L_rec += L_rec.item()
            total_L_user += L_user.item()
            total_L_mse += L_mse.item()

            if step % args.ri == 0:
                tqdm_batch.write(
                    f'[Step {step}] loss = {total_L_rec} + {total_L_user} + {total_L_mse} = '
                    f'{total_L_rec + total_L_user + total_L_mse}'
                )
                total_L_rec = 0
                total_L_user = 0
                total_L_mse = 0

        tqdm_batch.close()

        # warmup 阶段不验证
        if epoch <= args.warmup:
            continue

        # ---------- validation ----------
        model.eval()
        with torch.no_grad():
            valid_scores = valid_model(
                model,
                dataset.valid_data,
                dataset,
                args,
                ks=[20],
                diversity=False
            )

        score = valid_scores['ndcg@20']

        if score > best_score:
            best_score = score
            best_epoch = epoch
            patience = 0
            best_model = deepcopy(model.state_dict())

            model.eval()
            with torch.no_grad():
                best_test_res = valid_model(
                    model,
                    dataset.test_data,
                    dataset,
                    args,
                    ks=[3, 5, 10, 20],
                    diversity=True
                )
            best_test_time = time.time()
            print_results(args, best_test_res, start_time, best_test_time)

        else:
            patience += 1
            if patience >= args.patience:
                logger.print(
                    f'[ ][Epoch {epoch}] {valid_scores}, '
                    f'best = {best_score}, patience = {patience}/{args.patience}'
                )
                logger.print('[!!! Early Stop !!!]')
                break

        logger.print(
            f'[{"*" if patience == 0 else " "}][Epoch {epoch}] {valid_scores}, '
            f'best = {best_score}, patience = {patience}/{args.patience}'
        )

    end_time = time.time()

    # ---------- 只打印保存的 best test 结果 ----------
    if best_test_res is not None:
        print_results(args, best_test_res, start_time, end_time)

    logger.print(
        f'[Epoch] Total = {epoch}, Best = {best_epoch}, Best Score = {best_score}'
    )


if __name__ == '__main__':
    main()
