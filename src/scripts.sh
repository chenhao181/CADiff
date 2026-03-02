# MovieLens
python main.py -d movielens -luser 0.2 -lrec 0.4 -lmse 1.5 -temp 0.6  --warmup 160

# KuaiRec
python main.py -d kuairec -luser 0.1 -lrec 0.1 -1mse 1.2 -warmup 3

# Yelp
python main.py -d yelp -luser 0.05 -lrec 0.6 -lmse 1.0 -temp1.4 --warmup 190
# Books
python main.py -d books -luser 0.1 -lambda_rec_loss 0.4 -lmse 1.5  --warmup 50 -tau 32