#！/bin/bash

git reset --hard
rm -rf ../results
rm -rf ../logs
rm -rf ../*outputs
rm -rf ../data
rm -rf ../checkpoints
git pull
mkdir ../inputs
cd ../inputs/
wget http://static1.challenger.ai/ai_challenger_stock_train_20171125.zip
wget http://static1.challenger.ai/ai_challenger_stock_test_20171125.zip
unzip ai_challenger_stock_test_20171125.zip
unzip ai_challenger_stock_train_20171125.zip
mv data/20171125/ai_challenger_stock_test_20171125/stock_test_data_20171125.csv ./
mv data/20171125/ai_challenger_stock_train_20171125/stock_train_data_20171125.csv ./
rm *.zip
rm -rf data/
cd ../src