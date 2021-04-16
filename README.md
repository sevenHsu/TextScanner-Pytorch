## TextScanner
>Implementation of [TextScanner](https://arxiv.org/abs/1912.12422) with pytroch
>We trained this model on license plate number dataset([CCPD](https://github.com/detectRecog/CCPD)).
### Usage
   - dataset prepare
     > prepare the training dataset as the 'data' directory  
   - train
     > execute the 'scripts/train.sh'
   - test
     > execute the 'scripts/test.sh'
   - infer
     ```shell script
     python exp/infer.py --load_model_path checkpoints/txt_scan_res18_lr_1e-5_batch_16/best_val_error.pth
     ````
     ![1](data/images/0ffddba8-cc53-462c-9bc0-93a491f21819.jpg)
     ![2](data/images/480d8781-8303-46f1-8493-b7136300d4ec.jpg)
     ![3](data/images/6cd5cde4-9482-4ac7-8d8e-0b55fee4eb4c.jpg)
     ![4](data/images/78e8f654-b422-4ee2-a9cf-73e56c5e256f.jpg)
     ![5](data/images/8ceaf28a-5537-4fdd-9bc2-279877da2ab6.jpg)
     ![infer result](infer_result.jpg)
