* pip list
    * tensorflow-gpu (1.14.0)
    * sentencepiece (0.1.91)
    * fire (0.3.1)
    * numpy (1.17.0)
    * tqdm (4.46.0)
    
1. 데이터셋 다운로드
    *   train.json, val.json, test.json, song_meta.json 파일을 다운받아 dataset 폴더에 넣는다.

2. 데이터셋 준비
    *   data_loader/dump_datasets.py 실행

3. 노래, 태그 기반의 추천 모델 학습
    *   trainer/songs_tags_autoencoder.py 실행
    ```
    args.add_argument('--bs', type=int, default=128)
    args.add_argument('--gpu', type=int, default=6)
    args.add_argument('--tags_loss_weight', type=float, default=0.15)
    args.add_argument('--negative_loss_weight', type=float, default=1.0)
    args.add_argument('--warmup_steps', type=float, default=4000)
    ```

4. 플레이리스트 타이틀 기반의 추천 모델 학습
    *   trainer/plylst_title_model.py 실행 (3의 결과물이 필요)
    ```
    args.add_argument('--bs', type=int, default=128)
    args.add_argument('--gpu', type=int, default=6)
    args.add_argument('--tags_loss_weight', type=float, default=0.5)
    args.add_argument('--negative_loss_weight', type=float, default=1.0)
    args.add_argument('--warmup_steps', type=float, default=4000)
    ```

5. 두개의 추천 모델을 합쳐서 추천
    *   predictor/Ensemble.py 실행
    ```
    args.add_argument('--question', required=True)
    args.add_argument('--fout', required=True)
    args.add_argument('--bs', type=int, default=128)
    args.add_argument('--gpu', type=int, default=6)
    ```
    * question
        *   리더보드용 
        ```
        question=dataset/val.json
        ```
        *   validation set 
        ```
        question=dataset/questions/val.json
        ```    
        
    * ex)
    ```
    python predictor/Ensemble.py --question=dataset/questions/val.json --fout=reco_result/result.json
    ```        
    
6. validation set 채점하기
    ```
    fout: predictor/Ensemble.py에서 입력한 fout
    python evaluate.py evaluate --gt_fname=dataset/answers/val.json --rec_fname=fout/result.json
    ```