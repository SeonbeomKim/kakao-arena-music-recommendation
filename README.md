* 환경
    * python3.6
    * GPU: NVIDIA TITAN X
    * cuda: 10.0
    * cudnn: 7
    * ubuntu: 18.04
    
* pip list
    * tensorflow-gpu (1.14.0)
    * sentencepiece (0.1.91)
    * fire (0.3.1)
    * numpy (1.16.4)
    * tqdm (4.46.0)
    
* docker image
    * nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
    
1. 환경 설정
    *   clone 받은 repo로 이동
        * cd ./kakao-arena-music-recommendation
    *   export PYTHONPATH=$PWD
    
2. 데이터셋 다운로드
    *   train.json, val.json, test.json, song_meta.json 파일을 다운받아 dataset 폴더에 넣는다.
    
3. 데이터셋 전처리
    *     python3 data_loader/dump_datasets.py

4. 노래, 태그, 아티스트 기반의 추천 모델 학습
    *     python3 trainer/songs_tags_artists_model_trainer.py \
                --gpu={gpu number}
        ```
        args.add_argument('--bs', type=int, default=128)
        args.add_argument('--gpu', type=int, default=6)
        args.add_argument('--tags_loss_weight', type=float, default=0.15)
        args.add_argument('--artists_loss_weight', type=float, default=0.15)
        args.add_argument('--warmup_steps', type=float, default=4000)
        ```

4. 플레이리스트 타이틀 기반의 추천 모델 학습
    *     python3 trainer/plylst_title_model_trainer.py \
                --gpu={gpu number}
        ```
        args.add_argument('--bs', type=int, default=128)
        args.add_argument('--gpu', type=int, default=6)
        args.add_argument('--tags_loss_weight', type=float, default=0.15)
        args.add_argument('--warmup_steps', type=float, default=4000)
        ```

5. 두개의 추천 모델을 합쳐서 노래, 태그 추천
    *     python3 predictor/Ensemble.py \
                --gpu={gpu number} \
                --question_path={question_path} \
                --out_path={out_path}
        ```
        args.add_argument('--bs', type=int, default=128)
        args.add_argument('--gpu', type=int, default=6)
        args.add_argument('--title_importance', type=float, default=0.85)
        args.add_argument('--title_tag_weight', type=float, default=0.8)
        args.add_argument('--question_path', type=str, default='./dataset/val.json')
        args.add_argument('--out_path', type=str, default='./reco_result/results.json')
        ```
    * question_path
        *   리더보드용 validation set
        ```
        question_path=dataset/val.json
        ```
        *   offline test validation set 
        ```
        question_path=dataset/questions/val.json
        ```
        *   final test set
        ```
        question_path=dataset/test.json    
        ```
        
6. offline test validation set 채점하기
    *     python3 evaluate.py evaluate \
               --gt_fname=dataset/answers/val.json \
               --rec_fname=./reco_result/results.json    
    
* 대회 제출 포맷 코드 (0번 gpu로 실행되도록 처리)
    * 학습: python3 train.py
    * 추천: python3 inference.py
        * './reco_result/results.json' 파일 생성 
        
        
* 최종 제출한 모델
    * https://drive.google.com/file/d/1_4eStrTY3c96cigmgB8ydNg3Sqmz_y26/view?usp=sharing
    * 1~3 단계 수행 후 모델 파일을 clone 받은 kakao-arena-music-recommendation repo path에 삽입
    * 그 후 단계 5를 수행(원하는 입력에 대한 추천)하거나 inference.py를 수행(test 데이터에 대해서만 추천)하면 됨.
    
    * dataset/questions/val.json 기반 추천 결과 (dataset/answers/val.json 으로 채점)
        * music nDCG: 0.282904
        * Tag nDCG: 0.563226
        * Score: 0.324953    
    * dataset/val.json 기반 추천 결과 (리더보드)
        * music nDCG: 0.283642
        * Tag nDCG: 0.533388
        * Score: 0.321104
        