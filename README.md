# [NLP]  KoBART를 사용한 텍스트 요약 모델

#### 관련 연구 동향 및 논문 리서치

최근 텍스트 정보의 기하급수적인 증가로 인해 사용자는 원하는 정보를 찾는데 더욱 많은 시간을 투자하여 해당 정보를 일일히 확인하여 찾아야 하는 불편함이 가중되고 있다. 
이러한 문제를 해결하기 위해 정보 검색, 텍스트 마이닝을 이용한 문서 요약에 대한 관심이 고조되면서 이에 대한 연구개발 투자와 많은 연구들이 수행되고 있다. 또한 영상 및 텍스트 분야의 빠른 콘텐츠 소비 트렌드가 반영된 숏 폼 콘텐츠(Short-Form contents)의 수요가 급증하며 문서 요약의 중요도가 점점 높아지는 추세이다.
관련 기술로는 Text Rank, Attention mechanism, KoBART, koBERT, BERT with copy mechanism 등이 있다.


본프로젝트는 텍스트 요약에 특화된 성능을 띄는 `KoBART` 모델을 기반으로하여, 병원 전화상담센터에서 고객과의 통화 내용을 효율적으로 요약할 수 있는 모델을 만들고, 성능을 개선하는 것을 목표로 두고 있다.

***

## 1. Data
- [Dacon 한국어 문서 생성요약 AI 경진대회](https://dacon.io/competitions/official/235673/overview/description)의 학습데이터를 활용함
- [AI hub의 문서요약텍스트](https://aihub.or.kr/aidata/8054) 데이터를 활용함
- [AI hub의 한국어대화요약](https://aihub.or.kr/aidata/30714) 데이터를 활용함
- 학습 데이터에서 임의로 Train/Test 데이터를 생성함
- 데이터 탐색에 용이하게 csv 형태로 데이터를 변환함
- Data 구조: **train set 크기 (test set 크기)**
  - Dacon 신문기사: 34,242 (8,501)
  - AI hub 문서요약텍스트: 321,052 (40,132)
     - 법률문서: 24,029 (3,004)
     - 신문기사: 240,968 (30,121)
     - 사설잡지: 56,055 (7,007)
  - AI hub 한국어대화요약: 280,000 (35,000)
     - 개인및관계: 71,408 (8,926)
     - 미용과건강: 16,520 (2,065)
     - 상거래(쇼핑): 25,480 (3,185)
     - 시사교육: 14,840 (1,855)
     - 식음료: 29,400 (3,675)
     - 여가생활: 35,840 (4,480)
     - 일과직업: 20,152 (2,519)
     - 주거와 생활: 45,640 (5,705)
     - 행사: 20,720 (2,590)
- default로 Data/Training/train_v1.csv, data/Validation/dev_v1.csv 형태로 저장        
- 데이터 열 이름을 text와 abstractive로 통일

|text|abstractive|
|:----:|:-----------:|
|문서원문|요약문|

***

## 2. Hyper-parameter tuning

- Batch size
  - trainer 생성시 `auto_scale_batch_size='power'`로 설정
  - 2, 4, 8, 16, 32..등 batch sizee 늘려가며 훈련 시도
  - 최적의 batch size 탐색 후 16으로 사용           
    <img width="600" src="https://user-images.githubusercontent.com/44887886/131794480-3320f7bd-53f1-4c1a-849f-2738926755a3.png">

- Learing rate
  - 자동으로 초기 최적의 학습률을 찾아주는 lr finder 함수 사용
  - default learing rate: 3e-5   
  - 가장 낮은 손실의 학습률이 아니라 가장 급격한 하향 기울기를 가진 위치의 학습률 선택                        
  
  <img width="800" aligh='left' src="https://user-images.githubusercontent.com/44887886/131793752-796da55e-08ce-4a09-a017-b2956c65b1ad.png">

- Num workers
  - cpu 작업에 사용할 코어의 개수
  - 해당 환경에서 사용가능한 코어 개수 확인 후 적절하게 10으로 설정
  
***

## 3. 함수 구현

- Early Stopping
  - 학습 도중 validation loss가 특정 지점 이후 계속 올라감에도 불구하고 설정한 max epoch까지 학습이 진행 -> **overfitting**        
    <img width="550" src="https://user-images.githubusercontent.com/44887886/131795045-925b7ca4-1adc-438b-962b-a7007a871c1b.png">
  - Epoch 하나 끝날 때마다 val_loss 계산
  - 연속으로 10번 이상 val_loss 증가 시 train 중단     
     
- Test_step 및 Test_epoch_end  
  - test_step: 각 배치의 평균 루지 스코어를 구한 후 반환
  - test_epoch_end: epoch 하나가 끝나기 전에, test_step 함수에서 반환된 평균 루지스코어들의 평균을 계산하여 반환
  - 모델의 최종 성능은 **rouge.csv** 파일의 형태로 저장    

***

## 4. How To Train
- 실행 환경: gpu 2개 사용 (multi-gpu)
- 모델 새로 생성
```
!python src/KoBART/train.py \
--train_file='rsc/Data/Training/AI_hub/한국어대화요약/train_v1.csv' \
--test_file='rsc/Data/Validation/AI_hub/한국어대화요약/dev_v1.csv' \
--mode='train' \
--batch_size=16 \
--num_workers=10 \
--gradient_clip_val=1.0 \
--gpus=2 \
--accelerator=ddp \
--max_epochs=50 \
--default_root_dir='rsc/By_domain_ckpt/aihub한국어대화요약'
```
- 모델 추가 학습
```
!python src/KoBART/train.py \
--train_file='rsc/Data/Training/AI_hub/한국어대화요약/train_v1.csv' \
--test_file='rsc/Data/Validation/AI_hub/한국어대화요약/dev_v1.csv' \
--checkpoint_path='rsc/By_domain_ckpt/dacon신문기사/kobart_summary-model_chp/epoch=02-val_loss=1.326.ckpt' \
--hparams_file='rsc/By_domain_ckpt/dacon신문기사/tb_logs/default/version_0/hparams.yaml' \
--mode='train' \
--batch_size=16 \
--num_workers=10 \
--gradient_clip_val=1.0 \
--gpus=2 \
--accelerator=ddp \
--max_epochs=50 \
--default_root_dir='rsc/By_domain_ckpt/aihub한국어대화요약_dacon신문기사'
```

***

## 5. Visualization

<img width="650" src="https://user-images.githubusercontent.com/44887886/131796878-0a9ebc7a-a1e5-4361-8879-775244be4eaf.png">

