import torch
import os
import pandas as pd
from torch import nn, optim
from torch.utils.data import random_split, DataLoader, Dataset
from datetime import datetime
import wandb
import argparse

from pathlib import Path
BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
print(BASE_PATH, "!!!!!!!")


def get_data():
  train_dataset, validation_dataset, test_dataset = get_preprocessed_dataset()
  print(len(train_dataset), len(validation_dataset))

  train_data_loader = DataLoader(dataset=train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
  validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=len(validation_dataset))
  test_data_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))  # test data loader

  return train_data_loader, validation_data_loader, test_data_loader

class TitanicDataset(Dataset):
  def __init__(self, X, y):
    self.X = torch.FloatTensor(X) # 입력 데이터
    self.y = torch.LongTensor(y)  # 타겟 데이터

  def __len__(self):
    return len(self.X)  # 데이터셋 크기 반환

  def __getitem__(self, idx):  # 딕셔너리 형태로 입력과 타겟을 반환
    feature = self.X[idx]
    target = self.y[idx]
    return {'input': feature, 'target': target}

  def __str__(self):  # 데이터셋의 크기와 형상을 출력
    str = "Data Size: {0}, Input Shape: {1}, Target Shape: {2}".format(
      len(self.X), self.X.shape, self.y.shape
    )
    return str

class TitanicTestDataset(Dataset):
  def __init__(self, X):
    self.X = torch.FloatTensor(X)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    feature = self.X[idx]
    return {'input': feature}

  def __str__(self):
    str = "Data Size: {0}, Input Shape: {1}".format(
      len(self.X), self.X.shape
    )
    return str

def get_preprocessed_dataset():
    CURRENT_FILE_PATH = os.path.abspath('')  # ipynb에서 실행하기 위해 수정함
    train_data_path = os.path.join(CURRENT_FILE_PATH, "train.csv")
    test_data_path = os.path.join(CURRENT_FILE_PATH, "test.csv")

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    # 트레인과 테스트용 데이터 병합
    all_df = pd.concat([train_df, test_df], sort=False)

    # 데이터 전처리
    all_df = get_preprocessed_dataset_1(all_df)
    all_df = get_preprocessed_dataset_2(all_df)
    all_df = get_preprocessed_dataset_3(all_df)
    all_df = get_preprocessed_dataset_4(all_df)
    all_df = get_preprocessed_dataset_5(all_df)
    all_df = get_preprocessed_dataset_6(all_df)

    # 전처리된 데이터의 Survived 값이 있는 행을 학습용 데이터로 분리
    train_X = all_df[~all_df["Survived"].isnull()].drop("Survived", axis=1).reset_index(drop=True)
    train_y = train_df["Survived"]

    # Survived 값이 비어 있는 행을 테스트용 데이터로 분리
    test_X = all_df[all_df["Survived"].isnull()].drop("Survived", axis=1).reset_index(drop=True)

    dataset = TitanicDataset(train_X.values, train_y.values)
    train_dataset, validation_dataset = random_split(dataset, [0.8, 0.2])  # 학습 및 검증 데이터 분리
    test_dataset = TitanicTestDataset(test_X.values)

    return train_dataset, validation_dataset, test_dataset

def get_preprocessed_dataset_1(all_df):
    # Pclass별 Fare 평균값을 사용하여 Fare 결측치 메우기
    Fare_mean = all_df[["Pclass", "Fare"]].groupby("Pclass").mean().reset_index()
    Fare_mean.columns = ["Pclass", "Fare_mean"]
    all_df = pd.merge(all_df, Fare_mean, on="Pclass", how="left")
    all_df.loc[(all_df["Fare"].isnull()), "Fare"] = all_df["Fare_mean"]

    return all_df

def get_preprocessed_dataset_2(all_df):
    # name을 세 개의 컬럼으로 분리하여 다시 all_df에 합침
    name_df = all_df["Name"].str.split("[,.]", n=2, expand=True)
    name_df.columns = ["family_name", "honorific", "name"]
    name_df["family_name"] = name_df["family_name"].str.strip()
    name_df["honorific"] = name_df["honorific"].str.strip()
    name_df["name"] = name_df["name"].str.strip()
    all_df = pd.concat([all_df, name_df], axis=1)

    return all_df

def get_preprocessed_dataset_3(all_df):
    # honorific별 Age 평균값을 사용하여 Age 결측치 메우기
    honorific_age_mean = all_df[["honorific", "Age"]].groupby("honorific").median().round().reset_index()
    honorific_age_mean.columns = ["honorific", "honorific_age_mean", ]
    all_df = pd.merge(all_df, honorific_age_mean, on="honorific", how="left")
    all_df.loc[(all_df["Age"].isnull()), "Age"] = all_df["honorific_age_mean"]
    all_df = all_df.drop(["honorific_age_mean"], axis=1)

    return all_df

def get_preprocessed_dataset_4(all_df):
    # 가족수(family_num) 컬럼 새롭게 추가
    all_df["family_num"] = all_df["Parch"] + all_df["SibSp"]

    # 혼자탑승(alone) 컬럼 새롭게 추가
    all_df.loc[all_df["family_num"] == 0, "alone"] = 1
    all_df["alone"].fillna(0, inplace=True)

    # 학습에 불필요한 컬럼 제거
    all_df = all_df.drop(["PassengerId", "Name", "family_name", "name", "Ticket", "Cabin"], axis=1)

    return all_df

def get_preprocessed_dataset_5(all_df):
    # honorific 값 개수 줄이기
    all_df.loc[
    ~(
            (all_df["honorific"] == "Mr") |
            (all_df["honorific"] == "Miss") |
            (all_df["honorific"] == "Mrs") |
            (all_df["honorific"] == "Master")
    ),
    "honorific"
    ] = "other"
    all_df["Embarked"].fillna("missing", inplace=True)

    return all_df

def get_preprocessed_dataset_6(all_df):
    # 카테고리 변수를 LabelEncoder를 사용하여 수치값으로 변경하기
    category_features = all_df.columns[all_df.dtypes == "object"]
    from sklearn.preprocessing import LabelEncoder
    for category_feature in category_features:
        le = LabelEncoder()
        if all_df[category_feature].dtypes == "object":
          le = le.fit(all_df[category_feature])
          all_df[category_feature] = le.transform(all_df[category_feature])

    return all_df


class MyModel(nn.Module):
  def __init__(self, n_input, n_output):
    super().__init__()

    self.model = nn.Sequential(
      nn.Linear(n_input, wandb.config.n_hidden_unit_list[0]),
      nn.LeakyReLU(),
      nn.Linear(wandb.config.n_hidden_unit_list[0], wandb.config.n_hidden_unit_list[1]),
      nn.LeakyReLU(),
      nn.Linear(wandb.config.n_hidden_unit_list[1], n_output),
    )

  def forward(self, x):
    x = self.model(x)
    return x


def get_model_and_optimizer():
  my_model = MyModel(n_input=11, n_output=2)  # 이진 분류 문제이므로 출력은 2
  optimizer = optim.SGD(my_model.parameters(), lr=wandb.config.learning_rate)

  return my_model, optimizer


def training_loop(model, optimizer, train_data_loader, validation_data_loader, test_data_loader):
  n_epochs = wandb.config.epochs
  loss_fn = nn.CrossEntropyLoss()  # 이진 분류 문제이므로 CrossEntropyLoss 사용
  next_print_epoch = 100

  for epoch in range(1, n_epochs + 1):
    # 학습 단계
    model.train()
    loss_train = 0.0
    num_trains = 0
    for train_batch in train_data_loader:
      inputs, targets = train_batch['input'], train_batch['target']
      outputs = model(inputs)
      loss = loss_fn(outputs, targets)
      loss_train += loss.item()
      num_trains += 1

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # 검증 단계
    model.eval()
    loss_validation = 0.0
    num_validations = 0
    with torch.no_grad():
      for validation_batch in validation_data_loader:
        inputs, targets = validation_batch['input'], validation_batch['target']
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss_validation += loss.item()
        num_validations += 1

    # wandb 로그 기록
    wandb.log({
      "Epoch": epoch,
      "Training loss": loss_train / num_trains,
      "Validation loss": loss_validation / num_validations
    })

    # 특정 Epoch마다 테스트 데이터를 이용해 예측 수행 및 submission.csv 생성
    if epoch % 100 == 0:  # 예를 들어 100 에폭마다 테스트 예측 수행
      print(f"Performing test predictions at epoch {epoch}")
      generate_submission(model, test_data_loader)

    if epoch >= next_print_epoch:
      print(
        f"Epoch {epoch}, "
        f"Training loss {loss_train / num_trains:.4f}, "
        f"Validation loss {loss_validation / num_validations:.4f}"
      )
      next_print_epoch += 100

def generate_submission(model, test_data_loader):
    model.eval()  # 평가 모드 설정
    predictions = []

    with torch.no_grad():
        for batch in test_data_loader:
            inputs = batch['input']
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)  # 가장 높은 확률을 가진 클래스 선택
            predictions.extend(predicted.cpu().numpy())  # 예측 결과 수집

    # submission.csv 파일 생성
    submission_df = pd.DataFrame({
        "PassengerId": range(892, 892 + len(predictions)),  # 테스트 데이터 PassengerId는 892번부터 시작
        "Survived": predictions
    })
    submission_df.to_csv("submission.csv", index=False)
    print("submission.csv 생성 완료")

def main(args):
  current_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

  config = {
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'learning_rate': 1e-3,
    'n_hidden_unit_list': [20, 20],
  }

  wandb.init(
    mode="online" if args.wandb else "disabled",
    project="my_model_training",
    notes="Titanic model training with test evaluation",
    tags=["titanic", "classification"],
    name=current_time_str,
    config=config
  )
  print(args)
  print(wandb.config)

  train_data_loader, validation_data_loader, test_data_loader = get_data()

  model, optimizer = get_model_and_optimizer()

  print("#" * 50, 1)

  training_loop(
    model=model,
    optimizer=optimizer,
    train_data_loader=train_data_loader,
    validation_data_loader=validation_data_loader,
    test_data_loader=test_data_loader  # 테스트 데이터 로더 추가
  )
  
  wandb.finish()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--wandb", action=argparse.BooleanOptionalAction, default=False, help="True or False"
  )

  parser.add_argument(
    "-b", "--batch_size", type=int, default=512, help="Batch size (int, default: 512)"
  )

  parser.add_argument(
    "-e", "--epochs", type=int, default=1_000, help="Number of training epochs (int, default:1_000)"
  )

  args = parser.parse_args()

  main(args)
