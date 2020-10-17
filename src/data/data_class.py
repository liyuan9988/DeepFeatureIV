from typing import NamedTuple, Optional
import numpy as np
import torch


class TrainDataSet(NamedTuple):
    treatment: np.ndarray
    instrumental: np.ndarray
    covariate: Optional[np.ndarray]
    outcome: np.ndarray
    structural: np.ndarray


class TestDataSet(NamedTuple):
    treatment: np.ndarray
    covariate: Optional[np.ndarray]
    structural: np.ndarray


class TrainDataSetTorch(NamedTuple):
    treatment: torch.Tensor
    instrumental: torch.Tensor
    covariate: torch.Tensor
    outcome: torch.Tensor
    structural: torch.Tensor

    @classmethod
    def from_numpy(cls, train_data: TrainDataSet):
        covariate = None
        if train_data.covariate is not None:
            covariate = torch.tensor(train_data.covariate, dtype=torch.float32)
        return TrainDataSetTorch(treatment=torch.tensor(train_data.treatment, dtype=torch.float32),
                                 instrumental=torch.tensor(train_data.instrumental, dtype=torch.float32),
                                 covariate=covariate,
                                 outcome=torch.tensor(train_data.outcome, dtype=torch.float32),
                                 structural=torch.tensor(train_data.structural, dtype=torch.float32))

    def to_gpu(self):
        covariate = None
        if self.covariate is not None:
            covariate = self.covariate.cuda()
        return TrainDataSetTorch(treatment=self.treatment.cuda(),
                                 instrumental=self.instrumental.cuda(),
                                 covariate=covariate,
                                 outcome=self.outcome.cuda(),
                                 structural=self.structural.cuda())


class TestDataSetTorch(NamedTuple):
    treatment: torch.Tensor
    covariate: torch.Tensor
    structural: torch.Tensor

    @classmethod
    def from_numpy(cls, test_data: TestDataSet):
        covariate = None
        if test_data.covariate is not None:
            covariate = torch.tensor(test_data.covariate, dtype=torch.float32)
        return TestDataSetTorch(treatment=torch.tensor(test_data.treatment, dtype=torch.float32),
                                covariate=covariate,
                                structural=torch.tensor(test_data.structural, dtype=torch.float32))
    def to_gpu(self):
        covariate = None
        if self.covariate is not None:
            covariate = self.covariate.cuda()
        return TestDataSetTorch(treatment=self.treatment.cuda(),
                                covariate=covariate,
                                structural=self.structural.cuda())