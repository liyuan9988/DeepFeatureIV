from torch.utils.data import Dataset, DataLoader

from src.data.data_class import TrainDataSetTorch


class DataSetTorchInterface(Dataset):

    def __init__(self, org_data: TrainDataSetTorch):
        self.org_data = org_data

    def __len__(self):
        return self.org_data.treatment.shape[0]

    def __getitem__(self, idx):
        sub_treatment = self.org_data.treatment[idx]
        sub_instrumental = self.org_data.instrumental[idx]
        sub_outcome = self.org_data.outcome[idx]
        sub_structural = self.org_data.structural[idx]
        if self.org_data.covariate is not None:
            sub_covariate = self.org_data.covariate[idx]
            return (sub_treatment, sub_instrumental, sub_outcome, sub_structural, sub_covariate)
        else:
            return (sub_treatment, sub_instrumental, sub_outcome, sub_structural)


class DataLoaderWrapper:

    def __init__(self, dataloader: DataLoader):
        self.dataloader = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        data_tmp = next(self.dataloader)
        sub_treatment, sub_instrumental, sub_outcome, sub_structural = data_tmp[:4]
        sub_covariate = None
        if len(data_tmp) == 5:
            sub_covariate = data_tmp[4]
        return TrainDataSetTorch(treatment=sub_treatment,
                                 instrumental=sub_instrumental,
                                 outcome=sub_outcome,
                                 structural=sub_structural,
                                 covariate=sub_covariate)



def get_minibatch_loader(train_data: TrainDataSetTorch, batch_size: int = 1000):
    data_obj = DataSetTorchInterface(train_data)
    return DataLoaderWrapper(DataLoader(data_obj, batch_size=batch_size, shuffle=True))
