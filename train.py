import os
import torch as t
from tqdm import tqdm, trange
from model_zoo import BaseMode
from SpectrumDataset import train_dataset, X_test, y_test
from torch.utils.data import DataLoader


EPOCH = 10
BATCH_SIZE = 22
EPOCH_STEPS = train_dataset.data_len // 43

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = BaseMode(2000, n_class=43).cuda()
model.parameters_init()
model.get_loss()

optimizer = t.optim.Adam(model.parameters(),
                         lr=0.01,
                         betas=(0.9, 0.99),
                         weight_decay=0.0002)

for epoch in (range(EPOCH)):
    bar = trange(EPOCH_STEPS)
    for step in bar:
        for data in train_loader:
            train_data = t.cat(data[0], 0).cuda()
            train_labels = t.cat(data[1]).cuda().long()

            model.mode = 'train'
            preds = model(train_data)
            batch_loss = model.loss(preds, train_labels)

            optimizer.zero_grad()
            batch_loss.backward()
            # t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
            optimizer.step()
        bar.set_description("loss: {:.3f}".format(batch_loss))

    if not os.path.exists('./checkpoint'):
        os.makedirs('./checkpoint')
    t.save(model.state_dict(), './checkpoint/%03d_model.pth' % epoch)
    t.save({
        'optimizer': optimizer.state_dict(),
        'iter': 0,
        'epoch': epoch
    }, './checkpoint/%03d_optimizer.pth' % epoch)
    with t.no_grad():
        val_pred = t.argmax(model(t.from_numpy(X_test).cuda()))
        val_pred = val_pred.cpu().numpy()
        print("测试准确率：", float(sum(y_test==val_pred)) / y_test.shape[0])



