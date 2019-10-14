from data import getdata
from model import Encoder, EncoderLayer, Decoder, DecoderLayer, PositionwiseFeedforward, SelfAttention, Seq2Seq
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def train(model, iterator, pad_idx, clip):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.023)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.English.transpose(0, 1)
        trg = batch.Czech.transpose(0, 1)

        optimizer.zero_grad()

        output = model(src, trg[:, :-1])

        # output = [batch size, trg sent len - 1, output dim]
        # trg = [batch size, trg sent len]

        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg sent len - 1, output dim]
        # trg = [batch size * trg sent len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, PAD_IDX):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.English.transpose(0, 1)
            trg = batch.Czech.transpose(0, 1)

            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def main():
    train_iterator, valid_iterator, input_dim, output_dim, pad_idx = getdata()
    hid_dim = 256
    n_layers = 6
    n_heads = 8
    pf_dim = 512
    dropout = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = Encoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, EncoderLayer, SelfAttention, PositionwiseFeedforward,
                  dropout, device)

    dec = Decoder(output_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward,
                  dropout, device)
    model = Seq2Seq(enc, dec, pad_idx, device).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    N_EPOCHS = 10
    CLIP = 1

    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train(model, train_iterator, pad_idx, CLIP)
        valid_loss = evaluate(model, valid_iterator, pad_idx)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model_transformer.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


if __name__ == '__main__':
    main()
