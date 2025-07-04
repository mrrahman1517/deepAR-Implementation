def train_epoch(model, loader, opt, device="cpu"):
    model.train()
    total = 0
    for batch in loader:
        x, y = (t.to(device) for t in batch[:2])
        opt.zero_grad()
        #loss = model.loss(x, y)
        output = model(x)
        loss = model.lik.loss(output[:, -y.size(1):], y)
        #loss = model.lik.loss()
        loss.backward()
        opt.step()
        total += loss.item()*x.size(0)
    return total/len(loader.dataset)
