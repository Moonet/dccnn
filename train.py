import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
#from utils import UnitNormClipper


def train(train_iter, dev_iter, test_iter, model, args):
    # clipper = UnitNormClipper()
    if args.cuda:
        print ("gpu")
        model.cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    #optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    see = list(model.parameters())
    # optimizer = torch.optim.Adam([  {'params': model._modules['embed'].parameters()},
    #                                     {'params': model._modules['senti_embed'].parameters()},
    #                                     {'params': model._modules['convs1'].parameters()},
    #                                     {'params': model._modules['senti_conv'].parameters()},
    #                                     {'params': model._modules['fc1'].parameters(), 'weight_decay': 0.01}
    #                                   ],
    #                                  lr=args.lr)
    # # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay)
    steps = 0
    model.train()
    batch_size = args.batch_size
    for epoch in range(args.epochs):
        print ("epoch: " + str(epoch) +"\n")
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(1)  # batch first, index align
            
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
        
            optimizer.zero_grad()
            # args.length = feature.data.size()[1]
            logit = model(feature, feature)


            # print('logit vector', logit.size())
            # print('target vector', target.size())
            loss = F.cross_entropy(logit, target)
            loss.backward()
            # model._modules['fc1'].apply(clipper)
            # torch.nn.utils.clip_grad_norm( model._modules['fc1'], max_norm = 3)

            optimizer.step()
            # model._modules['fc1'].weight.data.clamp_(-1, 1)
            see2 = list(model.parameters())
            # model._modules['fc1'].apply(clipper)
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.data[0],
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            # if steps % args.test_interval == 0:
            #      eval(dev_iter, model, args)
            #      if test_iter is not None:
            #          eval(test_iter, model, args)
            # if steps % args.save_interval == 0:
            #     if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
            #     save_prefix = os.path.join(args.save_dir, 'snapshot')
            #     save_path = '{}_steps{}.pt'.format(save_prefix, steps)
            #     torch.save(model, save_path)
        # model._modules['fc1'].apply(clipper)
        eval(dev_iter, model, args)
        # if epoch == 4:
        #     if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
        #     save_prefix = os.path.join(args.save_dir, 'snapshot')
        #     save_path = '{}_steps{}.pt'.format(save_prefix, steps)
        #     torch.save(model, save_path)
        #     ci = eval(test_iter, model, args)
        #     save_index(ci)
        if test_iter is not None:
            eval(test_iter, model, args)


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    correct_index = []
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logit = model(feature, feature)
        loss = F.cross_entropy(logit, target, size_average=False)
        t = logit.data.cpu().numpy()
        avg_loss += loss.data[0]
        pre = (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()
        corrects += pre
        correct_index = torch.max(logit, 1)[1].view(target.size()).data.cpu().numpy()
    size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    accuracy = 100.0 * corrects/size
    print ('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))

    model.train()
    return correct_index

def save_index(index):
    file = open('necnn_index.txt', 'w')
    for i in index:
        file.write("%s\n" % i)
    file.close()




def predict(text, model, text_field, label_feild):
    assert isinstance(text, str)
    model.eval()
    text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.data[0][0]+1]
