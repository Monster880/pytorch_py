from tensorboardX import SummaryWriter

writer = SummaryWriter("log")
for i in range(100):
    writer.add_scalar("a", i, global_step=i)
    writer.add_scalar("b", i ** 2, global_step=i)
writer.close()
