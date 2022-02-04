
subplot(231)
plot(testPredict(:,1),'r')
hold on
plot(test_Y(:,1),'b')
hold off

subplot(232)
plot(testPredict(:,2),'r')
hold on
plot(test_Y(:,2),'b')
hold off

subplot(233)
plot(testPredict(:,3),'r')
hold on
plot(test_Y(:,3),'b')
hold off

subplot(234)
plot(testPredict(:,4),'r')
hold on
plot(test_Y(:,4),'b')
hold off

subplot(235)
plot(testPredict(:,5),'r')
hold on
plot(test_Y(:,5),'b')
hold off

subplot(236)
plot(testPredict(:,6),'r')
hold on
plot(test_Y(:,6),'b')
hold off
suptitle("标准归一化 epoch=10 adam seq-len=50 units=64,预测的结果")
