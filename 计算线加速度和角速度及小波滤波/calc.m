vl = zeros(19203,3);
acc = zeros(19203,3);
va = zeros(19203,3);
t = 0.01
for i = 2:19202
    vl(i,:) = (data(i+1,1:3)-data(i-1,1:3))/(2*t);
end
for i = 2:19202
    acc(i,:) = (vl(i+1,:)-vl(i-1,:))/(2*t);
end

for i = 2:19202
    va(i,:) = (data(i+1,4:6)-data(i-1,4:6))/(2*t);
end
figure 
subplot(2,3,1);
plot(acc(:,1));
hold on 
plot(newdata(:,1),'r');
title("x轴线加速度");


subplot(2,3,2)
plot(acc(:,2));
hold on 
plot(newdata(:,2),'r');
title("y轴线加速度");

subplot(2,3,3)
plot(acc(:,3));
hold on 
plot(newdata(:,3),'r');
title("z轴线加速度");


subplot(2,3,4)
plot(va(:,1));
hold on 
plot(newdata(:,4),'r');
title("row速度");

subplot(2,3,5)
plot(va(:,2));
hold on 
plot(newdata(:,5),'r');
title("pitch速度");

subplot(2,3,6)
plot(va(:,3));
hold on 
plot(newdata(:,6),'r');
title("yaw速度");

    