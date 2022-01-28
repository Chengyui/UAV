 
for i = 1:6
% 获取噪声信号  
sig = va_acc(:,i)

% 信号的分解  
[c,l]=wavedec(sig,4,'db4');

%提取四层细节分量和近似分量
a1=appcoef(c,l,'db4',1);
d1=detcoef(c,l,1);
a2=appcoef(c,l,'db4',2);
d2=detcoef(c,l,2);
a3=appcoef(c,l,'db4',3);
d3=detcoef(c,l,3);
a4=appcoef(c,l,'db4',4);
d4=detcoef(c,l,4);

% 重构小波分解向量，其中第一、二层的细节分量被置零
dd1=zeros(size(d1));
dd2=zeros(size(d2)); 
c1=[a4; d4; d3; dd2; dd1];
aa1(:,i)=waverec(c1,l,'db4');
end
% 作图
subplot(2,3,1);
plot(va_acc(:,1));
hold on;
plot(aa1(:,1),'r')
hold off;
title('x轴线加速度')
subplot(2,3,2);
plot(va_acc(:,2));
hold on;
plot(aa1(:,2),'r')
hold off;
title('y轴线加速度')
subplot(2,3,3);
plot(va_acc(:,3));
hold on;
plot(aa1(:,3),'r')
hold off;
title('z轴线加速度')
subplot(2,3,4);
plot(va_acc(:,4));
hold on;
plot(aa1(:,4),'r')
hold off;
title('row速度')
subplot(2,3,5);
plot(va_acc(:,5));
hold on;
plot(aa1(:,5),'r')
hold off;
title('pitch速度')
subplot(2,3,6);
plot(va_acc(:,6));
hold on;
plot(aa1(:,6),'r')
hold off;
title('yaw速度')
suptitle('使用db4小波变换分解到3-4层')