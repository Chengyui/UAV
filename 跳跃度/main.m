%%%先跳跃度剔除数据，后小波处理
seq = 1:1:19203
xa = [data(:,1),seq']
ya = [data(:,2),seq']
za = [data(:,3),seq']
row = [data(:,4),seq']
pitch = [data(:,5),seq']
yaw = [data(:,6),seq']

seq = seq'
mu = zeros(19203,12)
n  =19203
xa = sortrows(xa,1)
ya = sortrows(ya,1)
za = sortrows(za,1)
row = sortrows(row,1)
pitch = sortrows(pitch,1)
yaw = sortrows(yaw,1)
mu = [xa,ya,za,row,pitch,yaw]


%计算累积和
accu = [xa(:,1),ya(:,1),za(:,1),row(:,1),pitch(:,1),yaw(:,1)]
accu = cumsum(accu)
xk =[xa(:,1),ya(:,1),za(:,1),row(:,1),pitch(:,1),yaw(:,1)]

%计算mu
tic

  mu(:,1:2:11) = (accu + (n-seq).*xk)./seq

toc

%跳跃度
half = 9601
jump  = zeros(n,12)
jump(:,2:2:12) = mu(:,2:2:12)
jump(n,:) = mu(n,:)
jump(1:n-1,1:2:11) = mu(2:n,1:2:11)./mu(1:n-1,1:2:11)
pre_maxv = zeros(6,1)
pre_maxp = zeros(6,1)
for i = 1:6
    [pre_maxv(i),pre_maxp(i)] = max(jump(1:half,i*2-1))
end

late_maxv = zeros(6,1)
late_maxp = zeros(6,1)
for i = 1:6
     [late_maxv(i),late_maxp(i)] = max(jump(half:n,i*2-1))
end
plot(xa(pre_maxp(1,1):late_maxp(1,1)))
title("用跳跃度剔除后的x线加速度")
late_maxp = late_maxp+half
test = [1,2,3,4,5,6,7]
plot([0.1 4 7 19 17 19 25 31 34 45 52 61 64 76 87 101 116 141 181 240 446 503])

