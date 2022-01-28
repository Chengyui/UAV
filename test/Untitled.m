for i = 1:19203
    if sum(X(i,:))>5 || sum(X(i,:))<-5
        Y(i) = 1;
    end
end

        
    