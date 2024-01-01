function [preds] = NB(a, training_data, test_data, training_labels, test_labels)
i1 = find(training_labels==1);
i2 = find(training_labels==2);
training_data1 = training_data(i1,:);
training_data2 = training_data(i2,:);

n = length(training_data);
[n1 x] = size(i1);
[n2 x] = size(i2);
S1 = cov(training_data1);
S2 = cov(training_data2);
m1 = mean(training_data1);
m2 = mean(training_data2);

total = n1+n2;
preds = zeros(192-n,1);
preds2 = zeros(192-n,1);
pc1 = n1/total;
pc2 = n2/total;

for i=1:192-n
    g1x = -total/2*log(2*pi)-log(det(S1))/2-1/2*(test_data(i,:)-m1)*inv(S1)*transpose(test_data(i,:)-m1)+log(pc1);
    g2x = -total/2*log(2*pi)-log(det(S2))/2-1/2*(test_data(i,:)-m2)*inv(S2)*transpose(test_data(i,:)-m2)+log(pc2);
    if g1x>g2x
        preds(i) = 1;
    else
        preds(i) = 2;
    end
end