training_data = readmatrix("dataset.xlsx");
training_labels = training_data(:,1);
training_data = training_data(:,2:end);
n = 150;
k=9;
a=100;
randn = randperm(192,n);
itest = zeros(192-n,1);

for i=1:192-n
    for j=1:192
        if ismember(j,randn)==0 & ismember(j,itest)==0
            itest(i) = j;
            break
        end
    end
end

test_data = training_data(itest,:);
training_data = training_data(randn,:);
test_labels = training_labels(itest,:);
training_labels = training_labels(randn,:);

i1 = find(training_labels>=a);
i2 = find(training_labels<a);
training_labels(i1) = 1;
training_labels(i2) = 2;
for i=1:192-n
    if test_labels(i)>=a
        test_labels(i)=1;
    else
        test_labels(i)=2;
    end
end

knn_preds = KNN(k, training_data, test_data, training_labels, test_labels);
nb_preds = NB(a,training_data, test_data, training_labels, test_labels);

similarity = sum(nb_preds==knn_preds)/(192-n)
nb_acc = sum(nb_preds==test_labels)/(192-n)
knn_acc = sum(knn_preds == test_labels)/(192-n)







