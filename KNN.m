function [preds] = KNN(k, training_data, test_data, training_labels, test_labels)
  
  n = length(test_data(:,1));
  preds = zeros(length(test_labels),1);

  dist = pdist2(test_data, training_data);
  for t = 1:n
    [vals indices] = sort(dist(t,:));
    topk = indices(1:k);
    labels = training_labels(topk);
    most = mode(labels);
    preds(t) = most;
  end
end