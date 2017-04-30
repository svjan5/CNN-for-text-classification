clc
clear all
close all

load res_test.mat
load y_test.mat
load static_res.mat

y_test_org = y_test+1;

y_static = zeros(size(y_test_org));
y_nonstatic = zeros(size(y_test_org));
y_rand = zeros(size(y_test_org));
y_lstm = zeros(size(y_test_org));

for k = 1:length(y_test_org)
    [~,y_static(k)] = max(static_pred(k,:));
    [~,y_nonstatic(k)] = max(non_static(k,:));
    [~,y_rand(k)] = max(rand(k,:));
    [~,y_lstm(k)] = max(lstm(k,:));
end

y_pred = zeros(5,1);
for k = 1:5
    y_pred(k) = sum(y_test_org == k);
end


conf_mat_static = zeros(5);
conf_mat_nonstatic = zeros(5);
conf_mat_rand = zeros(5);
conf_mat_lstm = zeros(5);
for k = 1:length(y_test_org)
    conf_mat_static(y_test_org(k),y_static(k)) = conf_mat_static(y_test_org(k),y_static(k))+1/y_pred(y_test_org(k));
    conf_mat_nonstatic(y_test_org(k),y_nonstatic(k)) = conf_mat_nonstatic(y_test_org(k),y_nonstatic(k))+1/y_pred(y_test_org(k));
    conf_mat_rand(y_test_org(k),y_rand(k)) = conf_mat_rand(y_test_org(k),y_rand(k))+1/y_pred(y_test_org(k));
    conf_mat_lstm(y_test_org(k),y_lstm(k)) = conf_mat_lstm(y_test_org(k),y_lstm(k))+1/y_pred(y_test_org(k));
end

figure
imagesc(conf_mat_static)
ylabel('Actual class')
xlabel('Predicted class');
set(gca,'xTick',1:1:5)
set(gca,'xTickLabel',{'very negative', 'negative' , 'neutral' , 'positive' , 'very positive'})
set(gca,'yTick',1:1:5)
set(gca,'yTickLabel',{'very negative', 'negative' , 'neutral' , 'positive' , 'very positive'})
colorbar

figure
imagesc(conf_mat_nonstatic)
ylabel('Actual class')
xlabel('Predicted class');
set(gca,'xTick',1:1:5)
set(gca,'xTickLabel',{'very negative', 'negative' , 'neutral' , 'positive' , 'very positive'})
set(gca,'yTick',1:1:5)
set(gca,'yTickLabel',{'very negative', 'negative' , 'neutral' , 'positive' , 'very positive'})
colorbar

figure
imagesc(conf_mat_rand)
ylabel('Actual class')
xlabel('Predicted class');
set(gca,'xTick',1:1:5)
set(gca,'xTickLabel',{'very negative', 'negative' , 'neutral' , 'positive' , 'very positive'})
set(gca,'yTick',1:1:5)
set(gca,'yTickLabel',{'very negative', 'negative' , 'neutral' , 'positive' , 'very positive'})
colorbar

figure
imagesc(conf_mat_lstm)
ylabel('Actual class')
xlabel('Predicted class');
set(gca,'xTick',1:1:5)
set(gca,'xTickLabel',{'very negative', 'negative' , 'neutral' , 'positive' , 'very positive'})
set(gca,'yTick',1:1:5)
set(gca,'yTickLabel',{'very negative', 'negative' , 'neutral' , 'positive' , 'very positive'})
colorbar
