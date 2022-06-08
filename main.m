clear all;
close all;
clc;

DB=readmatrix('HeartDisease.csv');

%Normalización de datos
DB=minMaxNorm(DB);

%%
%Separación de base de datos en existencia o no existencia de enfermedad en
%historial familiar
[fh,I]=sort(DB(:,2));
DB=DB(I,:);
cant=size(find(DB(:,2)==0),1);
DBnfh=DB((1:cant),:);
DByfh=DB((cant+1:end),:);

%%
%Definición de parámetros
% %dobleces de k-fold
folds=5;
porcFoldSize=100/folds;
foldSizeyfh=floor(size(DByfh,1)/folds);
foldSizenfh=floor(size(DBnfh,1)/folds);
DByfhI=1:1:size(DByfh,1);
DBnfhI=1:1:size(DBnfh,1);
pruebas=1;
%K impar
K=-1;
%K par
%K=0;
%%
for j=1:50
    K=K+2;
     for i=1:folds
        %Definición de conjunto de entrenamiento y evaluación
        DByfhTestI=foldSizeyfh*(i-1)+1:foldSizeyfh*(i);
        DBnfhTestI=foldSizenfh*(i-1)+1:foldSizenfh*(i);
        DByfhTrainI=setxor(DByfhTestI,DByfhI);
        DBnfhTrainI=setxor(DBnfhTestI,DBnfhI);
        DBnfhTest=DBnfh(DBnfhTestI,:);
        DByfhTest=DByfh(DByfhTestI,:);
        DBnfhTrain=DBnfh(DBnfhTrainI,:);
        DByfhTrain=DByfh(DByfhTrainI,:);
        %%
        %Evaluación de conjunto de evaluación
        [cy]=KNN(DByfhTrain,DByfhTest,K);
        [cn]=KNN(DBnfhTrain,DBnfhTest,K);
        
        confMatYFHTest=confMatHD(cy,DByfhTest(:,5));
        confMatNFHTest=confMatHD(cn,DBnfhTest(:,5));
        [Acc,Pre,Sen,F1Sc]=MatEval(confMatYFHTest);
        modelYFH.accuracy(i)=Acc;
        modelYFH.precision(i)=Pre;
        modelYFH.sensitivity(i)=Sen;
        modelYFH.f1score(i)=F1Sc;
        [Acc,Pre,Sen,F1Sc]=MatEval(confMatNFHTest);
        modelNFH.accuracy(i)=Acc;
        modelNFH.precision(i)=Pre;
        modelNFH.sensitivity(i)=Sen;
        modelNFH.f1score(i)=F1Sc;
    end
    k(j)=K;
    modelNFH.ACC(j)=mean(modelNFH.accuracy);
    modelNFH.PRE(j)=mean(modelNFH.precision);
    modelNFH.SEN(j)=mean(modelNFH.sensitivity);
    modelNFH.F1Sc(j)=mean(modelNFH.f1score);
    modelYFH.ACC(j)=mean(modelYFH.accuracy);
    modelYFH.PRE(j)=mean(modelYFH.precision);
    modelYFH.SEN(j)=mean(modelYFH.sensitivity);
    modelYFH.F1Sc(j)=mean(modelYFH.f1score);
end

x=1:2:size(modelNFH.ACC,2);
fy=polyfit(k,modelYFH.ACC,4);
fn=polyfit(k,modelNFH.ACC,4);
yy = polyval(fy,k);
yn = polyval(fn,k);

subplot(121)
plot(k,modelYFH.ACC,"*");
hold on;
plot(k,yy);
title("k impar vs accuracy con historial familiar")
xlabel('vecinos considerados');
ylabel('tasa de clasificación');
ylim([.5 .8]);
subplot(122)
plot(k,modelNFH.ACC,"*");
hold on;
plot(k,yn);
title("k impar vs accuracy sin historial familiar")
xlabel('vecinos considerados');
ylabel('tasa de clasificación');
ylim([.5 .8]);








