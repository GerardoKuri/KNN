
%Función la cual recibe un conjunto de entrenamiento etiquetado, un
%conjunto de evalución y la cantidad k de vecinos a considerar.
%Regresa una rreglo respectivo a las predicciones que hace por medio del
%algoritmo knn.

function class=KNN(train, test, k)
S=size(test,1);
cant=size(train,1);
class=[];
for i=1:S
    for j=1:cant
        D(i,j) = sqrt(sum(([test(i,1),test(i,3),test(i,4)] - [train(j,1),train(j,3),train(j,4)]).^2));
    end
    [x,I]=sort(D(i,:),'ascend');
    t=train(I,:);
    y=mean(t(1:k,5));
    if y>0.5
        class(i)=1;
    elseif y<0.5
        class(i)=0;
    else
        class(i)=t(1);
    end
end
end
