for L=1:1
    for i=1:1
clear StackedESN;
close all;
clc;
alpha           = 0.8;
nLayers         = 4;
nInputDim       = 1;
nOutputDim      = 1;
nReservoirDim   = floor(1000/nLayers)*ones(1,nLayers);
nSpectraRadius  = [0.9,1.15*ones(1,nLayers)];
nSparseDegree   = 0.1+0.0*ones(1,nLayers);
delayorder      = 7+0*ones(nLayers,1);
SD_p            = 0.9;
%Acquire samples
nSamNum=3000;
nNeglectSampleNum=200;
fraction=2000/nSamNum;
scalefactor=[1.0,0.90*ones(1,nLayers)];
[Input_sequence_p,Output_sequence,nSamNum,PS]=generate_Smaples(nSamNum,17,0);
% plot(Output_sequence);
train_x=Input_sequence_p(1:floor(nSamNum*fraction),:);
train_y=Output_sequence(1:floor(nSamNum*fraction),:);
test_x=Input_sequence_p(floor(nSamNum*fraction)+1:nSamNum,:);
test_y=Output_sequence(floor(nSamNum*fraction)+1:nSamNum,:);
disp('building the network......');
tic
StackedESN= BuildStackedESN(nLayers,nInputDim,nOutputDim,...
    nReservoirDim,nSpectraRadius,nSparseDegree,SD_p,nSamNum);
disp('done......');
toc

disp('training network......');
tic
StackedESN=TrainStackedESN(train_x,train_y,StackedESN,nLayers,...
    nNeglectSampleNum,delayorder,scalefactor,alpha,1);
disp('done......');
toc
disp('testing network......');
tic
[StackedESN,PredictedOutput] = TestStackedESN(test_x,StackedESN,...
    nLayers,1,delayorder,scalefactor,alpha,1,nSamNum);
disp('done......');
toc
PredictedOutput=PredictedOutput(1:end);
test_y=test_y(1:end);
figure(1);
plot(PredictedOutput,'r-','LineWidth',1);
hold on;
plot(test_y,'b:','LineWidth',1);
xlabel('time step k');
ylabel('error');

set(gca,'FontSize',14,'FontName','TimeNewsRoman','FontWeight','bold');
set(get(gca,'YLabel'),'FontSize',14,'FontName','TimeNewsRoman','FontWeight','bold');
set(get(gca,'XLabel'),'FontSize',14,'FontName','TimeNewsRoman','FontWeight','bold');
set(gca,'linewidth',2);
PredictedError=(PredictedOutput-test_y);
PredictedRMSE=sqrt(sum(PredictedError(1:500).^2)/500)

figure(3);
plot(PredictedError,'r-.','LineWidth',1);
xlabel('time step k');
ylabel('error');

set(gca,'FontSize',14,'FontName','TimeNewsRoman','FontWeight','bold');
set(get(gca,'YLabel'),'FontSize',14,'FontName','TimeNewsRoman','FontWeight','bold');
set(get(gca,'XLabel'),'FontSize',14,'FontName','TimeNewsRoman','FontWeight','bold');
set(gca,'linewidth',2);
fid=fopen('NRMSED.txt','a+');
fprintf(fid,'%05.5f ',delayorder(1));
fprintf(fid,'%05.15f ',PredictedRMSE);
fprintf(fid,'%05.12f ',PredictedError(delayorder(1)*(nLayers-1)+84));
fprintf(fid,'%05.12f ',PredictedError(delayorder(1)*(nLayers-1)+120));
fprintf(fid,'\r\n');
fclose(fid);
    end
end

