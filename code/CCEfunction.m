%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Produces Pesaran's (2006) common correlated effects mean group (CCE-MG) estimator and 
% standard error (equations 53 and 58, respectively) and pooled (CCE-P) estimator and standard
% error (equations 65 and 69, respectively).
% Pesaran, Hashem M. 2006. "Estimation and Inference in Large Heterogeneous Panels with a Multifactor 
%	Error Structure." Econometrica, 74 (4), 967-1012.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% INPUTS:
% X1 = (TxNxp) matrix of independent variables
% Y1 = (TxN) matrix dependent variable
% order = the order of a unit-specic trend to be removed from data. order=0 specifies that no trend is removed
% S = matrix that places each cross-section, i, unit into one of M groups
%	(M<N) (i.e., grouping each county into a specific state). This is used for
%	removing group rather than unit trends. Set S=0 if no grouping is
%	necessary
% demean = indicator for whether data should be demeaned first, to remove two-way fixed effects 
%	(0-no demeaning, 1-remove time FE, 2-remove unit FE, 3-remove both)
% mgfe = indicator for whether unit-specific regressions should include an intercept (1-yes, 0-no)

% OUTPUTS:
% betaCCEmg = (px1) vector of CCE-MG regression coefficients
% seCCEmg = (px1) vector of standard errors for CCE-MG regression coefficients
% betaCCEp = (px1) vector of CCE-P regression coefficients
% seCCEp = (px1) vector of CCE-P standard errors for CCE-P regression coefficients
% betaCCEi = (pxN) matrix of CCE-MG unit-specific regression coefficients
% M = (TxT) filtering matrix for CCE-P
% Mmg = (TxT) filtering matrix for CCE-MG



function [betaCCEmg, seCCEmg, betaCCEp, seCCEp, betaCCEi, M, Mmg] = CCEfunction(X1,Y1,order,S,demean,mgfe); 

[T,N,p]=size(X1);


%%%%% 1. Remove state-specific trends and demean
[Xdet,Ydet] = RemoveTimeTrends(X1,Y1,order,S);
% remove fixed effects
if demean==0
    Xdot=Xdet;
    Ydot=Ydet;
elseif demean==1
    [Xdot,Ydot]=CSDemean(Xdet,Ydet,1);
elseif demean==2
    [Xdot,Ydot]=TimeDemean(Xdet,Ydet,1);
elseif demean==3
    [Xdot,Ydot]=DemeanData(Xdet,Ydet);
end
X=Xdot;
Y=Ydot;


%%%%% 2. Estimation
%Create the H-bar matrix -a proxy for the factors- and the M-bar matrix from Pesaran (2006)
Zw=zeros(T,p+1);
Zw(:,1)=mean(Y1,2);
for j=1:p
    Zw(:,1+j)=mean(X1(:,:,j),2);
end

if demean>=2
    Hw=[ones(T,1) Zw];
elseif demean<2 
    Hw=[Zw];
end
M=eye(T) - Hw*inv(Hw'*Hw)*Hw';

if mgfe==1
    Hwmg=[ones(T,1) Zw];
elseif mgfe==0
    Hwmg=[Zw];
end
Mmg=eye(T) - Hwmg*inv(Hwmg'*Hwmg)*Hwmg'; 


%%%%% 2a. Mean Group Estimator
% beta-CCEmg
betaCCEi=zeros(p,N);
for i=1:N
    Xi=zeros(T,p);
    for j=1:p
        Xi(:,j)=Xdet(:,i,j); % Note - always sends the non-demeaned data into the heterogeneous estimator
    end
    Yi=Ydet(:,i);
    betaCCEi(:,i)=inv(Xi'*Mmg*Xi)*Xi'*Mmg*Yi;
end
betaCCEmg=mean(betaCCEi,2);

% standard error-CCEmg
squaredCCEmg=zeros(p,p,N);
for i=1:N
    squaredCCEmg(:,:,i)=(betaCCEi(:,i)-betaCCEmg)*(betaCCEi(:,i)-betaCCEmg)';
end
varcovCCEmg=(1/(N*(N-1)))*sum(squaredCCEmg,3);
seCCEmg=sqrt(diag(varcovCCEmg));


%%%%% 2b. Pooled Estimator
% beta - CCEp
XX=zeros(p,p,N);
XY=zeros(p,1,N);
for i=1:N
    Xi=zeros(T,p);
    for j=1:p
        Xi(:,j)=X(:,i,j);
    end;
    XX(:,:,i)=(1/N)*(Xi'*M*Xi);
end
for i=1:N
    Xi=zeros(T,p);
    for j=1:p
        Xi(:,j)=X(:,i,j);
    end
    Yi=Y(:,i);
    XY(:,:,i)=(1/N)*(Xi'*M*Yi);
end
XXsum=sum(XX,3);
XYsum=sum(XY,3);
betaCCEp=inv(XXsum)*XYsum;

% standard error - CCEp
simats=zeros(p,p,N);
for i=1:N
    Xi=zeros(T,p);
    for j=1:p
        Xi(:,j)=X(:,i,j);
    end;
    simats(:,:,i)=(1/N)*((Xi'*M*Xi)/T);
end
sihat=sum(simats,3);
Rmats=zeros(p,p,N);
for i=1:N
    Xi=zeros(T,p);
    for j=1:p
        Xi(:,j)=X(:,i,j);
    end
    Rmats(:,:,i)=(((1/N)/sqrt(inv(N)*N*(1/N)^2))^2)*((Xi'*M*Xi)/T)*(betaCCEi(:,i)-betaCCEmg)*(betaCCEi(:,i)-betaCCEmg)'*((Xi'*M*Xi)/T);
end
Rhat=(1/(N-1))*sum(Rmats,3);
AvarbCCEp=(N*((1/N)^2))*inv(sihat)*Rhat*inv(sihat);
seCCEp=sqrt(diag(AvarbCCEp));


betaCCEmg;
seCCEmg;

betaCCEp;
seCCEp;



