%**************************************************************************
% Program outline:
% 1. Estimate wage parameters via OLS
% 2. Estimate flexible mlogit and generate CCP's
% 3. Using CCP's and E[ln wage], estimate structural flow utility parameters
% 4. Compare estimates with DGP
%**************************************************************************
%==========================================================================
% Import data
%==========================================================================
clear all;
clc;
% global Beta

Beta = .80; %take beta lower in simulations so finite dependence has some purchase. leave it higher when estimating on real data because higher is more realistic
% Beta = 0;

if Beta>0
	delete estimation_simpleFD.diary;
	diary  estimation_simpleFD.diary;
    load data_SC_exper_wage_6_choices
	load trueCCPs
	load trueFVs
elseif Beta==0
	delete estimation_simple_staticFD.diary;
	diary  estimation_simple_staticFD.diary;
    load data_SC_exper_wage_6_choices_static
end
tic;

seed = 1234;
rng(seed,'twister');
Choice = Y;
[N,T]=size(Y);
J = 2*nloc;
% pretend panel is 5 periods instead of 35 (to reduce time horizon issues)
Y         = Y        (:,1:5);
LY        = LY       (:,1:5);
Choice    = Choice   (:,1:5);
exper     = exper    (:,1:5);
lnEarn    = lnEarn   (:,1:5);
obsloc    = obsloc   (:,1:5);
time      = time     (:,1:5);
Pmat = cat(5,p1,p2,p3,p4,p5,p6);
Pmat      = Pmat     (:,:,:,1:7,:);
Pmat0     = Pmat     (:,:,:,1:5,:);
Pmat1     = Pmat     (:,:,:,2:6,:);
Pmat2     = Pmat     (:,:,:,3:7,:);
FVmat     = fv       (:,:,:,1:7);
FVmat0    = fv       (:,:,:,1:5);
FVmat1    = fv       (:,:,:,2:6);
FVmat2    = fv       (:,:,:,3:7);
[N,T]     = size     (Y)    ;
longY     = Y        (:)    ;
longLY    = LY       (:)    ;
longExper = exper    (:)    ;
PmatLong  = reshape(permute(Pmat0,[1 4 2 3 5]),[N*T J T+3 J]);
FVmatLong = reshape(permute(FVmat0,[1 4 2 3]),[N*T J T+3]);
ID = [1:N]'*ones(1,T);
locID = Choice-(Choice>nloc)*nloc;
num_moves = zeros(N,1);
for i=1:N
	num_moves(i) = numel(unique(locID(i,:)))-1;
end
disp('Number of moves in the data:')
tabulate(num_moves);
baseAlt = 1;

% generate employment and location switching dummies
a  = 1*kron(1-eye(J/nloc),ones(nloc)); % employment switching matrix
am = 1*kron(ones(J/nloc),1-eye(nloc)); % location switching matrix

%==========================================================================
% Estimate wage parameters
%==========================================================================
earnflag = Choice<=nloc;
[bwage,sewage] = lscov([ones(size(exper(earnflag))) Choice(earnflag)==2 Choice(earnflag)==3 exper(earnflag) exper(earnflag).^2],lnEarn(earnflag))
Nwage = sum(sum(earnflag))
NwageUnique = numel(unique(ID(earnflag)))

for j=1:nloc
	k=j+nloc;
	Ewage(:,j) = ewage(nloc,j,exper(:),bwage);
	Ewage(:,k) = ewage(nloc,k,exper(:),bwage);
	SC(:,j)    = (LY(:)> nloc);
	SC(:,k)    = (LY(:)<=nloc);
	MC(:,j)    = (LY(:)~=j & LY(:)~=k);
	MC(:,k)    = (LY(:)~=j & LY(:)~=k);
end

%========================================================================
% Estimate flexible logit
%==========================================================================
Adj         = zeros(N*T,J);
AdjTrueCCPs = zeros(N*T,J);
if Beta>0
	%--------------------------------------------------------------------------
	% prepare data
	%--------------------------------------------------------------------------
	X = [ones(N*T,2)];
	% Z = [prev_diff_emp(:) prev_diff_loc(:)];
	for j=1:nloc
		k=j+nloc;
		Z(:,:,j) = [ewage(nloc,j,exper(:),bwage) (LY(:)> nloc) (LY(:)~=j & LY(:)~=k)];
		Z(:,:,k) = [ewage(nloc,k,exper(:),bwage) (LY(:)<=nloc) (LY(:)~=j & LY(:)~=k)];
	end
	Y = Choice(:);
	
	%--------------------------------------------------------------------------
	% estimate mlogit
	%--------------------------------------------------------------------------
	% restriction matrix
	restrMat1(1 ,:) = [1 7  1 1 0]; % amenities loc 2 (work)
	restrMat1(2 ,:) = [2 0  0 0 0]; % unemp benefits loc 2 (work)
	restrMat1(3 ,:) = [3 9  1 1 0]; % amenities loc 3 (work)
	restrMat1(4 ,:) = [4 0  0 0 0]; % unemp benefits loc 3 (work)
	restrMat1(5 ,:) = [5 0  0 0 0]; % amenities loc 1 (home)
	restrMat1(6 ,:) = [6 10 1 1 0]; % unemp benefits loc 1 (home)
	restrMat1(7 ,:) = [8 10 1 1 0]; % unemp benefits loc 2 (home)
	
	options = optimset('Disp','iter','LargeScale','on','maxiter',1e8,'maxfuneval',1e8,'TolX',1e-6,'Tolfun',1e-6,'DerivativeCheck','off','GradObj','on','FinDiffType','central');
	optionstest = optimset('Disp','iter','LargeScale','on','maxiter',1,'maxfuneval',1,'GradObj','off','DerivativeCheck','off');
	if exist('flexLogitStartVal.mat','file')==2
		load flexLogitStartVal b
		startval = b;
	else
		startval = rand(13,1);
	end
	restrMat1
	[b,l,e,o,g,h] = fminunc('clogit',startval,options,restrMat1,Y,X,Z,baseAlt);
	[b,invH] = applyRestr(restrMat1,b,h);
	[b sqrt(diag(invH))]
	save flexLogitStartVal b
	P = pclogit(b,Y,X,Z,baseAlt);
	summarize(P);
	tabulate(Y);
	% load flexLogitStartVal b
	%=========================================================================
	% Construct CCP's from data and mlogit coefficients
	%===========================================================================
	X = [ones(N*T,2)];
	% Need to loop over Z matrix in flexible logit to cover each set of states
	tic
	% Loop 1: Pr({0,3} t+1 | {jprime,lprime} t)
	% loop over all alternatives to store in FV matrix
	for k=1:J
		jp = k<=nloc;
		% loop over all t+1 alternatives
		for j=1:J 
			Ztemp(:,:,j) = [ewage(nloc,j,longExper+jp,bwage) ones(N*T,1)*a(k,j) ones(N*T,1)*am(k,j)];
		end
		% calculate CCPs
		P = pclogit(b,longY,X,Ztemp,baseAlt);
		% store log CCPs in the Adjustment term for the period t alternative (k)
		Adj(:,k) = -Beta*log(P(:,nloc+3)) + Adj(:,k);
		for i=1:N*T
			AdjTrueCCPs(i,k) = -Beta*log(PmatLong(i,k,exper(i)+jp+1,nloc+3)) + AdjTrueCCPs(i,k);
		end
		% P1(k) = P(1,nloc+3);
	end
	
	% Loop 2: Pr({0,3} t+2 | {0,3} t+1, {jprime,lprime} t)
	% loop over all alternatives to store in FV matrix
	for k=1:J
		jp = k<=nloc;
		lp = k-nloc*(k>nloc);
		% loop over all t+2 alternatives, but now in the Z's, fix k=nloc+lp (the t+1 decision)
		for j=1:J 
			Ztemp(:,:,j) = [ewage(nloc,j,longExper+jp,bwage) ones(N*T,1)*a(nloc+3,j) ones(N*T,1)*am(nloc+3,j)];
		end
		% calculate CCPs
		P = pclogit(b,longY,X,Ztemp,baseAlt);
		% store log CCPs in the Adjustment term for the period t alternative (k)
		Adj(:,k) = -Beta^2*log(P(:,nloc+3)) + Adj(:,k);
		for i=1:N*T
			AdjTrueCCPs(i,k) = -Beta^2*log(PmatLong(i,nloc+3,exper(i)+jp+1,nloc+3)) + AdjTrueCCPs(i,k);
		end
		% P2(k) = P(1,nloc+3);
	end
	
	% Loop 3: Pr({jprime,lprime} t+1 | {0,3} t)
	% loop over all alternatives to store in FV matrix
	for k=1:J
		jp = k<=nloc;
		% loop over all t+1 alternatives, fixing k=nloc+3 (the t decision)
		for j=1:J 
			Ztemp(:,:,j) = [ewage(nloc,j,longExper,bwage) ones(N*T,1)*a(nloc+3,j) ones(N*T,1)*am(nloc+3,j)];
		end
		% calculate CCPs
		P = pclogit(b,longY,X,Ztemp,baseAlt);
		% store log CCPs in the Adjustment term for the period t alternative (k)
		Adj(:,k) = Beta*log(P(:,k)) + Adj(:,k);
		for i=1:N*T
			AdjTrueCCPs(i,k) = Beta*log(PmatLong(i,nloc+3,exper(i)+1,k)) + AdjTrueCCPs(i,k);
		end
		% P3(k) = P(1,3+nloc*(k>nloc));
	end
	
	% Loop 4: Pr({0,3} t+2 | {jprime,3} t+1, {0,3} t)
	% loop over all alternatives to store in FV matrix
	for k=1:J
		jp = k<=nloc;
		% loop over all t+2 alternatives, but now in the Z's, fix k=3 or nloc+3 (the t+1 decision)
		for j=1:J 
			Ztemp(:,:,j) = [ewage(nloc,j,longExper+jp,bwage) ones(N*T,1)*a(k,j) ones(N*T,1)*am(k,j)];
		end
		% calculate CCPs
		P = pclogit(b,longY,X,Ztemp,baseAlt);
		% store log CCPs in the Adjustment term for the period t alternative (k)
		Adj(:,k) = Beta^2*log(P(:,nloc+3)) + Adj(:,k);
		for i=1:N*T
			AdjTrueCCPs(i,k) = Beta^2*log(PmatLong(i,k,exper(i)+jp+1,nloc+3)) + AdjTrueCCPs(i,k);
		end
		% P4(k) = P(1,nloc+3);
	end
	disp(['Loop took ',num2str(toc/60),' minutes']);
	% [longLY(1:5) Adj(1:5,:)]
	% save FVsimpleFD Adj
	summarize(Adj);
	summarize(AdjTrueCCPs);
end
%==========================================================================
% Estimate structural model
%==========================================================================
%---------------------------------------------------------------------------
% prepare data
%---------------------------------------------------------------------------
% Xtilde = (1-Beta).*[ones(N*T,2)];
% Z = [prev_diff_emp(:) prev_diff_loc(:)];
Zloc    = zeros(N*T,nloc,J);
ZlocFut = zeros(N*T,nloc,J);
Ztilde  = zeros(N*T,4,J);
for j=1:nloc
	k=j+nloc;
	if j~=J-nloc
		Zloc(:,j,j)         =  (1-Beta);
		Zloc(:,J-nloc,j)    = -(1-Beta);
		ZlocFut(:,j,j)      =  ( -Beta);
		ZlocFut(:,J-nloc,j) = -( -Beta);
	end
	if k~=J
		Zloc(:,j,k)         =  (1-Beta);
		Zloc(:,J-nloc,k)    = -(1-Beta);
		ZlocFut(:,j,k)      =  ( -Beta);
		ZlocFut(:,J-nloc,k) = -( -Beta);
	end
	if k==J
		Zloc(:,:,[j j+nloc]) = zeros(N*T,nloc,2);
	end
	Ztilde0a(:,:,j) = [benefits(nloc,j,exper(:)) ewage(nloc,j,exper(:),bwage) (LY(:)> nloc) (LY(:)~=j & LY(:)~=k)];
	Ztilde0a(:,:,k) = [benefits(nloc,k,exper(:)) ewage(nloc,k,exper(:),bwage) (LY(:)<=nloc) (LY(:)~=j & LY(:)~=k)];
	Ztilde1a(:,:,j) = [benefits(nloc,J,exper(:)) ewage(nloc,J,exper(:),bwage) ones(N*T,1)*a(j,J) ones(N*T,1)*am(j,J)];
	Ztilde1a(:,:,k) = [benefits(nloc,J,exper(:)) ewage(nloc,J,exper(:),bwage) ones(N*T,1)*a(k,J) ones(N*T,1)*am(k,J)];
	Ztilde2a(:,:,j) = [benefits(nloc,J,exper(:)) ewage(nloc,J,exper(:),bwage) ones(N*T,1)*a(J,J) ones(N*T,1)*am(J,J)];
	Ztilde2a(:,:,k) = [benefits(nloc,J,exper(:)) ewage(nloc,J,exper(:),bwage) ones(N*T,1)*a(J,J) ones(N*T,1)*am(J,J)];
	Ztilde0b(:,:,j) = [benefits(nloc,J,exper(:)) ewage(nloc,J,exper(:),bwage) (LY(:)<=nloc) (LY(:)~=(J-nloc) & LY(:)~=J)];
	Ztilde0b(:,:,k) = [benefits(nloc,J,exper(:)) ewage(nloc,J,exper(:),bwage) (LY(:)<=nloc) (LY(:)~=(J-nloc) & LY(:)~=J)];
	Ztilde1b(:,:,j) = [benefits(nloc,j,exper(:)) ewage(nloc,j,exper(:),bwage) ones(N*T,1)*a(J,j) ones(N*T,1)*am(J,j)]; 
	Ztilde1b(:,:,k) = [benefits(nloc,k,exper(:)) ewage(nloc,k,exper(:),bwage) ones(N*T,1)*a(J,k) ones(N*T,1)*am(J,k)];
	Ztilde2b(:,:,j) = [benefits(nloc,J,exper(:)) ewage(nloc,J,exper(:),bwage) ones(N*T,1)*a(j,J) ones(N*T,1)*am(j,J)];
	Ztilde2b(:,:,k) = [benefits(nloc,J,exper(:)) ewage(nloc,J,exper(:),bwage) ones(N*T,1)*a(k,J) ones(N*T,1)*am(k,J)];
end
Ztilde = Ztilde0a-Ztilde0b + Beta*(Ztilde1a-Ztilde1b) + Beta.^2*(Ztilde2a-Ztilde2b);
ZtildeFut = Beta*(Ztilde1a-Ztilde1b) + Beta.^2*(Ztilde2a-Ztilde2b);
Z = cat(2,Zloc,Ztilde);
Zfut = cat(2,ZlocFut,ZtildeFut);
Y = Choice(:);
assert(size(Zloc,2)==nloc,'Zloc incorrectly constructed');
% !mutt -s "estimation_simpleFD: CCPs and Flow Utilities Completed" tmr17@duke.edu
futureUtil = nan(N*T,J);
for j=1:J
	futureUtil(:,j) = Zfut(:,:,j)*[0;.5;1;2;.4;-2;-6];
end
trueFV = nan(N*T,J);
for i=1:N*T
	LYi = longLY(i);
	l = LYi-nloc*(LYi>nloc);
	trueFV(i,:) = [FVmatLong(i,1:nloc,exper(i)+2) FVmatLong(i,nloc+1:J,exper(i)+1)]-FVmatLong(i,end,exper(i)+1);
end
% disp('Compare first 10 observations of true FV and estimated FV');
% trueFV(1:10,:)-Adj(1:10,:)-futureUtil(1:10,:)
disp('Summary stats of trueFV minus estimated FV');
summarize(trueFV-Adj-futureUtil);
disp('Summary stats of trueFV minus estimated FV (with true CCPs)');
summarize(trueFV-AdjTrueCCPs-futureUtil);
% AdjOld = Adj;
% Adj = trueFV-futureUtil;
% disp('Summary stats of new Adj vs. trueFV minus future flow util');
% summarize(Adj-trueFV-futureUtil);
% disp('Summary stats of new Adj vs. old Adj');
% summarize(Adj-AdjOld);
% [AdjOld(1:10,:) Adj(1:10,:)]
% save FVcomparerSimpleFD3 AdjOld Adj futureUtil trueFV
% return
Adj = AdjTrueCCPs;
%--------------------------------------------------------------------------
% DGP answers
%--------------------------------------------------------------------------
% wage equation coefficients
true_bwage = [7.25; 0.10; 0.20; 0.06; -.0008];

% concatenate answers
truth = b_ans;

%--------------------------------------------------------------------------
% estimate structural likelihood
%--------------------------------------------------------------------------
% restriction matrix
restrMat(1,:) = [baseAlt 0  0 0 0]; % amenities in base location
% restrMat = []; % amenities in base location

options = optimset('Disp','iter','LargeScale','on','maxiter',1e8,'maxfuneval',1e8,'TolX',1e-6,'Tolfun',1e-6,'DerivativeCheck','off','GradObj','on','FinDiffType','central');
o4Nu=optimset('Disp','Iter','LargeScale','on','MaxFunEvals',0,'MaxIter',0,'TolX',1e-6,'Tolfun',1e-6,'GradObj','off','DerivativeCheck','off','FinDiffType','central');
o4An=optimset('Disp','Iter','LargeScale','on','MaxFunEvals',0,'MaxIter',0,'TolX',1e-6,'Tolfun',1e-6,'GradObj','on' ,'DerivativeCheck','off','FinDiffType','central');

truth = [0;.5;1;2;.4;-2;-6];
if Beta==0
	startval = truth
else
	startval = truth
end
derivative_checker = false;
if derivative_checker==true
	[bstruc0,lstruc,e,o,gNum]=fminunc('clogitAdj2',startval,o4Nu,restrMat,Y(:),[],Z,baseAlt,[],Adj,1);
	[bstruc0,lstruc,e,o,gAna]=fminunc('clogitAdj2',startval,o4An,restrMat,Y(:),[],Z,baseAlt,[],Adj,1);
	dlmwrite ('gradientChecker.csv',[gNum gAna gNum-gAna]);
	return
end
[bstruc,lstruc,~,~,~,hstruc] = fminunc('clogitAdj2',startval,options,[],Y(:),[],Z,baseAlt,[],Adj,1);
[bstruc,invHstruc] = applyRestr(restrMat,bstruc,hstruc);
[bstruc sqrt(diag(invHstruc))]
% [bstruc sqrt(diag(inv(full(hstruc))))]

%--------------------------------------------------------------------------
% evaluate simulation
%--------------------------------------------------------------------------
% test if estimates lie in 95% CI around true:
in_true_CI = abs((bstruc-truth)./sqrt(diag(invHstruc)))<1.96;

[bwage true_bwage]
[bstruc truth in_true_CI]

disp(['Estimation took ',num2str(toc/60),' minutes'])
diary off;