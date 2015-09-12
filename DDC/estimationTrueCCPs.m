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
	delete estimationTrueCCPs.diary;
	diary  estimationTrueCCPs.diary;
    load data_SC_exper_wage_6_choices
	load trueCCPs
	load trueFVs
elseif Beta==0
	delete estimationStaticTrueCCPs.diary;
	diary  estimationStaticTrueCCPs.diary;
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
Adj = zeros(N*T,J);
if Beta>0
	% ========================================================================
	% Construct FV terms from data and true CCPs
	% ==========================================================================
	X = [ones(N*T,2)];
	% Need to loop over Z matrix in flexible logit to cover each set of states
	tic
	for i=1:N*T %:numel(longExper) % loop over individual-time observations
		LYi = longLY(i);
		l = LYi-nloc*(LYi>nloc);
		% Loop 1: Pr({0,lprime} t+1 | {jprime,lprime} t)
		% loop over all alternatives to store in FV matrix
		for k=1:J
			jp = k<=nloc;
			lp = k-nloc*(k>nloc);
			% store log CCPs in the Adjustment term for the period t alternative (k)
			% order of PmatLong: i,prev_decision,exper,curr_decision
			Adj(i,k) = -Beta*log(PmatLong(i,k,longExper(i)+1+jp,nloc+lp)) + Adj(i,k);
		end
		
		% Loop 2: Pr({0,l} t+2 | {0,lprime} t+1, {jprime,lprime} t)
		% loop over all alternatives to store in FV matrix
		for k=1:J
			jp = k<=nloc;
			lp = k-nloc*(k>nloc);
			% store log CCPs in the Adjustment term for the period t alternative (k)
			% order of PmatLong: i,prev_decision,exper,curr_decision
			Adj(i,k) = -Beta^2*log(PmatLong(i,nloc+lp,longExper(i)+1+jp,nloc+l)) + Adj(i,k);
		end
		
		% Loop 3: Pr({jprime,l} t+1 | {0,l} t)
		% loop over all alternatives to store in FV matrix
		for k=1:J
			jp = k<=nloc;
			% store log CCPs in the Adjustment term for the period t alternative (k)
			% order of PmatLong: i,prev_decision,exper,curr_decision
			Adj(i,k) = Beta*log(PmatLong(i,nloc+l,longExper(i)+1,l+nloc*(jp==0))) + Adj(i,k);
		end
		
		% Loop 4: Pr({0,l} t+2 | {jprime,l} t+1, {0,l} t)
		% loop over all alternatives to store in FV matrix
		for k=1:J
			jp = k<=nloc;
			% store log CCPs in the Adjustment term for the period t alternative (k)
			% order of PmatLong: i,prev_decision,exper,curr_decision
			Adj(i,k) = Beta^2*log(PmatLong(i,l+nloc*(jp==0),longExper(i)+1+jp,nloc+l)) + Adj(i,k);
		end
	end
	disp(['Loop took ',num2str(toc/3600),' hours']);
	[longLY(1:5) Adj(1:5,:)]
	save FVtrueCCPs Adj
	summarize(Adj);
end
%==========================================================================
% Estimate structural model
%==========================================================================
%---------------------------------------------------------------------------
% prepare data
%---------------------------------------------------------------------------
% Xtilde = (1-Beta).*[ones(N*T,2)];
% Z = [prev_diff_emp(:) prev_diff_loc(:)];
tic;
Zloc     = zeros(N*T,nloc,J);
ZlocFut  = zeros(N*T,nloc,J);
Ztilde   = zeros(N*T,4,J);
Ztilde0a = zeros(N*T,4,J);
Ztilde1a = zeros(N*T,4,J);
Ztilde2a = zeros(N*T,4,J);
Ztilde0b = zeros(N*T,4,J);
Ztilde1b = zeros(N*T,4,J);
Ztilde2b = zeros(N*T,4,J);
for i=1:N*T
	LYi = longLY(i);
	l = LYi-nloc*(LYi>nloc);
	for j=1:nloc
		lp = j;
		k=j+nloc;
		if j~=l
			Zloc(i,j,j)    =  (1+Beta);
			Zloc(i,l,j)    = -(1+Beta);
			ZlocFut(i,j,j) =  (  Beta);
			ZlocFut(i,l,j) = -(  Beta);
		end
		if k~=nloc+l
			Zloc(i,j,k)    =  (1+Beta);
			Zloc(i,l,k)    = -(1+Beta);
			ZlocFut(i,j,k) =  (  Beta);
			ZlocFut(i,l,k) = -(  Beta);
		end
		if k==nloc+l
			Zloc(i,:,[l l+nloc]) = zeros(1,nloc,2);
		end
		Ztilde0a(i,:,j) = [benefits(nloc,j      ,longExper(i)) ewage(nloc,j      ,longExper(i)  ,bwage) (LYi> nloc) (LYi~=j & LYi~=k)   ];
		Ztilde0a(i,:,k) = [benefits(nloc,k      ,longExper(i)) ewage(nloc,k      ,longExper(i)  ,bwage) (LYi<=nloc) (LYi~=j & LYi~=k)   ];
		Ztilde1a(i,:,j) = [benefits(nloc,lp+nloc,longExper(i)) ewage(nloc,lp+nloc,longExper(i)+1,bwage) a(j      ,lp+nloc) am(j      ,lp+nloc)];
		Ztilde1a(i,:,k) = [benefits(nloc,lp+nloc,longExper(i)) ewage(nloc,lp+nloc,longExper(i)  ,bwage) a(k      ,lp+nloc) am(k      ,lp+nloc)];
		Ztilde2a(i,:,j) = [benefits(nloc,l+nloc ,longExper(i)) ewage(nloc,l+nloc ,longExper(i)+1,bwage) a(lp+nloc,l+nloc ) am(lp+nloc,l+nloc )];
		Ztilde2a(i,:,k) = [benefits(nloc,l+nloc ,longExper(i)) ewage(nloc,l+nloc ,longExper(i)  ,bwage) a(lp+nloc,l+nloc ) am(lp+nloc,l+nloc )];
		
		Ztilde0b(i,:,j) = [benefits(nloc,l+nloc ,longExper(i)) ewage(nloc,l+nloc ,longExper(i)  ,bwage) (LYi<=nloc) (LYi~=l & LYi~=l+nloc)];
		Ztilde0b(i,:,k) = [benefits(nloc,l+nloc ,longExper(i)) ewage(nloc,l+nloc ,longExper(i)  ,bwage) (LYi<=nloc) (LYi~=l & LYi~=l+nloc)];
		Ztilde1b(i,:,j) = [benefits(nloc,l      ,longExper(i)) ewage(nloc,l      ,longExper(i)  ,bwage) a(l+nloc,l     ) am(l+nloc,l     )];
		Ztilde1b(i,:,k) = [benefits(nloc,l+nloc ,longExper(i)) ewage(nloc,l+nloc ,longExper(i)  ,bwage) a(l+nloc,l+nloc) am(l+nloc,l+nloc)];
		Ztilde2b(i,:,j) = [benefits(nloc,l+nloc ,longExper(i)) ewage(nloc,l+nloc ,longExper(i)+1,bwage) a(l     ,l+nloc) am(l     ,l+nloc)];
		Ztilde2b(i,:,k) = [benefits(nloc,l+nloc ,longExper(i)) ewage(nloc,l+nloc ,longExper(i)  ,bwage) a(l+nloc,l+nloc) am(l+nloc,l+nloc)];
	end
end
Ztilde = Ztilde0a-Ztilde0b + Beta*(Ztilde1a-Ztilde1b) + Beta.^2*(Ztilde2a-Ztilde2b);
ZtildeFut = Beta*(Ztilde1a-Ztilde1b) + Beta.^2*(Ztilde2a-Ztilde2b);
Z = cat(2,Zloc,Ztilde);
Zfut = cat(2,ZlocFut,ZtildeFut);
Y = Choice(:);
assert(size(Zloc,2)==nloc,'Zloc incorrectly constructed');
disp(['Loop took ',num2str(toc/60),' minutes']);
save flowUtilTrueCCPs Z* longLY longExper
% !mutt -s "estimation_simpleFD: CCPs and Flow Utilities Completed" tmr17@duke.edu
futureUtil = nan(N*T,J);
for j=1:J
	futureUtil(:,j) = Zfut(:,:,j)*[0;.5;1;2;.4;-2;-6];
end
trueFV = nan(N*T,J);
for i=1:N*T
	LYi = longLY(i);
	l = LYi-nloc*(LYi>nloc);
	trueFV(i,:) = [FVmatLong(i,1:nloc,exper(i)+2) FVmatLong(i,nloc+1:J,exper(i)+1)]-FVmatLong(i,l+nloc,exper(i)+1);
end
% disp('Compare first 10 observations of true FV and estimated FV');
% trueFV(1:10,:)-Adj(1:10,:)-futureUtil(1:10,:)
disp('Summary stats of trueFV minus estimated FV');
summarize(trueFV-Adj-futureUtil);
% disp('Summary stats of trueFV minus estimated FV (with true CCPs)');
% summarize(trueFV-AdjTrueCCPs-futureUtil);
save FVcomparerTrueCCPs trueFV Adj futureUtil Z* longLY longExper
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

options = optimset('Disp','iter','LargeScale','off','maxiter',1e8,'maxfuneval',1e8,'TolX',1e-6,'Tolfun',1e-6,'DerivativeCheck','off','GradObj','off','FinDiffType','central');
o4Nu=optimset('Disp','Iter','LargeScale','off','MaxFunEvals',0,'MaxIter',0,'TolX',1e-6,'Tolfun',1e-6,'GradObj','off','DerivativeCheck','off','FinDiffType','central');
o4An=optimset('Disp','Iter','LargeScale','off','MaxFunEvals',0,'MaxIter',0,'TolX',1e-6,'Tolfun',1e-6,'GradObj','on' ,'DerivativeCheck','off','FinDiffType','central');

truth = [0;.5;1;2;.4;-2;-6];
startval = truth;
% if Beta==0
	% startval = [0;.5;1;2;.4;-2;-6]
% else
	% startval = [0;2;4;2.5;.6;-2;-6]
% end
derivative_checker = false;
if derivative_checker==true
	[bstruc0,lstruc,e,o,gNum]=fminunc('clogitAdj1',startval,o4Nu,restrMat,Y(:),[],Z,baseAlt,[],Adj,1);
	[bstruc0,lstruc,e,o,gAna]=fminunc('clogitAdj1',startval,o4An,restrMat,Y(:),[],Z,baseAlt,[],Adj,1);
	dlmwrite ('gradientChecker.csv',[gNum gAna gNum-gAna]);
	return
end
[bstruc,lstruc,~,~,~,hstruc] = fminunc('clogitAdj1',startval,options,restrMat,Y(:),[],Z,baseAlt,[],Adj,1);
[bstruc,invHstruc] = applyRestr(restrMat,bstruc,hstruc);
[bstruc sqrt(diag(invHstruc))]

%--------------------------------------------------------------------------
% evaluate simulation
%--------------------------------------------------------------------------
% test if estimates lie in 95% CI around true:
in_true_CI = abs((bstruc-truth)./sqrt(diag(invHstruc)))<1.96;

[bwage true_bwage]
[bstruc truth in_true_CI]

disp(['Estimation took ',num2str(toc/60),' minutes'])
diary off;