%% get data

%inp
inp.Edc = readmatrix("data/inp_Edc.txt");
inp.Idc = readmatrix("data/inp_Idc.txt");
inp.Etrace = readmatrix("data/inp_Etrace.txt");
inp.Itrace = readmatrix("data/inp_Itrace.txt");

%inpseq
inpseq.slp = readmatrix("data/inpseq_slp.txt");
inpseq.on = readmatrix("data/inpseq_on.txt");
inpseq.length = readmatrix("data/inpseq_length.txt");

%lfp
lfp = readmatrix("data/lfp.txt");

%NE
NE = readmatrix("data/NE.txt");

%NI
NI = readmatrix("data/NI.txt");

%T
T = readmatrix("data/T.txt");

%tsp_E
tsp_E.times = readmatrix("data/tsp_E_times.txt");
tsp_E.celln = readmatrix("data/tsp_E_celln.txt");

%tsp_I
tsp_I.times = readmatrix("data/tsp_I_times.txt");
tsp_I.celln = readmatrix("data/tsp_I_celln.txt");

%veg
veg.E = readmatrix("data/veg_E.txt");
veg.I = readmatrix("data/veg_I.txt");
veg.ne = readmatrix("data/veg_ne.txt");
veg.ni = readmatrix("data/veg_ni.txt");

%% plot data

rpick = 2; % which ripple to plot
% inputs, rastergram, lfp and filteredLFP
tout = 0:0.001:T;
[ripples,spcount,recruit,filtLFP] = countRipples(T,lfp,tsp_E,tsp_I,NE,NI,inpseq);
rpt = ripples.time(rpick); %[=]s
%disp(rpt)
rpl = ripples.length(rpick)/1000; %[=]s

figure(1)
plot(tout,veg.E,'-k',tout,veg.I,'-r')
title('Example voltage')
ylabel('mV')
xlabel('time [=] s');
%xlim([xl xr]);

xl = rpt-0.02;
xr = rpt+rpl+0.02;
k1 = find(tout>=xl,1);
k2 = find(tout>=xr,1);
X = k1:k2;
figure(2)
subplot(311)
plot(tout(X),inp.Itrace(1,X),'-r',tout(X),inp.Etrace(1,X),'-k');
xlim([xl xr]);
title('current input from CA3');
subplot(312)
plot(tsp_E.times,tsp_E.celln+NI,'+k',tsp_I.times,tsp_I.celln,'+r');
xlim([xl xr]);
subplot(313)
tk = tout(X);
l1 = lfp(X);
l2 = filtLFP(X);
%plot(tk,[l1; l2]')
% set(gca,'xlimmode','manual');
xlim([xl xr]);
plotyy(tout(X),lfp(X),tout(X),filtLFP(X))
title('LFP');
xlabel('time [=] s');

tx = 0:0.0005:T;
figure(3)
subplot(311)
hist(tsp_E.times,tx);
%xlim([3.5 3.6]);
subplot(312)
hist(tsp_I.times,tx);
%xlim([3.5 3.6]);
subplot(313)
plot(tout,filtLFP);
%xlim([3.5 3.6]);
