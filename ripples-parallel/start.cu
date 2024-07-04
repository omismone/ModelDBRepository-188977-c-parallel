#include "omislib.cuh"

int main() {

    /* Randomize*/
    srand(time(NULL));

    struct pm p;
    p.CE = 200;     //pF
    p.glE = 10;     //nS
    p.ElE = -58;    //mV
    p.aE = 2;
    p.bE = 100;
    p.slpE = 2;     //mV
    p.twE = 120;    //ms
    p.VtE = -50;    //mV
    p.VrE = -46;    //mV

    p.CI = 200;     //pF
    p.glI = 10;     //nS
    p.ElI = -70;    //mV
    p.aI = 2;
    p.bI = 10;
    p.slpI = 2;     //mV 
    p.twI = 30;     //ms
    p.VtI = -50;    //mV
    p.VrI = -58;    //mV

    p.gnoiseE = 80;
    p.gnoiseI = 90;

    p.tauEr = 0.5;
    p.tauEd = 3.5;

    p.tauEIr = 0.9;
    p.tauEId = 3; //EPSP on inrn faster than on principal cells

    p.tauIr = 0.3;
    p.tauId = 2;  // Bartos et al 2002
    // BtoP is slower than BtoB
    p.tauIEr = 0.3;
    p.tauIEd = 3.5;

    p.gmaxII = 3 / 0.8;
    p.gmaxEI = 2 / 0.3;
    p.gmaxIE = 5 / 0.6;
    p.gmaxEE = p.gmaxEI / 10;//1on1 syn strength /percent synch efficacy

    p.VrevE = 0; // mV reversal of excitatory syn
    p.VrevI = -80;  // mV rev of inhibitory syn

    p.gvarEE = 0.01;
    p.gvarII = 0.01;

    p.gvarEI = 0.01;
    p.gvarIE = 0.01;

    p.DCstdI = 0.1;
    p.DCstdE = 0.1;

    p.Edc = 40;
    p.jmpE = 210;
    p.Idc = 180;
    p.jmpI = 700;

    p.seqsize = 10;
    p.dcbias = 2;

    /* Continuing initialization */
    double T = (double)2;    // simulation duration [s]
    int NE = 800;   // number of excitatory neurons
    int NI = 160;   // number of inhibitory neurons

    struct inpseq in;
    in.slp = 3; //[=]ms slope of current activation (bell shaped)
    double* starts;
    //inpseq_on = 1100:220:T*1000;
    int s1 = ceil(((1000 * T) - 1100) / 220);
    in.on.size = s1;
    in.on.val = (double*)malloc(in.on.size * sizeof(double));
    starts = (double*)malloc(in.on.size * sizeof(double));
    for (int i = 0; i < s1; i++) {
        in.on.val[i] = (double)(1100 + 220 * i);
        starts[i] = in.on.val[i] / 1000;
    }
    in.length = 50; // [=] ms sequence of input ends

    struct options opt;
    opt.nonoise = 0;
    opt.novar = 0;
    opt.noiseprc = 100;
    opt.storecurrs = 1;
    opt.seqassign = 1;

    p.dcbias = 2; // how much you want to bias the DC of cells that are selected to be in the sequence

    //int idc = 1;
    //char string[] = "example_1.mat";
    //printf("%s\n", string);

    NetworkRunSeqt(p, in, NE, NI, T, opt);

    /* free */
    free(in.on.val);

    printf("simulation done.\n");

    return 0;
}