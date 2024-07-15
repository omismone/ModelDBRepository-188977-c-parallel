#include "omislib.cuh"

#define ENOISE_BIN_PATH1 ".\\bin\\Enoise1.bin"
#define INOISE_BIN_PATH1 ".\\bin\\Inoise1.bin"
#define ENOISE_BIN_PATH2 ".\\bin\\Enoise2.bin"
#define INOISE_BIN_PATH2 ".\\bin\\Inoise2.bin"


/*
*   @see https://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution
*/
double RandNormal()
{
    double x = 0, y = 0;
    while (x == 0 || y == 0) {
        x = (double)rand() / (double)RAND_MAX;
        y = (double)rand() / (double)RAND_MAX;
    }
    double z = sqrt(-2 * log(x)) * cos(2 * PI * y);

    return z;
}

/*
*   compare function (descending order) for quick sort
*/
static int AscendCmp(const void* a, const void* b) {
    double c = (*(double*)b - *(double*)a);
    if (c > 0)
        return -1;
    else if (c < 0)
        return 1;
    else
        return 0;
}

/*
*   compare function (ascending order) for quick sort
*/
static int DescendCmp(const void* a, const void* b) {
    double c = (*(double*)b - *(double*)a);
    if (c < 0)
        return -1;
    else if (c > 0)
        return 1;
    else
        return 0;
}

/*
*   calculate the transposed of a matrix.
*
*   @param mat: the matrix.
*/
void Transpose(struct matrix* mat) {
    struct matrix res;
    res.size[0] = mat->size[1];
    res.size[1] = mat->size[0];
    res.val = (double*)malloc(res.size[0] * res.size[1] * sizeof(double));
    for (int i = 0; i < mat->size[0]; i++) {
        for (int j = 0; j < mat->size[1]; j++) {
            res.val[j * mat->size[0] + i] = mat->val[i * mat->size[1] + j];
        }
    }
    *mat = res;
}

void NetworkRunSeqt(clock_t tic_start, struct pm p, struct inpseq inps, int NE, int NI, double T, struct options opt) {
    /*outputs*/
    //struct Conn conn;
    struct Vbar vbar;
    struct Veg veg;
    struct vec lfp;
    struct Tsp tspE;
    struct Tsp tspI;
    struct Isynbar isynbar;
    struct Inp inp;
    double* seqs;

    int max_size_tsp = 15000;
    tspE.times.size = max_size_tsp; // max size
    tspE.times.val = (double *)malloc(tspE.times.size * sizeof(double));
    tspE.celln.size = max_size_tsp; // max size
    tspE.celln.val = (double*)malloc(tspE.celln.size * sizeof(double));
    int tspE_count = 0;

    tspI.times.size = max_size_tsp; // max size
    tspI.times.val = (double*)malloc(tspI.times.size * sizeof(double));
    tspI.celln.size = max_size_tsp; // max size
    tspI.celln.val = (double*)malloc(tspI.celln.size * sizeof(double));
    int tspI_count = 0;

    /* pre select the sequence */
    int* NEseq;
    int nL = 10;
    NEseq = (int*)malloc(nL * sizeof(int));
    seqs = (double*)malloc(nL * sizeof(double));
    for (int i = 0; i < nL; i++) {
        NEseq[i] = ((rand() % 80) + 1) + (i * 80);
        seqs[i] = NEseq[i];
    }
    qsort(seqs, nL, sizeof(double), AscendCmp);

    /* opt options: nonoise novar noiseprc */
    p.gnoiseE = opt.nonoise ? 0 : p.gnoiseE * (opt.noiseprc / 100);
    p.gnoiseI = opt.nonoise ? 0 : p.gnoiseI * (opt.noiseprc / 100);

    double* Edc_dist;
    Edc_dist = (double*)malloc(NE * sizeof(double));
    double Edc_dist_max = log(0); //= -inf
    for (int i = 0; i < NE; i++) {
        Edc_dist[i] = (RandNormal() * p.DCstdE * p.Edc) + p.Edc;
        Edc_dist_max = Edc_dist[i] > Edc_dist_max ? Edc_dist[i] : Edc_dist_max;
    }
    double* Idc_dist;
    Idc_dist = (double*)malloc(NI * sizeof(double));
    for (int i = 0; i < NI; i++) {
        Idc_dist[i] = (RandNormal() * p.DCstdI * p.Idc) + p.Idc;
    }

    double* w;
    w = (double*)malloc(nL * sizeof(double));
    for (int i = 0; i < nL; i++) {
        w[i] = RandNormal();
    }
    qsort(w, nL, sizeof(double), DescendCmp);
    for (int i = 0; i < nL; i++) {
        Edc_dist[NEseq[i]] = (Edc_dist_max + (p.dcbias * p.DCstdE * p.Edc)) * ((double)1 + (w[i] * (double)0.02));
    }
    inp.Edc = Edc_dist; //on matlab is transposed
    inp.Idc = Idc_dist;

    free(w);


    /* inputs */
    struct matrix MX; //matrix to sum only some incoming inputs of CA3
    MX.size[0] = 100;
    MX.size[1] = NE;
    MX.val = (double*)malloc(MX.size[0] * MX.size[1] * sizeof(double));
    for (int i = 0; i < MX.size[0]; i++) {
        for (int j = 0; j < MX.size[1]; j++) {
            MX.val[i * MX.size[1] + j] = 0;
        }
    }
    int kkS;
    for (int i = 0; i < nL; i++) {
        kkS = ceil(((float)(NEseq[i] - 1) / NE) * 100) - 1;
        MX.val[kkS * MX.size[1] + (NEseq[i] - 1)] = 1;
    }
    Transpose(&MX);  // after it will be used transposed
    free(NEseq);

    /* synapses */
    //printf("wire ntwk - ");
    //clock_t tic = clock();
    if (opt.novar) {
        p.gvarEE = 0;
        p.gvarII = 0;
        p.gvarEI = 0;
        p.gvarIE = 0;
    }

    double mn = p.gmaxEE / NE;
    double vr = p.gvarEE * mn;
    struct matrix GEE;
    GEE.size[0] = NE;
    GEE.size[1] = NE;
    GEE.val = (double*)malloc(GEE.size[0] * GEE.size[1] * sizeof(double));
    double g;
    for (int i = 0; i < GEE.size[0]; i++) {
        for (int j = 0; j < GEE.size[1]; j++) {
            g = RandNormal() * sqrt(vr) + mn;
            GEE.val[i * GEE.size[1] + j] = g > 0 ? g : 0;
        }
    }

    //conn.EtoE = GEE;
    Transpose(&GEE); // after is used only transposed

    mn = p.gmaxII / NI;
    vr = p.gvarII * mn;
    struct matrix GII;
    GII.size[0] = NI;
    GII.size[1] = NI;
    GII.val = (double*)malloc(GII.size[0] * GII.size[1] * sizeof(double));
    for (int i = 0; i < GII.size[0]; i++) {
        for (int j = 0; j < GII.size[1]; j++) {
            g = RandNormal() * sqrt(vr) + mn;
            GII.val[i * GII.size[1] + j] = g > 0 ? g : 0;
        }
    }

    //conn.ItoI = GII;

    mn = p.gmaxEI / NE;
    vr = p.gvarEI * mn;
    struct matrix GEI;
    GEI.size[0] = NE;
    GEI.size[1] = NI;
    GEI.val = (double*)malloc(GEI.size[0] * GEI.size[1] * sizeof(double));
    for (int i = 0; i < GEI.size[0]; i++) {
        for (int j = 0; j < GEI.size[1]; j++) {
            g = RandNormal() * sqrt(vr) + mn;
            GEI.val[i * GEI.size[1] + j] = g > 0 ? g : 0;
        }
    }

    //conn.EtoI = GEI;

    mn = p.gmaxIE / NI;
    vr = p.gvarIE * mn;
    struct matrix GIE;
    GIE.size[0] = NI;
    GIE.size[1] = NE;
    GIE.val = (double*)malloc(GIE.size[0] * GIE.size[1] * sizeof(double));
    for (int i = 0; i < GIE.size[0]; i++) {
        for (int j = 0; j < GIE.size[1]; j++) {
            g = RandNormal() * sqrt(vr) + mn;
            GIE.val[i * GIE.size[1] + j] = g > 0 ? g : 0;
        }
    }

    //conn.ItoE = GIE;
    Transpose(&GIE); // after is used only transposed

    //clock_t toc = clock();
    //printf("elapsed time is %.2lf seconds.\n", (double)(toc - tic) / CLOCKS_PER_SEC);


    /* initialize sim */

    // time
    double dt = 0.001; // [=]ms integration step

    // allocate simulation ouput
    int s = (ceil(1000 / dt) - 1) / 1000 + 1 + 1000 * (T - 1) + 1;
    vbar.E = (double*)malloc(s * sizeof(double));
    vbar.I = (double*)malloc(s * sizeof(double));
    veg.E = (double*)malloc(s * sizeof(double));
    veg.I = (double*)malloc(s * sizeof(double));
    lfp.size = s;
    lfp.val = (double*)malloc(s * sizeof(double));
    for (int i = 0; i < s; i++) {
        vbar.E[i] = 0;
        vbar.I[i] = 0;
        veg.E[i] = 0;
        veg.I[i] = 0;
        lfp.val[i] = 0;
    }

    // t = 0:dt:1000
    struct vec t; // one sec time axis 
    s = ceil(1000 / dt);
    t.size = s;
    t.val = (double*)malloc(s * sizeof(double));
    for (int i = 0; dt * i <= 1000; i++) {
        t.val[i] = dt * i;
    }

    // peak values of biexps signals
    double pvsE = exp((1 / (1 / p.tauEd - 1 / p.tauEr) * log(p.tauEd / p.tauEr)) / p.tauEr) - exp((1 / (1 / p.tauEd - 1 / p.tauEr) * log(p.tauEd / p.tauEr)) / p.tauEd);
    double fdE = exp(-dt / p.tauEd); //factor of decay
    double frE = exp(-dt / p.tauEr); //factor of rise

    double pvsI = exp((1 / (1 / p.tauId - 1 / p.tauIr) * log(p.tauId / p.tauIr)) / p.tauIr) - exp((1 / (1 / p.tauId - 1 / p.tauIr) * log(p.tauId / p.tauIr)) / p.tauId);
    double fdI = exp(-dt / p.tauId);
    double frI = exp(-dt / p.tauIr);

    double pvsIE = exp((1 / (1 / p.tauIEd - 1 / p.tauIEr) * log(p.tauIEd / p.tauIEr)) / p.tauIEr) - exp((1 / (1 / p.tauIEd - 1 / p.tauIEr) * log(p.tauIEd / p.tauIEr)) / p.tauIEd);
    double fdIE = exp(-dt / p.tauIEd);
    double frIE = exp(-dt / p.tauIEr);

    double pvsEI = exp((1 / (1 / p.tauEId - 1 / p.tauEIr) * log(p.tauEId / p.tauEIr)) / p.tauEIr) - exp((1 / (1 / p.tauEId - 1 / p.tauEIr) * log(p.tauEId / p.tauEIr)) / p.tauEId);
    double fdEI = exp(-dt / p.tauEId);
    double frEI = exp(-dt / p.tauEIr);

    int Eeg = (rand() % NE);
    veg.ne = Eeg;
    int Ieg = (rand() % NI);
    veg.ni = Ieg;

    s = (ceil(1000 / dt) - 1) / 1000 + 1 + 1000 * (T - 1) + 1;
    isynbar.ItoE.size[0] = NE;
    isynbar.ItoE.size[1] = s;
    isynbar.ItoE.val = (double*)malloc(isynbar.ItoE.size[0] * isynbar.ItoE.size[1] * sizeof(double));
    isynbar.EtoE.size[0] = NE;
    isynbar.EtoE.size[1] = s;
    isynbar.EtoE.val = (double*)malloc(isynbar.EtoE.size[0] * isynbar.EtoE.size[1] * sizeof(double));
    if (opt.storecurrs) {
        for (int i = 0; i < isynbar.ItoE.size[0]; i++) {
            for (int j = 0; j < isynbar.ItoE.size[1]; j++) {
                isynbar.ItoE.val[i * isynbar.ItoE.size[1] + j] = 0;
            }
        }
        for (int i = 0; i < isynbar.EtoE.size[0]; i++) {
            for (int j = 0; j < isynbar.EtoE.size[1]; j++) {
                isynbar.EtoE.val[i * isynbar.EtoE.size[1] + j] = 0;
            }
        }
    }

    s = ceil(1 / dt) + 1;
    struct matrix Einptrace;
    Einptrace.size[0] = NE;
    Einptrace.size[1] = s;
    Einptrace.val = (double*)malloc(Einptrace.size[0] * Einptrace.size[1] * sizeof(double));
    for (int i = 0; i < Einptrace.size[0]; i++) {
        for (int j = 0; j < Einptrace.size[1]; j++) {
            Einptrace.val[i * Einptrace.size[1] + j] = 0;
        }
    }

    s = ceil(1 / dt) + 1;
    struct matrix Iinptrace;
    Iinptrace.size[0] = NI;
    Iinptrace.size[1] = s;
    Iinptrace.val = (double*)malloc(Iinptrace.size[0] * Iinptrace.size[1] * sizeof(double));
    for (int i = 0; i < Iinptrace.size[0]; i++) {
        for (int j = 0; j < Iinptrace.size[1]; j++) {
            Iinptrace.val[i * Iinptrace.size[1] + j] = 0;
        }
    }

    /* for each second of simulation we generate new noise */

    /* the noise is taken by a binary file */

    int seqN = 1;
    struct matrix Enoise;
    Enoise.size[0] = NE;
    Enoise.size[1] = (int)ceil(1000 / dt) + 1;
    struct matrix Inoise;
    Inoise.size[0] = NI;
    Inoise.size[1] = (int)ceil(1000 / dt) + 1;
    Enoise.val = (double*)malloc(Enoise.size[0] * Enoise.size[1] * sizeof(double));
    Inoise.val = (double*)malloc(Inoise.size[0] * Inoise.size[1] * sizeof(double));

    // for the initialization of the first instant of simulation
    struct vec vE, wE, sE, sEI, erE, edE, erEI, edEI;
    vE.size = NE;
    wE.size = NE;
    sE.size = NE;
    sEI.size = NE;
    erE.size = NE;
    edE.size = NE;
    erEI.size = NE;
    edEI.size = NE;

    vE.val = (double *)malloc(NE * sizeof(double));
    wE.val = (double*)malloc(NE * sizeof(double));
    sE.val = (double*)malloc(NE * sizeof(double));
    sEI.val = (double*)malloc(NE * sizeof(double));
    erE.val = (double*)malloc(NE * sizeof(double));
    edE.val = (double*)malloc(NE * sizeof(double));
    erEI.val = (double*)malloc(NE * sizeof(double));
    edEI.val = (double*)malloc(NE * sizeof(double));

    struct vec vI, wI, sI, sIE, erI, edI, erIE, edIE;
    vI.size = NI;
    wI.size = NI;
    sI.size = NI;
    sIE.size = NI;
    erI.size = NI;
    edI.size = NI;
    erIE.size = NI;
    edIE.size = NI;

    vI.val = (double *)malloc(NI * sizeof(double));
    wI.val = (double*)malloc(NI * sizeof(double));
    sI.val = (double*)malloc(NI * sizeof(double));
    sIE.val = (double*)malloc(NI * sizeof(double));
    erI.val = (double*)malloc(NI * sizeof(double));
    edI.val = (double*)malloc(NI * sizeof(double));
    erIE.val = (double*)malloc(NI * sizeof(double));
    edIE.val = (double*)malloc(NI * sizeof(double));


    double tmin, tmax;
    struct vec stsec;
    stsec.size = inps.on.size; //max size, then realloc
    stsec.val = (double*)malloc(stsec.size * sizeof(double));

    int L = inps.length - inps.slp - 2;
    int L0 = 0; // inps.slp/inps.length
    int L1 = 1; // - L0

    double step = (double)1 / 99;
    int size = ceil((L1 - L0) / step) + 1;
    double* tbins, * tons, * toffs;
    tbins = (double*)malloc(size * sizeof(double));
    tons = (double*)malloc(size * sizeof(double));
    toffs = (double*)malloc(size * sizeof(double));

    struct matrix bmps;
    double* bt, * ebt;
    double ebt_max = 0;
    double bt_max = 0;
    bmps.size[0] = 100;
    bmps.size[1] = t.size;
    bmps.val = (double*)malloc(bmps.size[0] * bmps.size[1] * sizeof(double));
    bt = (double*)malloc(t.size * sizeof(double));
    ebt = (double*)malloc(t.size * sizeof(double));

    struct matrix tmp;
    tmp.size[0] = MX.size[0];
    tmp.size[1] = bmps.size[1];
    tmp.val = (double*)malloc(tmp.size[0] * tmp.size[1] * sizeof(double));

    struct matrix AEX, AIX;
    AEX.size[0] = NE;
    AEX.size[1] = bmps.size[1];
    AIX.size[0] = NI;
    AIX.size[1] = bmps.size[1];
    AEX.val = (double*)malloc(AEX.size[0] * AEX.size[1] * sizeof(double));
    AIX.val = (double*)malloc(AIX.size[0] * AIX.size[1] * sizeof(double));

    while (seqN <= T) {
        // reading the noise, to have it in a binary file @see ./scripts/SaveNoise.m 
        //printf("[reading noise]\n");

        FILE* fp = NULL;
        if (seqN == 1)
            fp = fopen(ENOISE_BIN_PATH1, "rb");
        else
            fp = fopen(ENOISE_BIN_PATH2, "rb");
        if (fp == NULL) {
            if(seqN == 1)
                perror("error while opening noise file \"Enoise1.bin\"");
            else if (seqN == 2)
                perror("error while opening noise file \"Enoise2.bin\"");
            else
                perror("error while opening noise files");

            exit(1);
        }
        int numElements = fread(Enoise.val, sizeof(double), Enoise.size[0] * Enoise.size[1], fp);
        if (numElements < Enoise.size[0] * Enoise.size[1]) {
            if (ferror(fp)) {
                if (seqN == 1)
                    perror("error while reading Enoise1.bin");
                else if (seqN == 2)
                    perror("error while reading Enoise2.bin");
                else
                    perror("error while reading noises files");
            }
            else {
                printf("[[warning: EOF before reading all the noises]]\n");
            }
            fclose(fp);
            exit(1);
        }
        fclose(fp);

        if (seqN == 1)
            fp = fopen(INOISE_BIN_PATH1, "rb");
        else
            fp = fopen(INOISE_BIN_PATH2, "rb");
        if (fp == NULL) {
            if (seqN == 1)
                perror("error while opening noise file \"Inoise1.bin\"");
            else if (seqN == 2)
                perror("error while opening noise file \"Inoise2.bin\"");
            else
                perror("error while opening noise files");

            exit(1);
        }
        numElements = fread(Inoise.val, sizeof(double), Inoise.size[0] * Inoise.size[1], fp);
        if (numElements < Inoise.size[0] * Inoise.size[1]) {
            if (ferror(fp)) {
                if (seqN == 1)
                    perror("error while reading Inoise1.bin");
                else if (seqN == 2)
                    perror("error while reading Inoise2.bin");
                else
                    perror("error while reading noises files");
            }
            else {
                printf("[[warning: EOF before reading all the noises]]\n");
            }
            fclose(fp);
            exit(1);
        }
        fclose(fp);

        //printf("integrating ODE\n");
        if (seqN == 1) {
            for (int i = 0; i < NE; i++) {
                vE.val[i] = ((double)rand() / (double)RAND_MAX) * (70 + p.VrE) - 70;
                wE.val[i] = p.aE * (vE.val[i] - p.ElE);
                sE.val[i] = 0;
                sEI.val[i] = 0;
                erE.val[i] = 0;
                edE.val[i] = 0;
                erEI.val[i] = 0;
                edEI.val[i] = 0;
            }

            for (int i = 0; i < NI; i++) {
                vI.val[i] = ((double)rand() / (double)RAND_MAX) * (70 + p.VrI) - 70;
                wI.val[i] = p.aI * (vI.val[i] - p.ElI);
                sI.val[i] = 0;
                sIE.val[i] = 0;
                erI.val[i] = 0;
                edI.val[i] = 0;
                erIE.val[i] = 0;
                edIE.val[i] = 0;
            }
        }
        else {
            tmin = (seqN - 1) * 1000 - 100;
            tmax = seqN * 1000 + 20;
            stsec.size = 0;
            for (int i = 0; i < inps.on.size; i++) {
                if (inps.on.val[i] >= tmin && inps.on.val[i] < tmax) {
                    if (i == inps.on.size - 1) // note that in the matlab version this part is wrong (so here)
                        break;
                    stsec.val[stsec.size] = inps.on.val[i];
                    stsec.size++;
                }
            }
            stsec.val = (double*)realloc(stsec.val, stsec.size * sizeof(double));

            for (int j = 0; j < t.size; j++) {
                for (int i = 0; i < 100; i++) {
                    bmps.val[i * bmps.size[1] + j] = 0;
                }
                bt[j] = 0;
                ebt[j] = 0;
            }

            for (int i = 0; i < stsec.size; i++) {
                // inside each ripple event
                stsec.val[i] = stsec.val[i] - ((seqN - 1) * 1000);
                double rplton = stsec.val[i];
                double rpltoff = rplton + inps.length;
                for (int k = 0; k < size; k++) {
                    tbins[k] = (L0 + (k * step)) * L;
                    tons[k] = rplton + inps.slp + 2 + tbins[k]; // start the Ecells bumps after the I cells are inhibiting already
                    toffs[k] = tons[k] + (inps.length / 99);
                }
                for (int j = 0; j < bmps.size[1]; j++) {
                    for (int k = 0; k < bmps.size[0]; k++) {
                        bmps.val[k * bmps.size[1] + j] = bmps.val[k * bmps.size[1] + j] + (1 / (1 + exp((tons[k] - t.val[j]) / 1.5)) * 1 / (1 + exp((t.val[j] - toffs[k]) / 1.5)));
                    }
                    bt[j] = bt[j] + ((1 / (1 + exp((rplton - t.val[j]) / inps.slp))) * (1 / (1 + exp((t.val[j] - rpltoff) / inps.slp))));
                    ebt[j] = ebt[j] + ((1 / (1 + exp((rplton - t.val[j]) / (inps.slp / 2)))) * (1 / (1 + exp((t.val[j] - rpltoff) / (inps.slp / 2)))));
                    ebt_max = ebt_max > ebt[j] ? ebt_max : ebt[j];
                    bt_max = bt_max > bt[j] ? bt_max : bt[j];
                }

            }


            double* dev_MX = 0;
            double* dev_bmps = 0;
            double* dev_tmp = 0;
            cublasHandle_t handle;
            cudaSetDevice(0);

            cublasCreate(&handle);

            cudaMalloc((void**)&dev_MX, MX.size[0] * MX.size[1] * sizeof(double));
            cudaMalloc((void**)&dev_bmps, bmps.size[0] * bmps.size[1] * sizeof(double));
            cudaMalloc((void**)&dev_tmp, tmp.size[0] * tmp.size[1] * sizeof(double));

            cudaMemcpy(dev_MX, MX.val, MX.size[0] * MX.size[1] * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_bmps, bmps.val, bmps.size[0] * bmps.size[1] * sizeof(double), cudaMemcpyHostToDevice);

            const double alpha = 1.0;
            const double beta = 0.0;
            int m = bmps.size[1];
            int n = MX.size[0];
            int k = bmps.size[0];
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, dev_bmps, n, dev_MX, k, &beta, dev_tmp, n);

            cudaMemcpy(tmp.val, dev_tmp, tmp.size[0] * tmp.size[1] * sizeof(double), cudaMemcpyDeviceToHost);

            cublasDestroy(handle);
            cudaFree(dev_MX);
            cudaFree(dev_bmps);
            cudaFree(dev_tmp);

            for (int k = 0; k < AEX.size[0]; k++) {
                for (int j = 0; j < AEX.size[1]; j++) {
                    AEX.val[k * AEX.size[1] + j] = 5 * tmp.val[k * tmp.size[1] + j] + ebt[j];
                }
            }

            for (int k = 0; k < AIX.size[0]; k++) {
                for (int j = 0; j < AIX.size[1]; j++) {
                    AIX.val[k * AIX.size[1] + j] = bt[j];
                }
            }

            double Escale = ebt_max;
            double ff;
            ff = Escale > 0 ? p.jmpE / Escale : 0;
            for (int k = 0; k < Enoise.size[0]; k++) {
                for (int j = 0; j < Enoise.size[1]; j++) {
                    Enoise.val[k * Enoise.size[1] + j] = Enoise.val[k * Enoise.size[1] + j] + (ff * AEX.val[k * AEX.size[1] + j]);
                }
            }
            double Iscale = bt_max;
            double gg;
            gg = Iscale > 0 ? p.jmpI / Iscale : 0;
            for (int k = 0; k < Inoise.size[0]; k++) {
                for (int j = 0; j < Inoise.size[1]; j++) {
                    Inoise.val[k * Inoise.size[1] + j] = Inoise.val[k * Inoise.size[1] + j] + (gg * AIX.val[k * Inoise.size[1] + j]);
                }
            }

            int step = 1 / dt;
            Einptrace.size[1] += AEX.size[1] / step; // should be always 1000 (so += AEX.size[1] / step) or lower if dt is bigger? (so += step)
            Einptrace.val = (double*)realloc(Einptrace.val, Einptrace.size[0] * Einptrace.size[1] * sizeof(double));
            for (int k = 0; k < Einptrace.size[0]; k++) {
                for (int j = step + 1; j < Einptrace.size[1]; j++) {
                    Einptrace.val[k * Einptrace.size[1] + j] = ff * AEX.val[k * AEX.size[1] + ((j - step) * step)];
                }
            }
            Iinptrace.size[1] += AIX.size[1] / step; // should be always 1000 (so += AEX.size[1] / step) or lower if dt is bigger? (so += step)
            Iinptrace.val = (double*)realloc(Iinptrace.val, Iinptrace.size[0] * Iinptrace.size[1] * sizeof(double));
            for (int k = 0; k < Iinptrace.size[0]; k++) {
                for (int j = step + 1; j < Iinptrace.size[1]; j++) {
                    Iinptrace.val[k * Iinptrace.size[1] + j] = gg * AIX.val[k * AIX.size[1] + ((j - step) * step)];
                }
            }

        }

        /* init gpu */
        cublasHandle_t handle; 
        cublasCreate(&handle);

        double* dev_GEE, * dev_GIE, * dev_GII, * dev_GEI;
        cudaMalloc((void**)&dev_GEE, GEE.size[0] * GEE.size[1] * sizeof(double));
        cudaMalloc((void**)&dev_GIE, GIE.size[0] * GIE.size[1] * sizeof(double));
        cudaMalloc((void**)&dev_GII, GII.size[0] * GII.size[1] * sizeof(double));
        cudaMalloc((void**)&dev_GEI, GEI.size[0] * GEI.size[1] * sizeof(double));
        cudaMemcpy(dev_GEE, GEE.val, GEE.size[0] * GEE.size[1] * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_GIE, GIE.val, GIE.size[0] * GIE.size[1] * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_GII, GII.val, GII.size[0] * GII.size[1] * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_GEI, GEI.val, GEI.size[0] * GEI.size[1] * sizeof(double), cudaMemcpyHostToDevice);

        double* dev_sE, * dev_sIE, * dev_sI, * dev_sEI;
        cudaMalloc((void**)&dev_sE, sE.size * sizeof(double));
        cudaMalloc((void**)&dev_sIE, sIE.size * sizeof(double));
        cudaMalloc((void**)&dev_sI, sI.size * sizeof(double));
        cudaMalloc((void**)&dev_sEI, sEI.size * sizeof(double));
        cudaMemcpy(dev_sE, sE.val, sE.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_sIE, sIE.val, sIE.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_sI, sI.val, sI.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_sEI, sEI.val, sEI.size * sizeof(double), cudaMemcpyHostToDevice);

        double* dev_tmp1, * dev_tmp2, * dev_tmp3, * dev_tmp4;
        cudaMalloc((void**)&dev_tmp1, NE * sizeof(double));
        cudaMalloc((void**)&dev_tmp2, NE * sizeof(double));
        cudaMalloc((void**)&dev_tmp3, NI * sizeof(double));
        cudaMalloc((void**)&dev_tmp4, NE * sizeof(double));

        double* dev_vE,* dev_vI;
        cudaMalloc((void**)&dev_vE, vE.size * sizeof(double));
        cudaMalloc((void**)&dev_vI, vI.size * sizeof(double));
        cudaMemcpy(dev_vE, vE.val, vE.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_vI, vI.val, vI.size * sizeof(double), cudaMemcpyHostToDevice);

        double *dev_lfp, * dev_vbarE, * dev_vbarI, * dev_vegE, *dev_vegI;
        cudaMalloc((void**)&dev_lfp, lfp.size * sizeof(double));
        cudaMalloc((void**)&dev_vbarE, lfp.size * sizeof(double));
        cudaMalloc((void**)&dev_vbarI, lfp.size * sizeof(double));
        cudaMalloc((void**)&dev_vegE, lfp.size * sizeof(double));
        cudaMalloc((void**)&dev_vegI, lfp.size * sizeof(double));
        cudaMemcpy(dev_lfp, lfp.val, lfp.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_vbarE, vbar.E, lfp.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_vbarI, vbar.I, lfp.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_vegE, veg.E, lfp.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_vegI, veg.I, lfp.size * sizeof(double), cudaMemcpyHostToDevice);

        double* dev_isynbarEtoE, * dev_isynbarItoE;
        cudaMalloc((void**)&dev_isynbarEtoE, isynbar.EtoE.size[0] * isynbar.EtoE.size[1] * sizeof(double));
        cudaMalloc((void**)&dev_isynbarItoE, isynbar.ItoE.size[0] * isynbar.ItoE.size[1] * sizeof(double));
        cudaMemcpy(dev_isynbarEtoE, isynbar.EtoE.val, isynbar.EtoE.size[0] * isynbar.EtoE.size[1] * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_isynbarItoE, isynbar.ItoE.val, isynbar.ItoE.size[0] * isynbar.ItoE.size[1] * sizeof(double), cudaMemcpyHostToDevice);

        double* dev_wE, * dev_wI;
        cudaMalloc((void**)&dev_wE, wE.size * sizeof(double));
        cudaMalloc((void**)&dev_wI, wI.size * sizeof(double));
        cudaMemcpy(dev_wE, wE.val, wE.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_wI, wI.val, wI.size * sizeof(double), cudaMemcpyHostToDevice);

        double* dev_Enoise, * dev_Inoise;
        cudaMalloc((void**)&dev_Enoise, Enoise.size[0] * Enoise.size[1] * sizeof(double));
        cudaMalloc((void**)&dev_Inoise, Inoise.size[0] * Inoise.size[1] * sizeof(double));
        cudaMemcpy(dev_Enoise, Enoise.val, Enoise.size[0] * Enoise.size[1] * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Inoise, Inoise.val, Inoise.size[0] * Inoise.size[1] * sizeof(double), cudaMemcpyHostToDevice);

        double *dev_erE, * dev_edE, * dev_erEI, * dev_edEI, * dev_erI, * dev_edI, * dev_erIE, * dev_edIE;
        cudaMalloc((void**)&dev_erE, erE.size * sizeof(double));
        cudaMalloc((void**)&dev_edE, edE.size * sizeof(double));
        cudaMalloc((void**)&dev_erEI, erEI.size * sizeof(double));
        cudaMalloc((void**)&dev_edEI, edEI.size * sizeof(double));
        cudaMalloc((void**)&dev_erI, erI.size * sizeof(double));
        cudaMalloc((void**)&dev_edI, edI.size * sizeof(double));
        cudaMalloc((void**)&dev_erIE, erIE.size * sizeof(double));
        cudaMalloc((void**)&dev_edIE, edIE.size * sizeof(double));
        cudaMemcpy(dev_erE, erE.val, erE.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_edE, edE.val, edE.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_erEI, erEI.val, erEI.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_edEI, edEI.val, edEI.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_erI, erI.val, erI.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_edI, edI.val, edI.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_erIE, erIE.val, erIE.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_edIE, edIE.val, edIE.size * sizeof(double), cudaMemcpyHostToDevice);


        double* dev_tspEtimes, * dev_tspEcelln, * dev_tspItimes, * dev_tspIcelln;
        cudaMalloc((void**)&dev_tspEtimes, tspE.times.size * sizeof(double));
        cudaMalloc((void**)&dev_tspEcelln, tspE.celln.size * sizeof(double));
        cudaMalloc((void**)&dev_tspItimes, tspI.times.size * sizeof(double));
        cudaMalloc((void**)&dev_tspIcelln, tspI.celln.size * sizeof(double));
        cudaMemcpy(dev_tspEtimes, tspE.times.val, tspE.times.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_tspEcelln, tspE.celln.val, tspE.celln.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_tspItimes, tspI.times.val, tspI.times.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_tspIcelln, tspI.celln.val, tspI.celln.size * sizeof(double), cudaMemcpyHostToDevice);

        int* dev_tspE_count, * dev_tspI_count;
        cudaMalloc((void**)&dev_tspE_count, sizeof(int));
        cudaMalloc((void**)&dev_tspI_count, sizeof(int));
        cudaMemcpy(dev_tspE_count, &tspE_count, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_tspI_count, &tspI_count, sizeof(int), cudaMemcpyHostToDevice);

        double* dev_t;
        cudaMalloc((void**)&dev_t, t.size * sizeof(double));
        cudaMemcpy(dev_t, t.val, t.size * sizeof(double), cudaMemcpyHostToDevice);


        dim3 dimBlockGen(1, 1, 1);
        dim3 dimGridGen(1, 1, 1);
        dim3 dimBlockE(32, 1, 1);
        int grid_size = NE / 32;
        if (NE % 32 != 0)
            grid_size++;
        dim3 dimGridE(grid_size, 1, 1);
        dim3 dimBlockI(32, 1, 1);
        grid_size = NI / 32;
        if (NI % 32 != 0)
            grid_size++;
        dim3 dimGridI(grid_size, 1, 1);

        const double dev_alpha = 1.0;
        const double dev_beta = 0.0;

        for (int idt = 1; idt < t.size; idt++) {
            // check the GEE(..) sizes and the transpositions
            cublasDgemv(handle, CUBLAS_OP_T, NE, NE, &dev_alpha, dev_GEE, NE, dev_sE, 1, &dev_beta, dev_tmp1, 1);
            cublasDgemv(handle, CUBLAS_OP_T, NI, NE, &dev_alpha, dev_GIE, NI, dev_sIE, 1, &dev_beta, dev_tmp2, 1);
            cublasDgemv(handle, CUBLAS_OP_T, NI, NI, &dev_alpha, dev_GII, NI, dev_sI, 1, &dev_beta, dev_tmp3, 1);
            cublasDgemv(handle, CUBLAS_OP_T, NI, NE, &dev_alpha, dev_GEI, NI, dev_sEI, 1, &dev_beta, dev_tmp4, 1);

            /* do on gpu */
            kernelGen << <dimGridGen, dimBlockGen >> > (NE, NI, dev_tmp1, dev_tmp2, idt, seqN, dev_vE, dev_vI, dev_lfp, dev_vbarE, dev_vbarI, dev_vegE, dev_vegI, Eeg, Ieg, opt.storecurrs, dev_isynbarEtoE, dev_isynbarItoE, lfp.size, p.VrevE, p.VrevI, dev_tspEtimes, dev_tspEcelln, dev_tspItimes, dev_tspIcelln, dev_tspE_count, dev_tspI_count, dev_t);
            kernelE << <dimGridE, dimBlockE >> > (NE, dev_sE, dev_sEI, dev_tmp1, dev_tmp2, idt, dev_vE, p.VrevE, p.VrevI, p.glE, p.ElE, p.slpE, p.VtE, p.CE, p.aE, p.twE, p.VrE, p.bE, dev_wE, dev_Enoise, Enoise.size[1], dt, dev_erE, dev_edE, dev_erEI, dev_edEI, fdE, frE, fdEI, frEI, pvsE, pvsEI);
            kernelI << <dimGridI, dimBlockI >> > (NI, dev_sIE, dev_sI, dev_tmp3, dev_tmp4, idt, dev_vI, p.VrevE, p.VrevI, p.glI, p.ElI, p.slpI, p.VtI, p.CI, p.aI, p.twI, p.VrI, p.bI, dev_wI, dev_Inoise, Inoise.size[1], dt, dev_erI, dev_edI, dev_erIE, dev_edIE, fdI, frI, fdIE, frIE, pvsI, pvsIE);
        }

        /* get back from gpu */

        // GEE(...) don't change
        cudaMemcpy(sE.val, dev_sE, sE.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(sIE.val, dev_sIE, sIE.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(sI.val, dev_sI, sI.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(sEI.val, dev_sEI, sEI.size * sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(vE.val, dev_vE, vE.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(vI.val, dev_vI, vI.size * sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(lfp.val, dev_lfp, lfp.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(vbar.E, dev_vbarE, lfp.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(vbar.I, dev_vbarI, lfp.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(veg.E, dev_vegE, lfp.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(veg.I, dev_vegI, lfp.size * sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(isynbar.EtoE.val, dev_isynbarEtoE, isynbar.EtoE.size[0] * isynbar.EtoE.size[1] * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(isynbar.ItoE.val, dev_isynbarItoE, isynbar.ItoE.size[0] * isynbar.ItoE.size[1] * sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(wE.val, dev_wE, wE.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(wI.val, dev_wI, wI.size * sizeof(double), cudaMemcpyDeviceToHost);

        // Enoise and Inoise don't change
        cudaMemcpy(erE.val, dev_erE, erE.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(edE.val, dev_edE, edE.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(erEI.val, dev_erEI, erEI.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(edEI.val, dev_edEI, edEI.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(erI.val, dev_erI, erI.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(edI.val, dev_edI, edI.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(erIE.val, dev_erIE, erIE.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(edIE.val, dev_edIE, edIE.size * sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(tspE.times.val, dev_tspEtimes, tspE.times.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(tspE.celln.val, dev_tspEcelln, tspE.celln.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(tspI.times.val, dev_tspItimes, tspI.times.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(tspI.celln.val, dev_tspIcelln, tspI.celln.size * sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(&tspE_count, dev_tspE_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&tspI_count, dev_tspI_count, sizeof(int), cudaMemcpyDeviceToHost);

        cudaMemcpy(t.val, dev_t, t.size * sizeof(double), cudaMemcpyDeviceToHost);

        /* destroy the trash */
        cublasDestroy(handle);

        cudaFree(dev_tmp1);
        cudaFree(dev_tmp2);
        cudaFree(dev_tmp3);
        cudaFree(dev_tmp4);

        cudaFree(dev_GEE);
        cudaFree(dev_GII);
        cudaFree(dev_GEI);
        cudaFree(dev_GIE);

        cudaFree(dev_sE);
        cudaFree(dev_sI);
        cudaFree(dev_sEI);
        cudaFree(dev_sIE);

        cudaFree(dev_vE);
        cudaFree(dev_vI);

        cudaFree(dev_lfp);
        cudaFree(dev_vbarE);
        cudaFree(dev_vbarI);
        cudaFree(dev_vegE);
        cudaFree(dev_vegI);

        cudaFree(dev_isynbarEtoE);
        cudaFree(dev_isynbarItoE);

        cudaFree(dev_wE);
        cudaFree(dev_wI);

        cudaFree(dev_Enoise);
        cudaFree(dev_Inoise);

        cudaFree(dev_erE);
        cudaFree(dev_edE);
        cudaFree(dev_erEI);
        cudaFree(dev_edEI);
        cudaFree(dev_erI);
        cudaFree(dev_edI);
        cudaFree(dev_erIE);
        cudaFree(dev_edIE);

        cudaFree(dev_tspEtimes);
        cudaFree(dev_tspEcelln);
        cudaFree(dev_tspItimes);
        cudaFree(dev_tspIcelln);

        cudaFree(dev_tspE_count);
        cudaFree(dev_tspI_count);

        cudaFree(dev_t);

        seqN++;
        //clock_t toc = clock();
        //printf("elapsed time is %.2lf seconds.\n", (double)(toc - tic) / CLOCKS_PER_SEC);
    }

    tspE.times.size = tspE_count;
    tspE.celln.size = tspE_count;
    //tspE.times.val = (double *)realloc(tspE.times.val, tspE.times.size * sizeof(double));
    //tspE.celln.val = (double *)realloc(tspE.celln.val, tspE.celln.size * sizeof(double));

    tspI.times.size = tspI_count;
    tspI.celln.size = tspI_count;
    //tspI.times.val = (double*)realloc(tspI.times.val, tspI.times.size * sizeof(double));
    //tspI.celln.val = (double*)realloc(tspI.celln.val, tspI.celln.size * sizeof(double));

    inp.Etrace = Einptrace;
    inp.Itrace = Iinptrace;
    clock_t toc = clock();
    printf("%.3lf\n", (double)(toc - tic_start) / CLOCKS_PER_SEC);

    fflush(stdout);
    /* free */

    free(tbins);
    free(tons);
    free(toffs);
    
    free(tmp.val);
    
    free(bmps.val);
    free(bt);
    free(ebt);
    
    free(AEX.val);
    free(AIX.val);
    
    free(MX.val);  
    free(GEE.val);
    free(GII.val);
    
    free(Enoise.val);
    free(Inoise.val);
    
    free(vE.val); 
    free(wE.val); 
    free(sE.val); 
    free(sEI.val); 
    free(erE.val); 
    free(edE.val); 
    free(erEI.val); 
    free(edEI.val);
    free(vI.val);
    free(wI.val);
    free(sI.val);
    free(sIE.val);
    free(erI.val);
    free(edI.val);
    free(erIE.val);
    free(edIE.val);
    free(stsec.val);

    //save results
    save(&veg, &lfp, &tspE, &tspI, &inp, &inps, T, NE, NI);
}