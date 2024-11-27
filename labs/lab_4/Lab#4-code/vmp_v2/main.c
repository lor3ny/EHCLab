/**
*            VMP: Value Counter Monitor
*
*            Versions 
*            - v1, May 2016; by Pedro Pinto
*            - v2, June 2023; by Vitória Correia 
*            
*            v1: library for runtime monitoring range values of program variables
*            v2: library for runtime monitoring values of program variables and with 
*                           implementation of substitution policies
*
*            SPeCS, FEUP.DEI, University of Porto, Portugal
*/

#include <stdio.h>

#include "valuecountermonitor.h"

int main() {

    VCM* vcm = vcm_init("test VCM", 2, 0, 4, 0, 10, 1);

    vcm_inc(vcm, 123.6543);
    vcm_inc(vcm, 123.654300001);
    vcm_inc(vcm, 123.65430000545757);
    vcm_inc(vcm, 123.65430000545757);
    vcm_inc(vcm, 1);
    vcm_inc(vcm, 1.0);
    vcm_inc(vcm, 1.01);
    vcm_inc(vcm, 147);
    vcm_inc(vcm, 2);
    vcm_inc(vcm, 3);
    vcm_inc(vcm, 4);
    vcm_inc(vcm, 5);
    vcm_inc(vcm, 6);
    vcm_inc(vcm, 7);
    vcm_inc(vcm, 500.1354);

    vcm_print(vcm);

    vcm_to_json(vcm, "vcm.json");

    vcm = vcm_destroy(vcm);

    return 0;
}
