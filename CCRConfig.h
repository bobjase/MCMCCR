#ifndef CCR_CONFIG_H
#define CCR_CONFIG_H

#include "CM.hpp"

// --------------------------------------------------------------------------
// CCR CONFIGURATION (Single Source of Truth)
// --------------------------------------------------------------------------
// This constructs the EXACT profile used by 'mcm -x11' on text files.
// It manually excludes models that get cut off by the kInputs limit in the binary.
// --------------------------------------------------------------------------
inline cm::CMProfile GetCCRProfile() {
    cm::CMProfile p;

    // Enable EXACTLY the 13 models seen in the -x11 text log.
    // ORDER MATTERS: This matches the initialization order in cm-inl.hpp
    p.EnableModel(cm::kModelOrder0);
    p.EnableModel(cm::kModelOrder1);
    p.EnableModel(cm::kModelOrder2);
    p.EnableModel(cm::kModelOrder3);
    p.EnableModel(cm::kModelOrder4);
    p.EnableModel(cm::kModelOrder5);
    p.EnableModel(cm::kModelBracket);
    p.EnableModel(cm::kModelSparse2);
    p.EnableModel(cm::kModelSparse3);
    p.EnableModel(cm::kModelSparse4);
    p.EnableModel(cm::kModelWord1);
    p.EnableModel(cm::kModelInterval);
    p.EnableModel(cm::kModelInterval2);

    // EXPLICITLY DISABLED (The "Cutoff" victims in -x11)
    // p.EnableModel(cm::kModelSparse34); 

    // Setup Match/LZP params to match -x11 defaults
    p.SetMatchModelOrder(7); 
    p.SetMinLZPLen(12); // Standard for Text Mode

    return p;
}

#endif // CCR_CONFIG_H