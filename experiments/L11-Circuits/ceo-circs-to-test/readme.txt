The circuits in this directory are for frag00 of the reactants (step 0 of the re_m path).

I ran CEO-ADAPT with CCSD screening to generate a suitable operator pool.

The "bootstrapping" ADAPT algorithm was used with 50,000 and 100,000 determinants.

All the circuits use the same integrals (fcidump.txt).

The output for the entire ADAPT calculations is adapt_output.txt.

I am only including every 25th circuit up to what has run so far. My guess is that there
is no use in going past 200 or 300 gates, but this should help test that. If we know a 
rough cap on this, then it will save computational time if we run the entire >450 set
of fragments for this pathway.

The reference energies included, but we should probably use ASCI and ASCI+PT2 if we go
to publish any comparison. I guess the CCSD(T) energy will be off a few mH from FCI.

 
