\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj: wcvar
Subject To
 w_0_ADM:                        w_0_ADM - 0.99 b_0_ADM + 1.01 s_0_ADM  = 0.1
 w_0_AXP:                        w_0_AXP - 0.99 b_0_AXP + 1.01 s_0_AXP  = 0.1
 w_0_DIS:                        w_0_DIS - 0.99 b_0_DIS + 1.01 s_0_DIS  = 0.1
 w_0_FDX:                        w_0_FDX - 0.99 b_0_FDX + 1.01 s_0_FDX  = 0.1
 w_0_IBM:                        w_0_IBM - 0.99 b_0_IBM + 1.01 s_0_IBM  = 0.1
 w_0_INTC:                       w_0_INTC - 0.99 b_0_INTC + 1.01 s_0_INTC
                                  = 0.1
 w_0_KO:                         w_0_KO - 0.99 b_0_KO + 1.01 s_0_KO  = 0.1
 w_0_MAT:                        w_0_MAT - 0.99 b_0_MAT + 1.01 s_0_MAT  = 0.1
 w_0_MSFT:                       w_0_MSFT - 0.99 b_0_MSFT + 1.01 s_0_MSFT
                                  = 0.1
 w_0_PG:                         w_0_PG - 0.99 b_0_PG + 1.01 s_0_PG  = 0.1
 w_0_balance_constraint:         b_0_ADM + b_0_AXP + b_0_DIS + b_0_FDX
                                 + b_0_IBM + b_0_INTC + b_0_KO + b_0_MAT
                                 + b_0_MSFT + b_0_PG - s_0_ADM - s_0_AXP
                                 - s_0_DIS - s_0_FDX - s_0_IBM - s_0_INTC
                                 - s_0_KO - s_0_MAT - s_0_MSFT - s_0_PG  = 0
 w_k1_ADM_s0_return_constraint:  1.08195 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.956243 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.08531 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.19421 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.21798 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.936209 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.967975 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.66909 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.36592 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.937419 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.82793 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.16589 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.891652 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.04082 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.889994 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.670786 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.05878 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.06613 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.905019 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.07603 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.52137 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.747398 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.804161 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.945169 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.11592 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.14536 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.37044 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.03631 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.834637 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.52143 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.34604 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.720015 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.06007 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.853562 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.08329 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.28002 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.744102 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.34328 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.24148 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.883666 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.926121 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.24483 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.39349 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.918654 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.902492 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.772912 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.59336 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.07569 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.819864 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.37611 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.822793 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.38192 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.854736 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.619736 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.29376 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.21648 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.890273 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.05356 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.69875 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.767571 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.731355 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.817767 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.38086 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.25441 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.13856 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.08129 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.15232 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.76543 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.62297 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.624841 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.875193 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.07394 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.13186 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.39121 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.804063 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.29677 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.29884 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.853256 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.894685 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.33851 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.51104 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.878929 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.07056 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.720763 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.84707 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.49434 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.658414 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.34843 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.919773 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.18312 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.694145 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.486753 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.53699 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.20913 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.15746 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.957915 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.67488 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.771451 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.712334 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.986545 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.00857 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.70121 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.32252 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.16835 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.04482 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.55928 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.52505 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.598269 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.956802 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.898486 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.02475 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.62692 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.721697 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.60517 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.15931 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.900821 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.962966 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.12246 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.34246 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.930807 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.02691 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.742626 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.47363 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.19592 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.887427 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.36253 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.871847 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.11489 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.76895 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.642996 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.15213 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.22515 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.05898 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.884546 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.58695 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.887451 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.861301 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.895753 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.11554 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.15595 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.36541 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.14636 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.869458 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.40707 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.29921 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.755316 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.917833 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.894785 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.14065 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.12582 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.852287 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.4099 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.21663 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.897984 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.893294 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.42757 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.14017 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.02601 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.795737 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.965217 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.4433 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.04485 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.630332 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.759626 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.36603 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.1689 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.03838 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.732185 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.67799 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.915645 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.946777 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.37195 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.09889 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.695842 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.649352 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.20024 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.14734 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 2.00524 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.622025 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.923033 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.658 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.36039 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.27466 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.57226 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.03263 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.42968 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.793769 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.88056 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.833934 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.855272 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.17343 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.928193 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.866367 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.04553 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.11321 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.912568 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.983475 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.878038 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.37463 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.22598 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.843653 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.31117 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.02396 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.98455 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.848308 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.909617 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.02626 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.015 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.993665 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.02524 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.36791 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.0197 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.0207 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.980431 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.35524 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.27714 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.41649 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.30081 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.03548 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.47689 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.23825 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.813947 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.890693 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.03142 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.09569 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.04918 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.847046 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.25663 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.2493 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.910911 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.888535 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.28098 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.21504 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.980232 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.85154 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.899047 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.39589 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.20362 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.678749 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.983041 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.08514 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.11188 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.857285 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.639183 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.53368 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.988057 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.964514 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.13412 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.33634 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.752763 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.760942 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.17652 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.0046 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.61158 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.995053 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.00826 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.29946 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.46661 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.15618 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.615658 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.987937 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.09934 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.880099 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.51628 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.715998 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.20462 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.18467 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.931365 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.91453 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.8267 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.02543 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.870007 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.96982 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.1547 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.52008 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.14036 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.703655 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.909473 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.31998 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.67993 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.873299 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.636382 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.63011 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.823173 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.845568 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.14903 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.19307 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.618723 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.679334 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.998279 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.52129 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 2.01778 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.603404 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.87171 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.9097 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.81078 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.65117 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.59609 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.22729 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.7613 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.907498 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.99374 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.744132 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.14368 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.15463 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.909517 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.929191 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.08298 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.37042 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.947744 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.07025 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.793695 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.41463 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.08588 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.841477 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.16818 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.03458 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.986279 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.858544 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.754963 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.14483 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.09717 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.14471 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.930848 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.42963 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.958483 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.813497 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.907941 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.872393 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.18159 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.15544 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.04461 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.91867 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.1903 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.17621 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.769643 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.0066 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.926725 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.07351 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.06586 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.860957 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.36559 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.15865 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.917593 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.898552 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.13841 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.24307 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.958901 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.0156 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.862654 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.31659 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.16966 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.915951 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.03722 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.07134 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.09992 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.889848 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.773297 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.09541 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.07661 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.99507 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.970764 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.16075 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.857876 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.820249 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.903218 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.970701 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.25311 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.07883 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.973721 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.01282 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.2651 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.29991 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.755456 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.01019 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.03598 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.04311 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.19138 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.877823 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.19502 w_k1_PG_s11 - w_k1_PG_s117
                                 + 0.99 b_k1_PG_s117 - 1.01 s_k1_PG_s117  = 0
 k1_s0_balance_constraint:       b_k1_ADM_s0 + b_k1_AXP_s0 + b_k1_DIS_s0
                                 + b_k1_FDX_s0 + b_k1_IBM_s0 + b_k1_INTC_s0
                                 + b_k1_KO_s0 + b_k1_MAT_s0 + b_k1_MSFT_s0
                                 + b_k1_PG_s0 - s_k1_ADM_s0 - s_k1_AXP_s0
                                 - s_k1_DIS_s0 - s_k1_FDX_s0 - s_k1_IBM_s0
                                 - s_k1_INTC_s0 - s_k1_KO_s0 - s_k1_MAT_s0
                                 - s_k1_MSFT_s0 - s_k1_PG_s0  = 0
 k1_s1_balance_constraint:       b_k1_ADM_s1 + b_k1_AXP_s1 + b_k1_DIS_s1
                                 + b_k1_FDX_s1 + b_k1_IBM_s1 + b_k1_INTC_s1
                                 + b_k1_KO_s1 + b_k1_MAT_s1 + b_k1_MSFT_s1
                                 + b_k1_PG_s1 - s_k1_ADM_s1 - s_k1_AXP_s1
                                 - s_k1_DIS_s1 - s_k1_FDX_s1 - s_k1_IBM_s1
                                 - s_k1_INTC_s1 - s_k1_KO_s1 - s_k1_MAT_s1
                                 - s_k1_MSFT_s1 - s_k1_PG_s1  = 0
 k1_s00_balance_constraint:      b_k1_ADM_s00 + b_k1_AXP_s00 + b_k1_DIS_s00
                                 + b_k1_FDX_s00 + b_k1_IBM_s00 + b_k1_INTC_s00
                                 + b_k1_KO_s00 + b_k1_MAT_s00 + b_k1_MSFT_s00
                                 + b_k1_PG_s00 - s_k1_ADM_s00 - s_k1_AXP_s00
                                 - s_k1_DIS_s00 - s_k1_FDX_s00 - s_k1_IBM_s00
                                 - s_k1_INTC_s00 - s_k1_KO_s00 - s_k1_MAT_s00
                                 - s_k1_MSFT_s00 - s_k1_PG_s00  = 0
 k1_s01_balance_constraint:      b_k1_ADM_s01 + b_k1_AXP_s01 + b_k1_DIS_s01
                                 + b_k1_FDX_s01 + b_k1_IBM_s01 + b_k1_INTC_s01
                                 + b_k1_KO_s01 + b_k1_MAT_s01 + b_k1_MSFT_s01
                                 + b_k1_PG_s01 - s_k1_ADM_s01 - s_k1_AXP_s01
                                 - s_k1_DIS_s01 - s_k1_FDX_s01 - s_k1_IBM_s01
                                 - s_k1_INTC_s01 - s_k1_KO_s01 - s_k1_MAT_s01
                                 - s_k1_MSFT_s01 - s_k1_PG_s01  = 0
 k1_s10_balance_constraint:      b_k1_ADM_s10 + b_k1_AXP_s10 + b_k1_DIS_s10
                                 + b_k1_FDX_s10 + b_k1_IBM_s10 + b_k1_INTC_s10
                                 + b_k1_KO_s10 + b_k1_MAT_s10 + b_k1_MSFT_s10
                                 + b_k1_PG_s10 - s_k1_ADM_s10 - s_k1_AXP_s10
                                 - s_k1_DIS_s10 - s_k1_FDX_s10 - s_k1_IBM_s10
                                 - s_k1_INTC_s10 - s_k1_KO_s10 - s_k1_MAT_s10
                                 - s_k1_MSFT_s10 - s_k1_PG_s10  = 0
 k1_s11_balance_constraint:      b_k1_ADM_s11 + b_k1_AXP_s11 + b_k1_DIS_s11
                                 + b_k1_FDX_s11 + b_k1_IBM_s11 + b_k1_INTC_s11
                                 + b_k1_KO_s11 + b_k1_MAT_s11 + b_k1_MSFT_s11
                                 + b_k1_PG_s11 - s_k1_ADM_s11 - s_k1_AXP_s11
                                 - s_k1_DIS_s11 - s_k1_FDX_s11 - s_k1_IBM_s11
                                 - s_k1_INTC_s11 - s_k1_KO_s11 - s_k1_MAT_s11
                                 - s_k1_MSFT_s11 - s_k1_PG_s11  = 0
 k1_s000_balance_constraint:     b_k1_ADM_s000 + b_k1_AXP_s000 + b_k1_DIS_s000
                                 + b_k1_FDX_s000 + b_k1_IBM_s000
                                 + b_k1_INTC_s000 + b_k1_KO_s000
                                 + b_k1_MAT_s000 + b_k1_MSFT_s000
                                 + b_k1_PG_s000 - s_k1_ADM_s000 - s_k1_AXP_s000
                                 - s_k1_DIS_s000 - s_k1_FDX_s000
                                 - s_k1_IBM_s000 - s_k1_INTC_s000
                                 - s_k1_KO_s000 - s_k1_MAT_s000
                                 - s_k1_MSFT_s000 - s_k1_PG_s000  = 0
 k1_s001_balance_constraint:     b_k1_ADM_s001 + b_k1_AXP_s001 + b_k1_DIS_s001
                                 + b_k1_FDX_s001 + b_k1_IBM_s001
                                 + b_k1_INTC_s001 + b_k1_KO_s001
                                 + b_k1_MAT_s001 + b_k1_MSFT_s001
                                 + b_k1_PG_s001 - s_k1_ADM_s001 - s_k1_AXP_s001
                                 - s_k1_DIS_s001 - s_k1_FDX_s001
                                 - s_k1_IBM_s001 - s_k1_INTC_s001
                                 - s_k1_KO_s001 - s_k1_MAT_s001
                                 - s_k1_MSFT_s001 - s_k1_PG_s001  = 0
 k1_s002_balance_constraint:     b_k1_ADM_s002 + b_k1_AXP_s002 + b_k1_DIS_s002
                                 + b_k1_FDX_s002 + b_k1_IBM_s002
                                 + b_k1_INTC_s002 + b_k1_KO_s002
                                 + b_k1_MAT_s002 + b_k1_MSFT_s002
                                 + b_k1_PG_s002 - s_k1_ADM_s002 - s_k1_AXP_s002
                                 - s_k1_DIS_s002 - s_k1_FDX_s002
                                 - s_k1_IBM_s002 - s_k1_INTC_s002
                                 - s_k1_KO_s002 - s_k1_MAT_s002
                                 - s_k1_MSFT_s002 - s_k1_PG_s002  = 0
 k1_s003_balance_constraint:     b_k1_ADM_s003 + b_k1_AXP_s003 + b_k1_DIS_s003
                                 + b_k1_FDX_s003 + b_k1_IBM_s003
                                 + b_k1_INTC_s003 + b_k1_KO_s003
                                 + b_k1_MAT_s003 + b_k1_MSFT_s003
                                 + b_k1_PG_s003 - s_k1_ADM_s003 - s_k1_AXP_s003
                                 - s_k1_DIS_s003 - s_k1_FDX_s003
                                 - s_k1_IBM_s003 - s_k1_INTC_s003
                                 - s_k1_KO_s003 - s_k1_MAT_s003
                                 - s_k1_MSFT_s003 - s_k1_PG_s003  = 0
 k1_s004_balance_constraint:     b_k1_ADM_s004 + b_k1_AXP_s004 + b_k1_DIS_s004
                                 + b_k1_FDX_s004 + b_k1_IBM_s004
                                 + b_k1_INTC_s004 + b_k1_KO_s004
                                 + b_k1_MAT_s004 + b_k1_MSFT_s004
                                 + b_k1_PG_s004 - s_k1_ADM_s004 - s_k1_AXP_s004
                                 - s_k1_DIS_s004 - s_k1_FDX_s004
                                 - s_k1_IBM_s004 - s_k1_INTC_s004
                                 - s_k1_KO_s004 - s_k1_MAT_s004
                                 - s_k1_MSFT_s004 - s_k1_PG_s004  = 0
 k1_s005_balance_constraint:     b_k1_ADM_s005 + b_k1_AXP_s005 + b_k1_DIS_s005
                                 + b_k1_FDX_s005 + b_k1_IBM_s005
                                 + b_k1_INTC_s005 + b_k1_KO_s005
                                 + b_k1_MAT_s005 + b_k1_MSFT_s005
                                 + b_k1_PG_s005 - s_k1_ADM_s005 - s_k1_AXP_s005
                                 - s_k1_DIS_s005 - s_k1_FDX_s005
                                 - s_k1_IBM_s005 - s_k1_INTC_s005
                                 - s_k1_KO_s005 - s_k1_MAT_s005
                                 - s_k1_MSFT_s005 - s_k1_PG_s005  = 0
 k1_s006_balance_constraint:     b_k1_ADM_s006 + b_k1_AXP_s006 + b_k1_DIS_s006
                                 + b_k1_FDX_s006 + b_k1_IBM_s006
                                 + b_k1_INTC_s006 + b_k1_KO_s006
                                 + b_k1_MAT_s006 + b_k1_MSFT_s006
                                 + b_k1_PG_s006 - s_k1_ADM_s006 - s_k1_AXP_s006
                                 - s_k1_DIS_s006 - s_k1_FDX_s006
                                 - s_k1_IBM_s006 - s_k1_INTC_s006
                                 - s_k1_KO_s006 - s_k1_MAT_s006
                                 - s_k1_MSFT_s006 - s_k1_PG_s006  = 0
 k1_s007_balance_constraint:     b_k1_ADM_s007 + b_k1_AXP_s007 + b_k1_DIS_s007
                                 + b_k1_FDX_s007 + b_k1_IBM_s007
                                 + b_k1_INTC_s007 + b_k1_KO_s007
                                 + b_k1_MAT_s007 + b_k1_MSFT_s007
                                 + b_k1_PG_s007 - s_k1_ADM_s007 - s_k1_AXP_s007
                                 - s_k1_DIS_s007 - s_k1_FDX_s007
                                 - s_k1_IBM_s007 - s_k1_INTC_s007
                                 - s_k1_KO_s007 - s_k1_MAT_s007
                                 - s_k1_MSFT_s007 - s_k1_PG_s007  = 0
 k1_s010_balance_constraint:     b_k1_ADM_s010 + b_k1_AXP_s010 + b_k1_DIS_s010
                                 + b_k1_FDX_s010 + b_k1_IBM_s010
                                 + b_k1_INTC_s010 + b_k1_KO_s010
                                 + b_k1_MAT_s010 + b_k1_MSFT_s010
                                 + b_k1_PG_s010 - s_k1_ADM_s010 - s_k1_AXP_s010
                                 - s_k1_DIS_s010 - s_k1_FDX_s010
                                 - s_k1_IBM_s010 - s_k1_INTC_s010
                                 - s_k1_KO_s010 - s_k1_MAT_s010
                                 - s_k1_MSFT_s010 - s_k1_PG_s010  = 0
 k1_s011_balance_constraint:     b_k1_ADM_s011 + b_k1_AXP_s011 + b_k1_DIS_s011
                                 + b_k1_FDX_s011 + b_k1_IBM_s011
                                 + b_k1_INTC_s011 + b_k1_KO_s011
                                 + b_k1_MAT_s011 + b_k1_MSFT_s011
                                 + b_k1_PG_s011 - s_k1_ADM_s011 - s_k1_AXP_s011
                                 - s_k1_DIS_s011 - s_k1_FDX_s011
                                 - s_k1_IBM_s011 - s_k1_INTC_s011
                                 - s_k1_KO_s011 - s_k1_MAT_s011
                                 - s_k1_MSFT_s011 - s_k1_PG_s011  = 0
 k1_s012_balance_constraint:     b_k1_ADM_s012 + b_k1_AXP_s012 + b_k1_DIS_s012
                                 + b_k1_FDX_s012 + b_k1_IBM_s012
                                 + b_k1_INTC_s012 + b_k1_KO_s012
                                 + b_k1_MAT_s012 + b_k1_MSFT_s012
                                 + b_k1_PG_s012 - s_k1_ADM_s012 - s_k1_AXP_s012
                                 - s_k1_DIS_s012 - s_k1_FDX_s012
                                 - s_k1_IBM_s012 - s_k1_INTC_s012
                                 - s_k1_KO_s012 - s_k1_MAT_s012
                                 - s_k1_MSFT_s012 - s_k1_PG_s012  = 0
 k1_s013_balance_constraint:     b_k1_ADM_s013 + b_k1_AXP_s013 + b_k1_DIS_s013
                                 + b_k1_FDX_s013 + b_k1_IBM_s013
                                 + b_k1_INTC_s013 + b_k1_KO_s013
                                 + b_k1_MAT_s013 + b_k1_MSFT_s013
                                 + b_k1_PG_s013 - s_k1_ADM_s013 - s_k1_AXP_s013
                                 - s_k1_DIS_s013 - s_k1_FDX_s013
                                 - s_k1_IBM_s013 - s_k1_INTC_s013
                                 - s_k1_KO_s013 - s_k1_MAT_s013
                                 - s_k1_MSFT_s013 - s_k1_PG_s013  = 0
 k1_s014_balance_constraint:     b_k1_ADM_s014 + b_k1_AXP_s014 + b_k1_DIS_s014
                                 + b_k1_FDX_s014 + b_k1_IBM_s014
                                 + b_k1_INTC_s014 + b_k1_KO_s014
                                 + b_k1_MAT_s014 + b_k1_MSFT_s014
                                 + b_k1_PG_s014 - s_k1_ADM_s014 - s_k1_AXP_s014
                                 - s_k1_DIS_s014 - s_k1_FDX_s014
                                 - s_k1_IBM_s014 - s_k1_INTC_s014
                                 - s_k1_KO_s014 - s_k1_MAT_s014
                                 - s_k1_MSFT_s014 - s_k1_PG_s014  = 0
 k1_s015_balance_constraint:     b_k1_ADM_s015 + b_k1_AXP_s015 + b_k1_DIS_s015
                                 + b_k1_FDX_s015 + b_k1_IBM_s015
                                 + b_k1_INTC_s015 + b_k1_KO_s015
                                 + b_k1_MAT_s015 + b_k1_MSFT_s015
                                 + b_k1_PG_s015 - s_k1_ADM_s015 - s_k1_AXP_s015
                                 - s_k1_DIS_s015 - s_k1_FDX_s015
                                 - s_k1_IBM_s015 - s_k1_INTC_s015
                                 - s_k1_KO_s015 - s_k1_MAT_s015
                                 - s_k1_MSFT_s015 - s_k1_PG_s015  = 0
 k1_s016_balance_constraint:     b_k1_ADM_s016 + b_k1_AXP_s016 + b_k1_DIS_s016
                                 + b_k1_FDX_s016 + b_k1_IBM_s016
                                 + b_k1_INTC_s016 + b_k1_KO_s016
                                 + b_k1_MAT_s016 + b_k1_MSFT_s016
                                 + b_k1_PG_s016 - s_k1_ADM_s016 - s_k1_AXP_s016
                                 - s_k1_DIS_s016 - s_k1_FDX_s016
                                 - s_k1_IBM_s016 - s_k1_INTC_s016
                                 - s_k1_KO_s016 - s_k1_MAT_s016
                                 - s_k1_MSFT_s016 - s_k1_PG_s016  = 0
 k1_s017_balance_constraint:     b_k1_ADM_s017 + b_k1_AXP_s017 + b_k1_DIS_s017
                                 + b_k1_FDX_s017 + b_k1_IBM_s017
                                 + b_k1_INTC_s017 + b_k1_KO_s017
                                 + b_k1_MAT_s017 + b_k1_MSFT_s017
                                 + b_k1_PG_s017 - s_k1_ADM_s017 - s_k1_AXP_s017
                                 - s_k1_DIS_s017 - s_k1_FDX_s017
                                 - s_k1_IBM_s017 - s_k1_INTC_s017
                                 - s_k1_KO_s017 - s_k1_MAT_s017
                                 - s_k1_MSFT_s017 - s_k1_PG_s017  = 0
 k1_s100_balance_constraint:     b_k1_ADM_s100 + b_k1_AXP_s100 + b_k1_DIS_s100
                                 + b_k1_FDX_s100 + b_k1_IBM_s100
                                 + b_k1_INTC_s100 + b_k1_KO_s100
                                 + b_k1_MAT_s100 + b_k1_MSFT_s100
                                 + b_k1_PG_s100 - s_k1_ADM_s100 - s_k1_AXP_s100
                                 - s_k1_DIS_s100 - s_k1_FDX_s100
                                 - s_k1_IBM_s100 - s_k1_INTC_s100
                                 - s_k1_KO_s100 - s_k1_MAT_s100
                                 - s_k1_MSFT_s100 - s_k1_PG_s100  = 0
 k1_s101_balance_constraint:     b_k1_ADM_s101 + b_k1_AXP_s101 + b_k1_DIS_s101
                                 + b_k1_FDX_s101 + b_k1_IBM_s101
                                 + b_k1_INTC_s101 + b_k1_KO_s101
                                 + b_k1_MAT_s101 + b_k1_MSFT_s101
                                 + b_k1_PG_s101 - s_k1_ADM_s101 - s_k1_AXP_s101
                                 - s_k1_DIS_s101 - s_k1_FDX_s101
                                 - s_k1_IBM_s101 - s_k1_INTC_s101
                                 - s_k1_KO_s101 - s_k1_MAT_s101
                                 - s_k1_MSFT_s101 - s_k1_PG_s101  = 0
 k1_s102_balance_constraint:     b_k1_ADM_s102 + b_k1_AXP_s102 + b_k1_DIS_s102
                                 + b_k1_FDX_s102 + b_k1_IBM_s102
                                 + b_k1_INTC_s102 + b_k1_KO_s102
                                 + b_k1_MAT_s102 + b_k1_MSFT_s102
                                 + b_k1_PG_s102 - s_k1_ADM_s102 - s_k1_AXP_s102
                                 - s_k1_DIS_s102 - s_k1_FDX_s102
                                 - s_k1_IBM_s102 - s_k1_INTC_s102
                                 - s_k1_KO_s102 - s_k1_MAT_s102
                                 - s_k1_MSFT_s102 - s_k1_PG_s102  = 0
 k1_s103_balance_constraint:     b_k1_ADM_s103 + b_k1_AXP_s103 + b_k1_DIS_s103
                                 + b_k1_FDX_s103 + b_k1_IBM_s103
                                 + b_k1_INTC_s103 + b_k1_KO_s103
                                 + b_k1_MAT_s103 + b_k1_MSFT_s103
                                 + b_k1_PG_s103 - s_k1_ADM_s103 - s_k1_AXP_s103
                                 - s_k1_DIS_s103 - s_k1_FDX_s103
                                 - s_k1_IBM_s103 - s_k1_INTC_s103
                                 - s_k1_KO_s103 - s_k1_MAT_s103
                                 - s_k1_MSFT_s103 - s_k1_PG_s103  = 0
 k1_s104_balance_constraint:     b_k1_ADM_s104 + b_k1_AXP_s104 + b_k1_DIS_s104
                                 + b_k1_FDX_s104 + b_k1_IBM_s104
                                 + b_k1_INTC_s104 + b_k1_KO_s104
                                 + b_k1_MAT_s104 + b_k1_MSFT_s104
                                 + b_k1_PG_s104 - s_k1_ADM_s104 - s_k1_AXP_s104
                                 - s_k1_DIS_s104 - s_k1_FDX_s104
                                 - s_k1_IBM_s104 - s_k1_INTC_s104
                                 - s_k1_KO_s104 - s_k1_MAT_s104
                                 - s_k1_MSFT_s104 - s_k1_PG_s104  = 0
 k1_s105_balance_constraint:     b_k1_ADM_s105 + b_k1_AXP_s105 + b_k1_DIS_s105
                                 + b_k1_FDX_s105 + b_k1_IBM_s105
                                 + b_k1_INTC_s105 + b_k1_KO_s105
                                 + b_k1_MAT_s105 + b_k1_MSFT_s105
                                 + b_k1_PG_s105 - s_k1_ADM_s105 - s_k1_AXP_s105
                                 - s_k1_DIS_s105 - s_k1_FDX_s105
                                 - s_k1_IBM_s105 - s_k1_INTC_s105
                                 - s_k1_KO_s105 - s_k1_MAT_s105
                                 - s_k1_MSFT_s105 - s_k1_PG_s105  = 0
 k1_s106_balance_constraint:     b_k1_ADM_s106 + b_k1_AXP_s106 + b_k1_DIS_s106
                                 + b_k1_FDX_s106 + b_k1_IBM_s106
                                 + b_k1_INTC_s106 + b_k1_KO_s106
                                 + b_k1_MAT_s106 + b_k1_MSFT_s106
                                 + b_k1_PG_s106 - s_k1_ADM_s106 - s_k1_AXP_s106
                                 - s_k1_DIS_s106 - s_k1_FDX_s106
                                 - s_k1_IBM_s106 - s_k1_INTC_s106
                                 - s_k1_KO_s106 - s_k1_MAT_s106
                                 - s_k1_MSFT_s106 - s_k1_PG_s106  = 0
 k1_s107_balance_constraint:     b_k1_ADM_s107 + b_k1_AXP_s107 + b_k1_DIS_s107
                                 + b_k1_FDX_s107 + b_k1_IBM_s107
                                 + b_k1_INTC_s107 + b_k1_KO_s107
                                 + b_k1_MAT_s107 + b_k1_MSFT_s107
                                 + b_k1_PG_s107 - s_k1_ADM_s107 - s_k1_AXP_s107
                                 - s_k1_DIS_s107 - s_k1_FDX_s107
                                 - s_k1_IBM_s107 - s_k1_INTC_s107
                                 - s_k1_KO_s107 - s_k1_MAT_s107
                                 - s_k1_MSFT_s107 - s_k1_PG_s107  = 0
 k1_s110_balance_constraint:     b_k1_ADM_s110 + b_k1_AXP_s110 + b_k1_DIS_s110
                                 + b_k1_FDX_s110 + b_k1_IBM_s110
                                 + b_k1_INTC_s110 + b_k1_KO_s110
                                 + b_k1_MAT_s110 + b_k1_MSFT_s110
                                 + b_k1_PG_s110 - s_k1_ADM_s110 - s_k1_AXP_s110
                                 - s_k1_DIS_s110 - s_k1_FDX_s110
                                 - s_k1_IBM_s110 - s_k1_INTC_s110
                                 - s_k1_KO_s110 - s_k1_MAT_s110
                                 - s_k1_MSFT_s110 - s_k1_PG_s110  = 0
 k1_s111_balance_constraint:     b_k1_ADM_s111 + b_k1_AXP_s111 + b_k1_DIS_s111
                                 + b_k1_FDX_s111 + b_k1_IBM_s111
                                 + b_k1_INTC_s111 + b_k1_KO_s111
                                 + b_k1_MAT_s111 + b_k1_MSFT_s111
                                 + b_k1_PG_s111 - s_k1_ADM_s111 - s_k1_AXP_s111
                                 - s_k1_DIS_s111 - s_k1_FDX_s111
                                 - s_k1_IBM_s111 - s_k1_INTC_s111
                                 - s_k1_KO_s111 - s_k1_MAT_s111
                                 - s_k1_MSFT_s111 - s_k1_PG_s111  = 0
 k1_s112_balance_constraint:     b_k1_ADM_s112 + b_k1_AXP_s112 + b_k1_DIS_s112
                                 + b_k1_FDX_s112 + b_k1_IBM_s112
                                 + b_k1_INTC_s112 + b_k1_KO_s112
                                 + b_k1_MAT_s112 + b_k1_MSFT_s112
                                 + b_k1_PG_s112 - s_k1_ADM_s112 - s_k1_AXP_s112
                                 - s_k1_DIS_s112 - s_k1_FDX_s112
                                 - s_k1_IBM_s112 - s_k1_INTC_s112
                                 - s_k1_KO_s112 - s_k1_MAT_s112
                                 - s_k1_MSFT_s112 - s_k1_PG_s112  = 0
 k1_s113_balance_constraint:     b_k1_ADM_s113 + b_k1_AXP_s113 + b_k1_DIS_s113
                                 + b_k1_FDX_s113 + b_k1_IBM_s113
                                 + b_k1_INTC_s113 + b_k1_KO_s113
                                 + b_k1_MAT_s113 + b_k1_MSFT_s113
                                 + b_k1_PG_s113 - s_k1_ADM_s113 - s_k1_AXP_s113
                                 - s_k1_DIS_s113 - s_k1_FDX_s113
                                 - s_k1_IBM_s113 - s_k1_INTC_s113
                                 - s_k1_KO_s113 - s_k1_MAT_s113
                                 - s_k1_MSFT_s113 - s_k1_PG_s113  = 0
 k1_s114_balance_constraint:     b_k1_ADM_s114 + b_k1_AXP_s114 + b_k1_DIS_s114
                                 + b_k1_FDX_s114 + b_k1_IBM_s114
                                 + b_k1_INTC_s114 + b_k1_KO_s114
                                 + b_k1_MAT_s114 + b_k1_MSFT_s114
                                 + b_k1_PG_s114 - s_k1_ADM_s114 - s_k1_AXP_s114
                                 - s_k1_DIS_s114 - s_k1_FDX_s114
                                 - s_k1_IBM_s114 - s_k1_INTC_s114
                                 - s_k1_KO_s114 - s_k1_MAT_s114
                                 - s_k1_MSFT_s114 - s_k1_PG_s114  = 0
 k1_s115_balance_constraint:     b_k1_ADM_s115 + b_k1_AXP_s115 + b_k1_DIS_s115
                                 + b_k1_FDX_s115 + b_k1_IBM_s115
                                 + b_k1_INTC_s115 + b_k1_KO_s115
                                 + b_k1_MAT_s115 + b_k1_MSFT_s115
                                 + b_k1_PG_s115 - s_k1_ADM_s115 - s_k1_AXP_s115
                                 - s_k1_DIS_s115 - s_k1_FDX_s115
                                 - s_k1_IBM_s115 - s_k1_INTC_s115
                                 - s_k1_KO_s115 - s_k1_MAT_s115
                                 - s_k1_MSFT_s115 - s_k1_PG_s115  = 0
 k1_s116_balance_constraint:     b_k1_ADM_s116 + b_k1_AXP_s116 + b_k1_DIS_s116
                                 + b_k1_FDX_s116 + b_k1_IBM_s116
                                 + b_k1_INTC_s116 + b_k1_KO_s116
                                 + b_k1_MAT_s116 + b_k1_MSFT_s116
                                 + b_k1_PG_s116 - s_k1_ADM_s116 - s_k1_AXP_s116
                                 - s_k1_DIS_s116 - s_k1_FDX_s116
                                 - s_k1_IBM_s116 - s_k1_INTC_s116
                                 - s_k1_KO_s116 - s_k1_MAT_s116
                                 - s_k1_MSFT_s116 - s_k1_PG_s116  = 0
 k1_s117_balance_constraint:     b_k1_ADM_s117 + b_k1_AXP_s117 + b_k1_DIS_s117
                                 + b_k1_FDX_s117 + b_k1_IBM_s117
                                 + b_k1_INTC_s117 + b_k1_KO_s117
                                 + b_k1_MAT_s117 + b_k1_MSFT_s117
                                 + b_k1_PG_s117 - s_k1_ADM_s117 - s_k1_AXP_s117
                                 - s_k1_DIS_s117 - s_k1_FDX_s117
                                 - s_k1_IBM_s117 - s_k1_INTC_s117
                                 - s_k1_KO_s117 - s_k1_MAT_s117
                                 - s_k1_MSFT_s117 - s_k1_PG_s117  = 0
 k1_wcvar_constraint:            wcvar - var_1 - 0.224999932265815 z_k1_s0000
                                 - 0.582999170226644 z_k1_s0001
                                 - 0.273999824048164 z_k1_s0002
                                 - 0.225999838498124 z_k1_s0003
                                 - 0.143999956650122 z_k1_s0004
                                 - 0.579000666897224 z_k1_s0005
                                 - 0.361999143290015 z_k1_s0006
                                 - 0.410999035072364 z_k1_s0007
                                 - 1.76300254021053 z_k1_s0010
                                 - 0.371000392841717 z_k1_s0011
                                 - 0.417000436191088 z_k1_s0012
                                 - 1.07000316932461 z_k1_s0013
                                 - 0.444000187911085 z_k1_s0014
                                 - 1.67300115880823 z_k1_s0015
                                 - 0.488000200562627 z_k1_s0016
                                 - 0.401000190608456 z_k1_s0017
                                 - 0.27900035969515 z_k1_s0020
                                 - 0.273000469366413 z_k1_s0021
                                 - 0.648001500725298 z_k1_s0022
                                 - 0.132000316936653 z_k1_s0023
                                 - 0.137000225543934 z_k1_s0024
                                 - 0.499002101124872 z_k1_s0025
                                 - 0.410999463429836 z_k1_s0026
                                 - 0.654000784453052 z_k1_s0027
                                 - 0.360000406179363 z_k1_s0030
                                 - 0.62500081297818 z_k1_s0031
                                 - 0.398000871652097 z_k1_s0032
                                 - 0.373000606272712 z_k1_s0033
                                 - 0.426999735758585 z_k1_s0034
                                 - 0.674002373230926 z_k1_s0035
                                 - 0.804000493160119 z_k1_s0036
                                 - 0.220000161976182 z_k1_s0037
                                 - 0.508000407245974 z_k1_s0040
                                 - 0.236999313192615 z_k1_s0041
                                 - 0.261999463510347 z_k1_s0042
                                 - 0.540997586655838 z_k1_s0043
                                 - 0.80099677044534 z_k1_s0044
                                 - 0.593997563274162 z_k1_s0045
                                 - 0.215999722316575 z_k1_s0046
                                 - 0.559000104143451 z_k1_s0047
                                 - 0.272999579146891 z_k1_s0050
                                 - 0.497998869906001 z_k1_s0051
                                 - 0.23299956315665 z_k1_s0052
                                 - 0.662997968430069 z_k1_s0053
                                 - 0.349000316177013 z_k1_s0054
                                 - 0.352997928630547 z_k1_s0055
                                 - 0.631000243833557 z_k1_s0056
                                 - 0.365000860972093 z_k1_s0057
                                 - 1.12800356870266 z_k1_s0060
                                 - 0.301000650871268 z_k1_s0061
                                 - 0.597002959299655 z_k1_s0062
                                 - 0.519999566418869 z_k1_s0063
                                 - 0.338000890574714 z_k1_s0064
                                 - 0.237000373557485 z_k1_s0065
                                 - 0.235000645779517 z_k1_s0066
                                 - 0.552002245281067 z_k1_s0067
                                 - 0.339999200502756 z_k1_s0070
                                 - 0.122999912270044 z_k1_s0071
                                 - 0.34900035733918 z_k1_s0072
                                 - 0.548999422046847 z_k1_s0073
                                 - 0.186999823776423 z_k1_s0074
                                 - 0.264000557213491 z_k1_s0075
                                 - 0.406000657305666 z_k1_s0076
                                 - 0.416999143440102 z_k1_s0077
                                 - 0.342999731842405 z_k1_s0100
                                 - 0.327000785065875 z_k1_s0101
                                 - 0.263000391958094 z_k1_s0102
                                 - 0.22000014223256 z_k1_s0103
                                 - 0.246999142180734 z_k1_s0104
                                 - 0.168999958518594 z_k1_s0105
                                 - 0.599001304273527 z_k1_s0106
                                 - 0.135000066342699 z_k1_s0107
                                 - 0.159999668913602 z_k1_s0110
                                 - 0.0580000522312764 z_k1_s0111
                                 - 0.0780000373454914 z_k1_s0112
                                 - 0.156999867246579 z_k1_s0113
                                 - 0.16499969169217 z_k1_s0114
                                 - 0.0740000191226366 z_k1_s0115
                                 - 0.156999867246579 z_k1_s0116
                                 - 0.211000537255354 z_k1_s0117
                                 - 0.117999965224232 z_k1_s0120
                                 - 0.0570000831016064 z_k1_s0121
                                 - 0.0710000789756679 z_k1_s0122
                                 - 0.131999961098294 z_k1_s0123
                                 - 0.152999954909386 z_k1_s0124
                                 - 0.170999949604608 z_k1_s0125
                                 - 0.167999950488738 z_k1_s0126
                                 - 0.128999961982423 z_k1_s0127
                                 - 0.369999792224115 z_k1_s0130
                                 - 0.153999814233984 z_k1_s0131
                                 - 0.158999805220807 z_k1_s0132
                                 - 0.152999678276846 z_k1_s0133
                                 - 0.248998494985499 z_k1_s0134
                                 - 0.525998959973885 z_k1_s0135
                                 - 0.257998800201251 z_k1_s0136
                                 - 0.427000194593063 z_k1_s0137
                                 - 0.113000509123914 z_k1_s0140
                                 - 0.0770000914417708 z_k1_s0141
                                 - 0.295000459004267 z_k1_s0142
                                 - 0.129000136893344 z_k1_s0143
                                 - 0.130000113628934 z_k1_s0144
                                 - 0.124000253215397 z_k1_s0145
                                 - 0.0690000683569016 z_k1_s0146
                                 - 0.109000602181556 z_k1_s0147
                                 - 0.120000463496303 z_k1_s0150
                                 - 0.127000542578785 z_k1_s0151
                                 - 0.116000418306313 z_k1_s0152
                                 - 0.363001870483095 z_k1_s0153
                                 - 0.294001834458596 z_k1_s0154
                                 - 0.237000446998419 z_k1_s0155
                                 - 0.104000431436909 z_k1_s0156
                                 - 0.126000531281287 z_k1_s0157
                                 - 0.0720000359051263 z_k1_s0160
                                 - 0.162000027986494 z_k1_s0161
                                 - 0.11799964231086 z_k1_s0162
                                 - 0.148999602036636 z_k1_s0163
                                 - 0.171000322874857 z_k1_s0164
                                 - 0.0720000359051263 z_k1_s0165
                                 - 0.11799964231086 z_k1_s0166
                                 - 0.194000020477643 z_k1_s0167
                                 - 0.324999346262176 z_k1_s0170
                                 - 0.143999609753317 z_k1_s0171
                                 - 0.0749998834296546 z_k1_s0172
                                 - 0.206999327570647 z_k1_s0173
                                 - 0.101999698808656 z_k1_s0174
                                 - 0.0939998360665411 z_k1_s0175
                                 - 0.251999762867287 z_k1_s0176
                                 - 0.286999440988403 z_k1_s0177
                                 - 0.252000058346042 z_k1_s1000
                                 - 0.644999697462696 z_k1_s1001
                                 - 0.54599922266565 z_k1_s1002
                                 - 0.194000225667346 z_k1_s1003
                                 - 0.334000258081813 z_k1_s1004
                                 - 0.4650014632885 z_k1_s1005
                                 - 0.716999714132994 z_k1_s1006
                                 - 0.462001010718517 z_k1_s1007
                                 - 0.49500008086738 z_k1_s1010
                                 - 0.528002246262819 z_k1_s1011
                                 - 0.386999866859563 z_k1_s1012
                                 - 0.174000224790135 z_k1_s1013
                                 - 0.449000089716101 z_k1_s1014
                                 - 0.251000597370219 z_k1_s1015
                                 - 0.230000495757468 z_k1_s1016
                                 - 0.186000398569079 z_k1_s1017
                                 - 0.0950000458470985 z_k1_s1020
                                 - 0.171000082524777 z_k1_s1021
                                 - 0.0390000734215214 z_k1_s1022
                                 - 0.0879998084685443 z_k1_s1023
                                 - 0.0839998845382558 z_k1_s1024
                                 - 0.0590000635735132 z_k1_s1025
                                 - 0.130000491738771 z_k1_s1026
                                 - 0.0750004261956559 z_k1_s1027
                                 - 0.05400010153615 z_k1_s1030
                                 - 0.0900004585607083 z_k1_s1031
                                 - 0.116000571744882 z_k1_s1032
                                 - 0.113999828575706 z_k1_s1033
                                 - 0.166000054944054 z_k1_s1034
                                 - 0.129000628336969 z_k1_s1035
                                 - 0.0840001386562028 z_k1_s1036
                                 - 0.114999766159607 z_k1_s1037
                                 - 0.0690001834248814 z_k1_s1040
                                 - 0.0790002065168851 z_k1_s1041
                                 - 0.0810004038557328 z_k1_s1042
                                 - 0.0840004188133526 z_k1_s1043
                                 - 0.144000717965747 z_k1_s1044
                                 - 0.186000124370561 z_k1_s1045
                                 - 0.0390001141488703 z_k1_s1046
                                 - 0.121000335623375 z_k1_s1047
                                 - 0.271999928545416 z_k1_s1050
                                 - 0.84499922494426 z_k1_s1051
                                 - 0.324999701901638 z_k1_s1052
                                 - 0.250999925553305 z_k1_s1053
                                 - 0.421999123346004 z_k1_s1054
                                 - 0.248000173097351 z_k1_s1055
                                 - 0.284000095569521 z_k1_s1056
                                 - 0.245999952393285 z_k1_s1057
                                 - 0.0730001461741274 z_k1_s1060
                                 - 0.0500000868288213 z_k1_s1061
                                 - 0.0980001454884284 z_k1_s1062
                                 - 0.203000742369983 z_k1_s1063
                                 - 0.0750001743433416 z_k1_s1064
                                 - 0.0590001253900661 z_k1_s1065
                                 - 0.0890001951274028 z_k1_s1066
                                 - 0.235000663876096 z_k1_s1067
                                 - 0.0940004249167051 z_k1_s1070
                                 - 0.122000087316789 z_k1_s1071
                                 - 0.174000614632814 z_k1_s1072
                                 - 0.0510000603439364 z_k1_s1073
                                 - 0.112000265602553 z_k1_s1074
                                 - 0.0670000982871582 z_k1_s1075
                                 - 0.0490000960010891 z_k1_s1076
                                 - 0.139000430631878 z_k1_s1077
                                 - 0.379999595469091 z_k1_s1100
                                 - 0.28799963756293 z_k1_s1101
                                 - 0.917999890487343 z_k1_s1102
                                 - 0.86000075983938 z_k1_s1103
                                 - 0.183999915812322 z_k1_s1104
                                 - 0.404999783644536 z_k1_s1105
                                 - 0.574000402751009 z_k1_s1106
                                 - 0.471999961474961 z_k1_s1107
                                 - 0.363999263049581 z_k1_s1110
                                 - 0.874999258780163 z_k1_s1111
                                 - 1.12299649868972 z_k1_s1112
                                 - 1.1199962779253 z_k1_s1113
                                 - 0.333998946300801 z_k1_s1114
                                 - 1.03200030714358 z_k1_s1115
                                 - 0.887996013436126 z_k1_s1116
                                 - 0.566998864179126 z_k1_s1117
                                 - 0.570000523653474 z_k1_s1120
                                 - 0.51700045841183 z_k1_s1121
                                 - 0.630999982582337 z_k1_s1122
                                 - 0.886000159556242 z_k1_s1123
                                 - 0.85499824552939 z_k1_s1124
                                 - 1.89499925189364 z_k1_s1125
                                 - 1.09799896912235 z_k1_s1126
                                 - 0.804997499357806 z_k1_s1127
                                 - 0.396000787216072 z_k1_s1130
                                 - 0.465999142612765 z_k1_s1131
                                 - 0.243000112591662 z_k1_s1132
                                 - 0.510001116747976 z_k1_s1133
                                 - 0.487998431680288 z_k1_s1134
                                 - 0.562000671271272 z_k1_s1135
                                 - 0.48600158358339 z_k1_s1136
                                 - 0.245000017088708 z_k1_s1137
                                 - 1.20700220094793 z_k1_s1140
                                 - 0.976001264068485 z_k1_s1141
                                 - 1.45700389700028 z_k1_s1142
                                 - 1.70999938946009 z_k1_s1143
                                 - 0.373000442731032 z_k1_s1144
                                 - 1.46400583530853 z_k1_s1145
                                 - 1.20300390706164 z_k1_s1146
                                 - 1.4579985464646 z_k1_s1147
                                 - 0.586997871032971 z_k1_s1150
                                 - 0.246999838399041 z_k1_s1151
                                 - 0.279999709256099 z_k1_s1152
                                 - 0.818997732087929 z_k1_s1153
                                 - 0.610997993383355 z_k1_s1154
                                 - 0.714999845233771 z_k1_s1155
                                 - 0.233999805167552 z_k1_s1156
                                 - 0.472001084558796 z_k1_s1157
                                 - 1.65500445585079 z_k1_s1160
                                 - 0.568000738152427 z_k1_s1161
                                 - 0.544000584964158 z_k1_s1162
                                 - 0.89700072815589 z_k1_s1163
                                 - 0.461000832538942 z_k1_s1164
                                 - 0.68500381699788 z_k1_s1165
                                 - 0.754998711290707 z_k1_s1166
                                 - 1.09800367685953 z_k1_s1167
                                 - 0.375000449072724 z_k1_s1170
                                 - 0.453000514695865 z_k1_s1171
                                 - 0.234999725739201 z_k1_s1172
                                 - 0.218999970526625 z_k1_s1173
                                 - 0.403999983689279 z_k1_s1174
                                 - 0.358000123090922 z_k1_s1175
                                 - 1.08499935443811 z_k1_s1176
                                 - 0.343999772917717 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.991965 w_k1_ADM_s000
                                 + 1.02981 w_k1_AXP_s000
                                 + 0.95207 w_k1_DIS_s000
                                 + 1.08152 w_k1_FDX_s000
                                 + 0.74704 w_k1_IBM_s000
                                 + 0.929875 w_k1_INTC_s000
                                 + 0.901185 w_k1_KO_s000
                                 + 0.690874 w_k1_MAT_s000
                                 + 0.974262 w_k1_MSFT_s000
                                 + 0.954707 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.29389 w_k1_ADM_s000
                                 + 1.45359 w_k1_AXP_s000 + 1.399 w_k1_DIS_s000
                                 + 1.24906 w_k1_FDX_s000
                                 + 1.26367 w_k1_IBM_s000
                                 + 1.24337 w_k1_INTC_s000
                                 + 1.29171 w_k1_KO_s000 + 1.15521 w_k1_MAT_s000
                                 + 1.05797 w_k1_MSFT_s000
                                 + 1.20307 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.700681 w_k1_ADM_s000
                                 + 0.872025 w_k1_AXP_s000
                                 + 0.727302 w_k1_DIS_s000
                                 + 0.809124 w_k1_FDX_s000
                                 + 1.07747 w_k1_IBM_s000
                                 + 1.05586 w_k1_INTC_s000
                                 + 1.01489 w_k1_KO_s000
                                 + 0.847225 w_k1_MAT_s000
                                 + 0.835117 w_k1_MSFT_s000
                                 + 0.857681 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.795001 w_k1_ADM_s000
                                 + 0.718813 w_k1_AXP_s000
                                 + 0.816377 w_k1_DIS_s000
                                 + 0.957382 w_k1_FDX_s000
                                 + 0.705565 w_k1_IBM_s000
                                 + 1.16107 w_k1_INTC_s000
                                 + 0.87889 w_k1_KO_s000
                                 + 0.475899 w_k1_MAT_s000
                                 + 0.946368 w_k1_MSFT_s000
                                 + 0.888431 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.83499 w_k1_ADM_s000
                                 + 1.13311 w_k1_AXP_s000
                                 + 0.828949 w_k1_DIS_s000
                                 + 0.886643 w_k1_FDX_s000
                                 + 1.44546 w_k1_IBM_s000
                                 + 1.18737 w_k1_INTC_s000
                                 + 1.23822 w_k1_KO_s000 + 1.02286 w_k1_MAT_s000
                                 + 0.927561 w_k1_MSFT_s000
                                 + 1.10586 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.924046 w_k1_ADM_s000
                                 + 0.920345 w_k1_AXP_s000
                                 + 1.01502 w_k1_DIS_s000
                                 + 1.02276 w_k1_FDX_s000
                                 + 0.914466 w_k1_IBM_s000
                                 + 1.05719 w_k1_INTC_s000
                                 + 1.05576 w_k1_KO_s000
                                 + 0.739793 w_k1_MAT_s000
                                 + 0.999247 w_k1_MSFT_s000
                                 + 0.987329 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.10773 w_k1_ADM_s000
                                 + 0.954581 w_k1_AXP_s000
                                 + 1.28164 w_k1_DIS_s000
                                 + 1.12281 w_k1_FDX_s000
                                 + 0.975078 w_k1_IBM_s000
                                 + 1.30507 w_k1_INTC_s000
                                 + 1.27524 w_k1_KO_s000 + 0.68251 w_k1_MAT_s000
                                 + 1.16817 w_k1_MSFT_s000
                                 + 1.02444 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.774173 w_k1_ADM_s000
                                 + 0.706761 w_k1_AXP_s000
                                 + 0.659797 w_k1_DIS_s000
                                 + 0.771915 w_k1_FDX_s000
                                 + 0.764324 w_k1_IBM_s000
                                 + 1.10095 w_k1_INTC_s000
                                 + 0.806742 w_k1_KO_s000
                                 + 0.479876 w_k1_MAT_s000
                                 + 0.829245 w_k1_MSFT_s000
                                 + 0.775355 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.00427 w_k1_ADM_s001
                                 + 0.976737 w_k1_AXP_s001
                                 + 0.7859 w_k1_DIS_s001
                                 + 0.880588 w_k1_FDX_s001
                                 + 0.980594 w_k1_IBM_s001
                                 + 0.91139 w_k1_INTC_s001
                                 + 0.824021 w_k1_KO_s001
                                 + 0.856575 w_k1_MAT_s001
                                 + 0.954004 w_k1_MSFT_s001
                                 + 1.04402 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.004 w_k1_ADM_s001
                                 + 0.691955 w_k1_AXP_s001
                                 + 1.08547 w_k1_DIS_s001
                                 + 0.869632 w_k1_FDX_s001
                                 + 0.942084 w_k1_IBM_s001
                                 + 0.829128 w_k1_INTC_s001
                                 + 0.932429 w_k1_KO_s001
                                 + 0.611295 w_k1_MAT_s001
                                 + 0.925074 w_k1_MSFT_s001
                                 + 1.00888 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.37769 w_k1_ADM_s001
                                 + 1.11705 w_k1_AXP_s001
                                 + 1.09685 w_k1_DIS_s001 + 1.2731 w_k1_FDX_s001
                                 + 0.778974 w_k1_IBM_s001
                                 + 1.21572 w_k1_INTC_s001
                                 + 0.869285 w_k1_KO_s001
                                 + 0.688611 w_k1_MAT_s001
                                 + 1.19327 w_k1_MSFT_s001
                                 + 1.09165 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.912874 w_k1_ADM_s001
                                 + 0.923855 w_k1_AXP_s001
                                 + 0.849939 w_k1_DIS_s001
                                 + 0.861942 w_k1_FDX_s001
                                 + 1.44493 w_k1_IBM_s001
                                 + 0.977777 w_k1_INTC_s001
                                 + 1.14052 w_k1_KO_s001 + 1.1967 w_k1_MAT_s001
                                 + 0.866613 w_k1_MSFT_s001
                                 + 0.953692 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.11514 w_k1_ADM_s001
                                 + 0.65442 w_k1_AXP_s001
                                 + 0.805681 w_k1_DIS_s001
                                 + 0.885708 w_k1_FDX_s001
                                 + 0.640296 w_k1_IBM_s001
                                 + 1.0789 w_k1_INTC_s001
                                 + 0.871329 w_k1_KO_s001
                                 + 0.427645 w_k1_MAT_s001
                                 + 1.03476 w_k1_MSFT_s001
                                 + 0.946503 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.36611 w_k1_ADM_s001
                                 + 1.34193 w_k1_AXP_s001
                                 + 1.52175 w_k1_DIS_s001
                                 + 1.30273 w_k1_FDX_s001
                                 + 1.22055 w_k1_IBM_s001
                                 + 1.14497 w_k1_INTC_s001
                                 + 1.12542 w_k1_KO_s001
                                 + 0.862708 w_k1_MAT_s001
                                 + 1.27269 w_k1_MSFT_s001
                                 + 1.11793 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.85554 w_k1_ADM_s001
                                 + 0.559977 w_k1_AXP_s001
                                 + 0.533335 w_k1_DIS_s001
                                 + 0.667596 w_k1_FDX_s001
                                 + 0.669565 w_k1_IBM_s001
                                 + 0.737499 w_k1_INTC_s001
                                 + 0.572442 w_k1_KO_s001
                                 + 0.406157 w_k1_MAT_s001
                                 + 0.672683 w_k1_MSFT_s001
                                 + 0.841265 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.07617 w_k1_ADM_s001
                                 + 1.01607 w_k1_AXP_s001
                                 + 0.89962 w_k1_DIS_s001
                                 + 1.00806 w_k1_FDX_s001
                                 + 0.696566 w_k1_IBM_s001
                                 + 1.03877 w_k1_INTC_s001
                                 + 0.81962 w_k1_KO_s001
                                 + 0.493785 w_k1_MAT_s001
                                 + 0.998057 w_k1_MSFT_s001
                                 + 0.909008 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.898781 w_k1_ADM_s002
                                 + 0.699436 w_k1_AXP_s002
                                 + 0.936787 w_k1_DIS_s002
                                 + 0.748494 w_k1_FDX_s002
                                 + 1.05713 w_k1_IBM_s002
                                 + 0.874573 w_k1_INTC_s002
                                 + 0.996914 w_k1_KO_s002
                                 + 0.776742 w_k1_MAT_s002
                                 + 0.834059 w_k1_MSFT_s002
                                 + 0.897979 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.891769 w_k1_ADM_s002
                                 + 0.537511 w_k1_AXP_s002
                                 + 0.76335 w_k1_DIS_s002
                                 + 0.765399 w_k1_FDX_s002
                                 + 0.711626 w_k1_IBM_s002
                                 + 1.04266 w_k1_INTC_s002
                                 + 0.856189 w_k1_KO_s002
                                 + 0.580866 w_k1_MAT_s002
                                 + 0.875929 w_k1_MSFT_s002
                                 + 0.870624 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.882088 w_k1_ADM_s002
                                 + 1.1485 w_k1_AXP_s002 + 1.05201 w_k1_DIS_s002
                                 + 1.11479 w_k1_FDX_s002
                                 + 0.940825 w_k1_IBM_s002
                                 + 1.14296 w_k1_INTC_s002 + 1.0775 w_k1_KO_s002
                                 + 1.46768 w_k1_MAT_s002
                                 + 1.10758 w_k1_MSFT_s002
                                 + 1.12678 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.871166 w_k1_ADM_s002
                                 + 0.821266 w_k1_AXP_s002
                                 + 0.733048 w_k1_DIS_s002
                                 + 0.88047 w_k1_FDX_s002
                                 + 1.21754 w_k1_IBM_s002
                                 + 0.779508 w_k1_INTC_s002
                                 + 0.988734 w_k1_KO_s002
                                 + 0.982143 w_k1_MAT_s002
                                 + 0.737385 w_k1_MSFT_s002
                                 + 0.951705 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.713389 w_k1_ADM_s002
                                 + 0.904686 w_k1_AXP_s002
                                 + 0.707241 w_k1_DIS_s002
                                 + 0.728338 w_k1_FDX_s002
                                 + 1.07933 w_k1_IBM_s002
                                 + 0.81643 w_k1_INTC_s002
                                 + 0.859497 w_k1_KO_s002
                                 + 1.02733 w_k1_MAT_s002
                                 + 0.749381 w_k1_MSFT_s002
                                 + 0.872196 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.885992 w_k1_ADM_s002
                                 + 1.02629 w_k1_AXP_s002
                                 + 0.911449 w_k1_DIS_s002
                                 + 0.980578 w_k1_FDX_s002
                                 + 0.67047 w_k1_IBM_s002
                                 + 0.929984 w_k1_INTC_s002
                                 + 0.888368 w_k1_KO_s002
                                 + 0.994315 w_k1_MAT_s002
                                 + 1.17613 w_k1_MSFT_s002
                                 + 0.907763 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.827605 w_k1_ADM_s002
                                 + 0.737452 w_k1_AXP_s002
                                 + 0.943411 w_k1_DIS_s002
                                 + 0.819517 w_k1_FDX_s002
                                 + 1.45593 w_k1_IBM_s002
                                 + 0.84396 w_k1_INTC_s002
                                 + 1.24525 w_k1_KO_s002 + 1.24839 w_k1_MAT_s002
                                 + 0.876194 w_k1_MSFT_s002
                                 + 0.962025 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.00485 w_k1_ADM_s002
                                 + 1.31853 w_k1_AXP_s002 + 1.5696 w_k1_DIS_s002
                                 + 1.26594 w_k1_FDX_s002
                                 + 1.31629 w_k1_IBM_s002
                                 + 1.44494 w_k1_INTC_s002
                                 + 1.49361 w_k1_KO_s002 + 1.69137 w_k1_MAT_s002
                                 + 1.31751 w_k1_MSFT_s002
                                 + 1.28012 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.20595 w_k1_ADM_s003
                                 + 1.12542 w_k1_AXP_s003
                                 + 1.21146 w_k1_DIS_s003
                                 + 1.09108 w_k1_FDX_s003 + 1.13 w_k1_IBM_s003
                                 + 1.00148 w_k1_INTC_s003
                                 + 1.09217 w_k1_KO_s003 + 1.19478 w_k1_MAT_s003
                                 + 1.40024 w_k1_MSFT_s003
                                 + 1.08972 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.807326 w_k1_ADM_s003
                                 + 0.716892 w_k1_AXP_s003
                                 + 0.649143 w_k1_DIS_s003
                                 + 0.796325 w_k1_FDX_s003
                                 + 0.736059 w_k1_IBM_s003
                                 + 0.900305 w_k1_INTC_s003
                                 + 0.726169 w_k1_KO_s003
                                 + 0.585854 w_k1_MAT_s003
                                 + 1.02223 w_k1_MSFT_s003
                                 + 0.772064 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.37732 w_k1_ADM_s003
                                 + 1.31868 w_k1_AXP_s003
                                 + 1.59138 w_k1_DIS_s003
                                 + 1.43157 w_k1_FDX_s003
                                 + 1.30733 w_k1_IBM_s003
                                 + 1.31157 w_k1_INTC_s003
                                 + 1.35202 w_k1_KO_s003 + 1.51681 w_k1_MAT_s003
                                 + 1.49911 w_k1_MSFT_s003
                                 + 1.23107 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.1588 w_k1_ADM_s003
                                 + 1.2228 w_k1_AXP_s003 + 1.0947 w_k1_DIS_s003
                                 + 1.22969 w_k1_FDX_s003
                                 + 0.812501 w_k1_IBM_s003
                                 + 1.16829 w_k1_INTC_s003
                                 + 1.03884 w_k1_KO_s003
                                 + 0.899569 w_k1_MAT_s003
                                 + 1.27799 w_k1_MSFT_s003
                                 + 1.00168 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.931616 w_k1_ADM_s003
                                 + 0.782164 w_k1_AXP_s003
                                 + 0.874203 w_k1_DIS_s003
                                 + 0.793179 w_k1_FDX_s003
                                 + 1.20966 w_k1_IBM_s003
                                 + 0.997686 w_k1_INTC_s003
                                 + 1.16202 w_k1_KO_s003 + 1.19325 w_k1_MAT_s003
                                 + 1.06086 w_k1_MSFT_s003
                                 + 0.995902 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.983753 w_k1_ADM_s003
                                 + 0.8025 w_k1_AXP_s003
                                 + 0.896446 w_k1_DIS_s003
                                 + 0.904142 w_k1_FDX_s003
                                 + 0.770972 w_k1_IBM_s003
                                 + 0.923681 w_k1_INTC_s003
                                 + 0.847688 w_k1_KO_s003
                                 + 0.656432 w_k1_MAT_s003
                                 + 1.09274 w_k1_MSFT_s003
                                 + 0.897112 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.2072 w_k1_ADM_s003
                                 + 1.12813 w_k1_AXP_s003
                                 + 0.959701 w_k1_DIS_s003
                                 + 0.933299 w_k1_FDX_s003
                                 + 1.23259 w_k1_IBM_s003
                                 + 0.800728 w_k1_INTC_s003
                                 + 1.08912 w_k1_KO_s003 + 1.29577 w_k1_MAT_s003
                                 + 1.14563 w_k1_MSFT_s003
                                 + 0.983066 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.1471 w_k1_ADM_s003
                                 + 1.55858 w_k1_AXP_s003
                                 + 1.13637 w_k1_DIS_s003
                                 + 1.07345 w_k1_FDX_s003
                                 + 1.01864 w_k1_IBM_s003
                                 + 1.02651 w_k1_INTC_s003
                                 + 1.01248 w_k1_KO_s003 + 1.2101 w_k1_MAT_s003
                                 + 1.09784 w_k1_MSFT_s003
                                 + 1.03806 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.877367 w_k1_ADM_s004
                                 + 1.06055 w_k1_AXP_s004 + 1.0972 w_k1_DIS_s004
                                 + 0.941949 w_k1_FDX_s004
                                 + 1.35196 w_k1_IBM_s004
                                 + 1.26084 w_k1_INTC_s004
                                 + 1.24724 w_k1_KO_s004 + 1.1965 w_k1_MAT_s004
                                 + 1.0891 w_k1_MSFT_s004 + 1.03855 w_k1_PG_s004
                                 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.06457 w_k1_ADM_s004
                                 + 1.09875 w_k1_AXP_s004
                                 + 1.62717 w_k1_DIS_s004
                                 + 1.28546 w_k1_FDX_s004
                                 + 1.43936 w_k1_IBM_s004
                                 + 1.33739 w_k1_INTC_s004
                                 + 1.38355 w_k1_KO_s004 + 1.36115 w_k1_MAT_s004
                                 + 1.35685 w_k1_MSFT_s004
                                 + 1.13876 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.758093 w_k1_ADM_s004
                                 + 0.7489 w_k1_AXP_s004
                                 + 0.724159 w_k1_DIS_s004
                                 + 0.846777 w_k1_FDX_s004
                                 + 0.583102 w_k1_IBM_s004
                                 + 1.05566 w_k1_INTC_s004
                                 + 0.740795 w_k1_KO_s004
                                 + 0.419305 w_k1_MAT_s004
                                 + 1.01108 w_k1_MSFT_s004
                                 + 0.808596 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.11363 w_k1_ADM_s004
                                 + 1.5104 w_k1_AXP_s004 + 1.45294 w_k1_DIS_s004
                                 + 1.28775 w_k1_FDX_s004
                                 + 1.44747 w_k1_IBM_s004
                                 + 1.32699 w_k1_INTC_s004
                                 + 1.25848 w_k1_KO_s004 + 1.59693 w_k1_MAT_s004
                                 + 1.18336 w_k1_MSFT_s004
                                 + 1.12203 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.12239 w_k1_ADM_s004
                                 + 1.07124 w_k1_AXP_s004
                                 + 1.15761 w_k1_DIS_s004
                                 + 1.16149 w_k1_FDX_s004
                                 + 0.86473 w_k1_IBM_s004
                                 + 1.24827 w_k1_INTC_s004
                                 + 0.993415 w_k1_KO_s004
                                 + 0.802232 w_k1_MAT_s004
                                 + 1.18681 w_k1_MSFT_s004
                                 + 0.895186 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.05498 w_k1_ADM_s004
                                 + 0.824617 w_k1_AXP_s004
                                 + 1.02887 w_k1_DIS_s004
                                 + 0.950317 w_k1_FDX_s004
                                 + 0.857681 w_k1_IBM_s004
                                 + 1.05311 w_k1_INTC_s004
                                 + 0.962336 w_k1_KO_s004
                                 + 0.722425 w_k1_MAT_s004
                                 + 0.967383 w_k1_MSFT_s004
                                 + 0.958858 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.680949 w_k1_ADM_s004
                                 + 0.563379 w_k1_AXP_s004
                                 + 0.507121 w_k1_DIS_s004
                                 + 0.574158 w_k1_FDX_s004
                                 + 0.736968 w_k1_IBM_s004
                                 + 0.84059 w_k1_INTC_s004
                                 + 0.657799 w_k1_KO_s004
                                 + 0.511673 w_k1_MAT_s004
                                 + 0.746656 w_k1_MSFT_s004
                                 + 0.700132 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.754341 w_k1_ADM_s004
                                 + 0.694336 w_k1_AXP_s004
                                 + 0.692002 w_k1_DIS_s004
                                 + 0.743061 w_k1_FDX_s004
                                 + 0.931924 w_k1_IBM_s004
                                 + 0.79078 w_k1_INTC_s004
                                 + 0.835039 w_k1_KO_s004
                                 + 1.01354 w_k1_MAT_s004
                                 + 0.833187 w_k1_MSFT_s004
                                 + 0.823697 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.16627 w_k1_ADM_s005
                                 + 1.2343 w_k1_AXP_s005 + 1.34339 w_k1_DIS_s005
                                 + 1.38435 w_k1_FDX_s005
                                 + 0.848823 w_k1_IBM_s005
                                 + 1.15346 w_k1_INTC_s005
                                 + 1.05501 w_k1_KO_s005
                                 + 0.901149 w_k1_MAT_s005
                                 + 1.34975 w_k1_MSFT_s005
                                 + 1.27271 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.08357 w_k1_ADM_s005
                                 + 0.830014 w_k1_AXP_s005
                                 + 0.853346 w_k1_DIS_s005
                                 + 0.881848 w_k1_FDX_s005
                                 + 1.24378 w_k1_IBM_s005
                                 + 0.887343 w_k1_INTC_s005
                                 + 0.963351 w_k1_KO_s005
                                 + 1.01634 w_k1_MAT_s005
                                 + 0.947542 w_k1_MSFT_s005
                                 + 1.02202 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.30046 w_k1_ADM_s005
                                 + 1.29178 w_k1_AXP_s005
                                 + 1.89172 w_k1_DIS_s005
                                 + 1.46712 w_k1_FDX_s005
                                 + 1.28074 w_k1_IBM_s005
                                 + 1.25425 w_k1_INTC_s005
                                 + 1.48072 w_k1_KO_s005 + 1.70303 w_k1_MAT_s005
                                 + 1.53569 w_k1_MSFT_s005
                                 + 1.19419 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.05173 w_k1_ADM_s005
                                 + 0.790654 w_k1_AXP_s005
                                 + 1.00355 w_k1_DIS_s005
                                 + 1.04009 w_k1_FDX_s005
                                 + 0.852733 w_k1_IBM_s005
                                 + 1.02655 w_k1_INTC_s005
                                 + 0.952233 w_k1_KO_s005
                                 + 0.589689 w_k1_MAT_s005
                                 + 1.03983 w_k1_MSFT_s005
                                 + 1.08318 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.10416 w_k1_ADM_s005
                                 + 1.23498 w_k1_AXP_s005
                                 + 0.906859 w_k1_DIS_s005
                                 + 1.05122 w_k1_FDX_s005
                                 + 1.12991 w_k1_IBM_s005
                                 + 1.05447 w_k1_INTC_s005
                                 + 1.01139 w_k1_KO_s005 + 1.24062 w_k1_MAT_s005
                                 + 1.05789 w_k1_MSFT_s005
                                 + 1.12486 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.804427 w_k1_ADM_s005
                                 + 0.604201 w_k1_AXP_s005
                                 + 0.768078 w_k1_DIS_s005
                                 + 0.859113 w_k1_FDX_s005
                                 + 0.625172 w_k1_IBM_s005
                                 + 0.829677 w_k1_INTC_s005
                                 + 0.661565 w_k1_KO_s005
                                 + 0.430645 w_k1_MAT_s005
                                 + 0.895517 w_k1_MSFT_s005
                                 + 0.917315 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.26391 w_k1_ADM_s005
                                 + 1.0977 w_k1_AXP_s005 + 1.31106 w_k1_DIS_s005
                                 + 1.1694 w_k1_FDX_s005 + 1.39643 w_k1_IBM_s005
                                 + 1.3006 w_k1_INTC_s005 + 1.23787 w_k1_KO_s005
                                 + 1.34654 w_k1_MAT_s005
                                 + 1.38562 w_k1_MSFT_s005
                                 + 1.16187 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.873894 w_k1_ADM_s005
                                 + 0.769256 w_k1_AXP_s005
                                 + 0.613113 w_k1_DIS_s005
                                 + 0.748662 w_k1_FDX_s005
                                 + 0.743439 w_k1_IBM_s005
                                 + 0.795137 w_k1_INTC_s005
                                 + 0.709482 w_k1_KO_s005
                                 + 0.692334 w_k1_MAT_s005
                                 + 0.803462 w_k1_MSFT_s005
                                 + 0.904181 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.811899 w_k1_ADM_s006
                                 + 1.29763 w_k1_AXP_s006
                                 + 1.24478 w_k1_DIS_s006
                                 + 1.07217 w_k1_FDX_s006
                                 + 1.35432 w_k1_IBM_s006
                                 + 1.04516 w_k1_INTC_s006
                                 + 1.26507 w_k1_KO_s006 + 1.56449 w_k1_MAT_s006
                                 + 0.908325 w_k1_MSFT_s006
                                 + 1.13686 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.866458 w_k1_ADM_s006
                                 + 0.788012 w_k1_AXP_s006
                                 + 0.660804 w_k1_DIS_s006
                                 + 0.834664 w_k1_FDX_s006
                                 + 0.647504 w_k1_IBM_s006
                                 + 0.947078 w_k1_INTC_s006
                                 + 0.72627 w_k1_KO_s006
                                 + 0.580098 w_k1_MAT_s006
                                 + 0.852914 w_k1_MSFT_s006
                                 + 0.795452 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.73243 w_k1_ADM_s006
                                 + 0.857334 w_k1_AXP_s006
                                 + 0.674378 w_k1_DIS_s006
                                 + 0.802872 w_k1_FDX_s006
                                 + 0.971956 w_k1_IBM_s006
                                 + 0.829064 w_k1_INTC_s006
                                 + 0.814844 w_k1_KO_s006
                                 + 0.994465 w_k1_MAT_s006
                                 + 0.818519 w_k1_MSFT_s006
                                 + 0.852928 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.03879 w_k1_ADM_s006
                                 + 1.18298 w_k1_AXP_s006 + 1.3203 w_k1_DIS_s006
                                 + 1.22159 w_k1_FDX_s006
                                 + 0.916222 w_k1_IBM_s006
                                 + 1.10713 w_k1_INTC_s006
                                 + 1.12623 w_k1_KO_s006
                                 + 0.877955 w_k1_MAT_s006
                                 + 1.10562 w_k1_MSFT_s006
                                 + 1.19205 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.40486 w_k1_ADM_s006
                                 + 1.19994 w_k1_AXP_s006
                                 + 1.04873 w_k1_DIS_s006
                                 + 1.13492 w_k1_FDX_s006
                                 + 0.937398 w_k1_IBM_s006
                                 + 0.98001 w_k1_INTC_s006
                                 + 0.946304 w_k1_KO_s006
                                 + 0.814568 w_k1_MAT_s006
                                 + 0.953956 w_k1_MSFT_s006
                                 + 0.982135 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.26276 w_k1_ADM_s006
                                 + 0.915661 w_k1_AXP_s006
                                 + 0.969401 w_k1_DIS_s006
                                 + 1.04676 w_k1_FDX_s006
                                 + 0.648149 w_k1_IBM_s006
                                 + 0.863645 w_k1_INTC_s006
                                 + 0.876721 w_k1_KO_s006
                                 + 0.532212 w_k1_MAT_s006
                                 + 0.923498 w_k1_MSFT_s006
                                 + 0.951464 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.50467 w_k1_ADM_s006
                                 + 1.81709 w_k1_AXP_s006
                                 + 1.71594 w_k1_DIS_s006
                                 + 1.45659 w_k1_FDX_s006
                                 + 1.34323 w_k1_IBM_s006
                                 + 1.22633 w_k1_INTC_s006
                                 + 1.42931 w_k1_KO_s006 + 1.80699 w_k1_MAT_s006
                                 + 1.258 w_k1_MSFT_s006 + 1.32627 w_k1_PG_s006
                                 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.904401 w_k1_ADM_s006
                                 + 0.806524 w_k1_AXP_s006
                                 + 1.00193 w_k1_DIS_s006
                                 + 0.897357 w_k1_FDX_s006
                                 + 0.942354 w_k1_IBM_s006
                                 + 0.850518 w_k1_INTC_s006
                                 + 0.894142 w_k1_KO_s006
                                 + 0.862078 w_k1_MAT_s006
                                 + 0.825174 w_k1_MSFT_s006
                                 + 0.822806 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.71452 w_k1_ADM_s007
                                 + 0.955928 w_k1_AXP_s007
                                 + 0.925046 w_k1_DIS_s007
                                 + 0.907387 w_k1_FDX_s007
                                 + 1.18854 w_k1_IBM_s007
                                 + 1.11863 w_k1_INTC_s007 + 1.0636 w_k1_KO_s007
                                 + 1.00561 w_k1_MAT_s007
                                 + 0.984315 w_k1_MSFT_s007
                                 + 0.900164 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.48903 w_k1_ADM_s007
                                 + 1.64909 w_k1_AXP_s007
                                 + 1.27096 w_k1_DIS_s007
                                 + 1.32426 w_k1_FDX_s007
                                 + 1.47542 w_k1_IBM_s007
                                 + 1.26263 w_k1_INTC_s007 + 1.1704 w_k1_KO_s007
                                 + 1.38397 w_k1_MAT_s007
                                 + 1.23066 w_k1_MSFT_s007
                                 + 1.17988 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.42604 w_k1_ADM_s007
                                 + 1.29318 w_k1_AXP_s007
                                 + 1.29243 w_k1_DIS_s007
                                 + 1.33658 w_k1_FDX_s007 + 1.03 w_k1_IBM_s007
                                 + 1.20721 w_k1_INTC_s007
                                 + 1.19364 w_k1_KO_s007 + 1.11638 w_k1_MAT_s007
                                 + 1.30843 w_k1_MSFT_s007
                                 + 1.12151 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.886466 w_k1_ADM_s007
                                 + 0.943532 w_k1_AXP_s007
                                 + 0.972604 w_k1_DIS_s007
                                 + 1.06394 w_k1_FDX_s007
                                 + 0.782557 w_k1_IBM_s007
                                 + 1.11038 w_k1_INTC_s007
                                 + 0.900148 w_k1_KO_s007
                                 + 0.759365 w_k1_MAT_s007
                                 + 1.10314 w_k1_MSFT_s007
                                 + 1.01243 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.07294 w_k1_ADM_s007
                                 + 1.2461 w_k1_AXP_s007 + 1.26284 w_k1_DIS_s007
                                 + 1.01034 w_k1_FDX_s007
                                 + 1.62167 w_k1_IBM_s007
                                 + 1.38084 w_k1_INTC_s007
                                 + 1.34819 w_k1_KO_s007 + 1.5998 w_k1_MAT_s007
                                 + 1.27231 w_k1_MSFT_s007
                                 + 1.08552 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.14664 w_k1_ADM_s007
                                 + 1.20232 w_k1_AXP_s007
                                 + 1.14033 w_k1_DIS_s007
                                 + 1.11329 w_k1_FDX_s007
                                 + 1.24236 w_k1_IBM_s007
                                 + 1.1639 w_k1_INTC_s007 + 1.27173 w_k1_KO_s007
                                 + 1.1759 w_k1_MAT_s007
                                 + 1.16587 w_k1_MSFT_s007
                                 + 1.07624 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.878593 w_k1_ADM_s007
                                 + 0.752971 w_k1_AXP_s007
                                 + 0.845111 w_k1_DIS_s007
                                 + 0.793668 w_k1_FDX_s007
                                 + 1.02166 w_k1_IBM_s007
                                 + 1.04613 w_k1_INTC_s007
                                 + 0.908179 w_k1_KO_s007
                                 + 0.646447 w_k1_MAT_s007
                                 + 0.919451 w_k1_MSFT_s007
                                 + 0.952186 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.839208 w_k1_ADM_s007
                                 + 0.761551 w_k1_AXP_s007
                                 + 0.649072 w_k1_DIS_s007
                                 + 0.776161 w_k1_FDX_s007
                                 + 0.731246 w_k1_IBM_s007
                                 + 0.985865 w_k1_INTC_s007
                                 + 0.734181 w_k1_KO_s007
                                 + 0.626189 w_k1_MAT_s007
                                 + 0.86406 w_k1_MSFT_s007
                                 + 0.782989 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.00004 w_k1_ADM_s010
                                 + 0.85072 w_k1_AXP_s010 + 1.1119 w_k1_DIS_s010
                                 + 1.00001 w_k1_FDX_s010
                                 + 0.990585 w_k1_IBM_s010
                                 + 1.09646 w_k1_INTC_s010
                                 + 1.09919 w_k1_KO_s010
                                 + 0.830756 w_k1_MAT_s010
                                 + 1.12251 w_k1_MSFT_s010
                                 + 0.994022 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.02335 w_k1_ADM_s010
                                 + 0.929957 w_k1_AXP_s010
                                 + 0.912106 w_k1_DIS_s010
                                 + 1.07117 w_k1_FDX_s010
                                 + 0.67888 w_k1_IBM_s010
                                 + 1.15496 w_k1_INTC_s010
                                 + 0.825497 w_k1_KO_s010 + 0.6272 w_k1_MAT_s010
                                 + 0.975939 w_k1_MSFT_s010
                                 + 0.971037 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.28987 w_k1_ADM_s010
                                 + 1.47083 w_k1_AXP_s010
                                 + 1.15036 w_k1_DIS_s010
                                 + 1.23417 w_k1_FDX_s010
                                 + 0.996278 w_k1_IBM_s010
                                 + 1.16792 w_k1_INTC_s010
                                 + 1.09657 w_k1_KO_s010 + 1.12745 w_k1_MAT_s010
                                 + 1.00849 w_k1_MSFT_s010
                                 + 1.09757 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.68337 w_k1_ADM_s010
                                 + 0.690772 w_k1_AXP_s010
                                 + 0.824931 w_k1_DIS_s010
                                 + 0.787801 w_k1_FDX_s010
                                 + 1.0026 w_k1_IBM_s010
                                 + 0.897941 w_k1_INTC_s010
                                 + 0.883402 w_k1_KO_s010
                                 + 0.884847 w_k1_MAT_s010
                                 + 0.893737 w_k1_MSFT_s010
                                 + 0.850571 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.693753 w_k1_ADM_s010
                                 + 0.727391 w_k1_AXP_s010
                                 + 0.662688 w_k1_DIS_s010
                                 + 0.7608 w_k1_FDX_s010
                                 + 0.775773 w_k1_IBM_s010
                                 + 0.668872 w_k1_INTC_s010
                                 + 0.699198 w_k1_KO_s010
                                 + 0.986629 w_k1_MAT_s010
                                 + 0.651931 w_k1_MSFT_s010
                                 + 0.853016 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.08015 w_k1_ADM_s010
                                 + 1.62312 w_k1_AXP_s010
                                 + 1.79206 w_k1_DIS_s010
                                 + 1.53034 w_k1_FDX_s010
                                 + 1.19882 w_k1_IBM_s010
                                 + 1.31343 w_k1_INTC_s010
                                 + 1.39775 w_k1_KO_s010 + 1.74383 w_k1_MAT_s010
                                 + 1.3433 w_k1_MSFT_s010 + 1.27749 w_k1_PG_s010
                                 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.910085 w_k1_ADM_s010
                                 + 1.02114 w_k1_AXP_s010
                                 + 1.05559 w_k1_DIS_s010
                                 + 1.03313 w_k1_FDX_s010
                                 + 1.35139 w_k1_IBM_s010
                                 + 1.14937 w_k1_INTC_s010
                                 + 1.19439 w_k1_KO_s010 + 1.57294 w_k1_MAT_s010
                                 + 1.08491 w_k1_MSFT_s010 + 1.0693 w_k1_PG_s010
                                 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.02562 w_k1_ADM_s010
                                 + 1.17796 w_k1_AXP_s010
                                 + 1.32659 w_k1_DIS_s010
                                 + 1.32832 w_k1_FDX_s010
                                 + 0.777548 w_k1_IBM_s010
                                 + 1.22946 w_k1_INTC_s010
                                 + 1.05158 w_k1_KO_s010 + 1.13295 w_k1_MAT_s010
                                 + 1.23892 w_k1_MSFT_s010
                                 + 0.991598 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.17348 w_k1_ADM_s011
                                 + 0.833317 w_k1_AXP_s011
                                 + 1.01402 w_k1_DIS_s011
                                 + 1.05482 w_k1_FDX_s011
                                 + 0.869337 w_k1_IBM_s011
                                 + 0.783592 w_k1_INTC_s011
                                 + 0.985561 w_k1_KO_s011
                                 + 0.827887 w_k1_MAT_s011
                                 + 0.939449 w_k1_MSFT_s011
                                 + 0.925551 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.03081 w_k1_ADM_s011
                                 + 1.29135 w_k1_AXP_s011
                                 + 1.27553 w_k1_DIS_s011
                                 + 1.10696 w_k1_FDX_s011
                                 + 1.08159 w_k1_IBM_s011
                                 + 0.82243 w_k1_INTC_s011 + 1.21 w_k1_KO_s011
                                 + 1.54953 w_k1_MAT_s011
                                 + 1.10822 w_k1_MSFT_s011
                                 + 0.980267 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.74727 w_k1_ADM_s011
                                 + 1.46447 w_k1_AXP_s011
                                 + 1.76882 w_k1_DIS_s011
                                 + 1.59629 w_k1_FDX_s011
                                 + 0.839506 w_k1_IBM_s011
                                 + 1.09272 w_k1_INTC_s011
                                 + 1.34744 w_k1_KO_s011 + 1.10723 w_k1_MAT_s011
                                 + 1.48599 w_k1_MSFT_s011
                                 + 1.26222 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.48582 w_k1_ADM_s011
                                 + 1.15434 w_k1_AXP_s011
                                 + 1.22026 w_k1_DIS_s011
                                 + 1.24382 w_k1_FDX_s011
                                 + 0.873634 w_k1_IBM_s011
                                 + 1.10206 w_k1_INTC_s011 + 1.0664 w_k1_KO_s011
                                 + 0.860628 w_k1_MAT_s011
                                 + 1.17241 w_k1_MSFT_s011
                                 + 1.04778 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.887295 w_k1_ADM_s011
                                 + 0.76742 w_k1_AXP_s011
                                 + 0.728711 w_k1_DIS_s011
                                 + 0.872622 w_k1_FDX_s011
                                 + 0.690761 w_k1_IBM_s011
                                 + 0.664906 w_k1_INTC_s011
                                 + 0.706144 w_k1_KO_s011
                                 + 0.828192 w_k1_MAT_s011
                                 + 0.751864 w_k1_MSFT_s011
                                 + 0.729656 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.804547 w_k1_ADM_s011
                                 + 0.796574 w_k1_AXP_s011
                                 + 1.02696 w_k1_DIS_s011
                                 + 0.759305 w_k1_FDX_s011
                                 + 1.5781 w_k1_IBM_s011
                                 + 0.778955 w_k1_INTC_s011
                                 + 1.19831 w_k1_KO_s011 + 1.02623 w_k1_MAT_s011
                                 + 0.798628 w_k1_MSFT_s011
                                 + 0.833309 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.16391 w_k1_ADM_s011
                                 + 1.22819 w_k1_AXP_s011 + 1.3406 w_k1_DIS_s011
                                 + 1.01616 w_k1_FDX_s011
                                 + 1.52336 w_k1_IBM_s011
                                 + 0.754756 w_k1_INTC_s011
                                 + 1.42573 w_k1_KO_s011 + 1.53 w_k1_MAT_s011
                                 + 0.981586 w_k1_MSFT_s011
                                 + 1.09184 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.901785 w_k1_ADM_s011
                                 + 0.659866 w_k1_AXP_s011
                                 + 0.854247 w_k1_DIS_s011
                                 + 0.821491 w_k1_FDX_s011
                                 + 1.10173 w_k1_IBM_s011
                                 + 0.82007 w_k1_INTC_s011
                                 + 1.14738 w_k1_KO_s011
                                 + 0.907232 w_k1_MAT_s011
                                 + 0.965634 w_k1_MSFT_s011
                                 + 0.925319 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.21715 w_k1_ADM_s012
                                 + 1.05907 w_k1_AXP_s012
                                 + 1.16661 w_k1_DIS_s012
                                 + 1.18838 w_k1_FDX_s012 + 0.7007 w_k1_IBM_s012
                                 + 1.16917 w_k1_INTC_s012
                                 + 0.945157 w_k1_KO_s012
                                 + 0.567938 w_k1_MAT_s012
                                 + 1.11325 w_k1_MSFT_s012
                                 + 1.11316 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.01522 w_k1_ADM_s012
                                 + 1.51676 w_k1_AXP_s012
                                 + 1.36964 w_k1_DIS_s012
                                 + 1.16624 w_k1_FDX_s012
                                 + 1.75591 w_k1_IBM_s012
                                 + 1.13759 w_k1_INTC_s012
                                 + 1.26639 w_k1_KO_s012 + 2.24624 w_k1_MAT_s012
                                 + 1.21287 w_k1_MSFT_s012
                                 + 1.18368 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.717357 w_k1_ADM_s012
                                 + 0.724392 w_k1_AXP_s012
                                 + 0.466565 w_k1_DIS_s012
                                 + 0.715051 w_k1_FDX_s012
                                 + 0.863505 w_k1_IBM_s012
                                 + 0.818533 w_k1_INTC_s012
                                 + 0.560973 w_k1_KO_s012
                                 + 0.681526 w_k1_MAT_s012
                                 + 0.773568 w_k1_MSFT_s012
                                 + 0.798539 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.782776 w_k1_ADM_s012
                                 + 0.664058 w_k1_AXP_s012
                                 + 0.72126 w_k1_DIS_s012
                                 + 0.863185 w_k1_FDX_s012
                                 + 0.70829 w_k1_IBM_s012
                                 + 1.15245 w_k1_INTC_s012
                                 + 0.808789 w_k1_KO_s012
                                 + 0.524524 w_k1_MAT_s012
                                 + 1.01463 w_k1_MSFT_s012
                                 + 0.850267 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.2956 w_k1_ADM_s012
                                 + 0.981525 w_k1_AXP_s012
                                 + 1.07419 w_k1_DIS_s012
                                 + 1.06656 w_k1_FDX_s012
                                 + 0.993133 w_k1_IBM_s012
                                 + 1.2074 w_k1_INTC_s012
                                 + 0.899573 w_k1_KO_s012
                                 + 0.775799 w_k1_MAT_s012
                                 + 1.06812 w_k1_MSFT_s012
                                 + 1.10023 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.17285 w_k1_ADM_s012
                                 + 1.4756 w_k1_AXP_s012 + 1.35473 w_k1_DIS_s012
                                 + 1.43779 w_k1_FDX_s012
                                 + 1.05352 w_k1_IBM_s012
                                 + 1.70568 w_k1_INTC_s012 + 1.1085 w_k1_KO_s012
                                 + 1.2179 w_k1_MAT_s012
                                 + 1.49156 w_k1_MSFT_s012 + 1.2624 w_k1_PG_s012
                                 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.793718 w_k1_ADM_s012
                                 + 0.872137 w_k1_AXP_s012
                                 + 0.664862 w_k1_DIS_s012
                                 + 0.755098 w_k1_FDX_s012
                                 + 1.08979 w_k1_IBM_s012
                                 + 1.01813 w_k1_INTC_s012
                                 + 0.791018 w_k1_KO_s012
                                 + 0.772769 w_k1_MAT_s012
                                 + 0.773197 w_k1_MSFT_s012
                                 + 0.948459 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.25849 w_k1_ADM_s012
                                 + 1.3478 w_k1_AXP_s012 + 1.13029 w_k1_DIS_s012
                                 + 1.00141 w_k1_FDX_s012
                                 + 1.39001 w_k1_IBM_s012
                                 + 1.27938 w_k1_INTC_s012
                                 + 1.02423 w_k1_KO_s012 + 1.53137 w_k1_MAT_s012
                                 + 1.11607 w_k1_MSFT_s012
                                 + 1.15503 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.01934 w_k1_ADM_s013
                                 + 1.07005 w_k1_AXP_s013
                                 + 0.936082 w_k1_DIS_s013
                                 + 1.02433 w_k1_FDX_s013
                                 + 0.98019 w_k1_IBM_s013
                                 + 1.03435 w_k1_INTC_s013
                                 + 0.914887 w_k1_KO_s013
                                 + 1.13245 w_k1_MAT_s013
                                 + 1.11444 w_k1_MSFT_s013
                                 + 0.936323 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.43864 w_k1_ADM_s013
                                 + 1.38381 w_k1_AXP_s013
                                 + 1.96881 w_k1_DIS_s013
                                 + 1.31752 w_k1_FDX_s013 + 1.2548 w_k1_IBM_s013
                                 + 1.45683 w_k1_INTC_s013
                                 + 1.60004 w_k1_KO_s013 + 1.47012 w_k1_MAT_s013
                                 + 1.51911 w_k1_MSFT_s013
                                 + 1.27791 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.35844 w_k1_ADM_s013
                                 + 0.947939 w_k1_AXP_s013
                                 + 1.21485 w_k1_DIS_s013
                                 + 1.13927 w_k1_FDX_s013
                                 + 0.753261 w_k1_IBM_s013
                                 + 1.10432 w_k1_INTC_s013
                                 + 1.00843 w_k1_KO_s013
                                 + 0.704417 w_k1_MAT_s013
                                 + 1.11504 w_k1_MSFT_s013
                                 + 0.973675 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.04801 w_k1_ADM_s013
                                 + 0.895306 w_k1_AXP_s013
                                 + 0.778187 w_k1_DIS_s013
                                 + 1.01368 w_k1_FDX_s013
                                 + 0.597017 w_k1_IBM_s013
                                 + 1.2376 w_k1_INTC_s013
                                 + 0.826021 w_k1_KO_s013
                                 + 0.67106 w_k1_MAT_s013
                                 + 1.11535 w_k1_MSFT_s013
                                 + 0.898198 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.676723 w_k1_ADM_s013
                                 + 0.777481 w_k1_AXP_s013
                                 + 0.82495 w_k1_DIS_s013
                                 + 0.887253 w_k1_FDX_s013
                                 + 0.798409 w_k1_IBM_s013
                                 + 1.0607 w_k1_INTC_s013
                                 + 0.933297 w_k1_KO_s013
                                 + 0.730207 w_k1_MAT_s013
                                 + 1.00037 w_k1_MSFT_s013
                                 + 0.870223 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.17387 w_k1_ADM_s013
                                 + 1.15368 w_k1_AXP_s013
                                 + 0.958507 w_k1_DIS_s013
                                 + 0.930988 w_k1_FDX_s013
                                 + 1.3161 w_k1_IBM_s013
                                 + 1.00983 w_k1_INTC_s013
                                 + 1.16293 w_k1_KO_s013 + 1.54186 w_k1_MAT_s013
                                 + 1.03856 w_k1_MSFT_s013
                                 + 0.903775 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.769311 w_k1_ADM_s013
                                 + 0.765648 w_k1_AXP_s013
                                 + 0.603756 w_k1_DIS_s013
                                 + 0.703043 w_k1_FDX_s013
                                 + 0.805893 w_k1_IBM_s013
                                 + 0.854504 w_k1_INTC_s013
                                 + 0.84933 w_k1_KO_s013
                                 + 0.891451 w_k1_MAT_s013
                                 + 0.858639 w_k1_MSFT_s013
                                 + 0.770219 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.26267 w_k1_ADM_s013
                                 + 1.2768 w_k1_AXP_s013 + 1.26318 w_k1_DIS_s013
                                 + 1.25442 w_k1_FDX_s013 + 1.0656 w_k1_IBM_s013
                                 + 1.47356 w_k1_INTC_s013 + 1.1212 w_k1_KO_s013
                                 + 1.07138 w_k1_MAT_s013
                                 + 1.40786 w_k1_MSFT_s013
                                 + 0.972203 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.903468 w_k1_ADM_s014
                                 + 0.728297 w_k1_AXP_s014
                                 + 0.950725 w_k1_DIS_s014
                                 + 0.865613 w_k1_FDX_s014
                                 + 0.817216 w_k1_IBM_s014
                                 + 0.879719 w_k1_INTC_s014
                                 + 0.818123 w_k1_KO_s014
                                 + 0.639807 w_k1_MAT_s014
                                 + 0.855358 w_k1_MSFT_s014
                                 + 0.93408 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.20461 w_k1_ADM_s014
                                 + 0.882792 w_k1_AXP_s014
                                 + 0.771881 w_k1_DIS_s014
                                 + 1.01662 w_k1_FDX_s014
                                 + 0.674387 w_k1_IBM_s014
                                 + 0.893492 w_k1_INTC_s014
                                 + 0.799389 w_k1_KO_s014
                                 + 0.435332 w_k1_MAT_s014
                                 + 0.826532 w_k1_MSFT_s014
                                 + 0.84154 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.03331 w_k1_ADM_s014
                                 + 1.15157 w_k1_AXP_s014
                                 + 1.15514 w_k1_DIS_s014
                                 + 1.01214 w_k1_FDX_s014
                                 + 1.06617 w_k1_IBM_s014
                                 + 1.17203 w_k1_INTC_s014
                                 + 1.13179 w_k1_KO_s014
                                 + 0.875628 w_k1_MAT_s014
                                 + 1.07773 w_k1_MSFT_s014
                                 + 1.12058 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.849542 w_k1_ADM_s014
                                 + 0.76722 w_k1_AXP_s014
                                 + 0.613049 w_k1_DIS_s014
                                 + 0.747687 w_k1_FDX_s014
                                 + 0.888918 w_k1_IBM_s014
                                 + 0.848241 w_k1_INTC_s014
                                 + 0.898253 w_k1_KO_s014
                                 + 0.803705 w_k1_MAT_s014
                                 + 0.790228 w_k1_MSFT_s014
                                 + 0.854461 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.36452 w_k1_ADM_s014
                                 + 0.862742 w_k1_AXP_s014
                                 + 0.72775 w_k1_DIS_s014
                                 + 0.851381 w_k1_FDX_s014
                                 + 1.00201 w_k1_IBM_s014
                                 + 0.891347 w_k1_INTC_s014
                                 + 1.02699 w_k1_KO_s014
                                 + 0.932149 w_k1_MAT_s014
                                 + 0.765501 w_k1_MSFT_s014
                                 + 0.921727 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.18079 w_k1_ADM_s014
                                 + 1.2351 w_k1_AXP_s014 + 1.17931 w_k1_DIS_s014
                                 + 0.94694 w_k1_FDX_s014
                                 + 1.65311 w_k1_IBM_s014
                                 + 0.956768 w_k1_INTC_s014
                                 + 1.27417 w_k1_KO_s014 + 1.58716 w_k1_MAT_s014
                                 + 0.945623 w_k1_MSFT_s014
                                 + 1.10794 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.14838 w_k1_ADM_s014
                                 + 0.987869 w_k1_AXP_s014 + 1.151 w_k1_DIS_s014
                                 + 1.17962 w_k1_FDX_s014
                                 + 0.619189 w_k1_IBM_s014
                                 + 0.992966 w_k1_INTC_s014
                                 + 0.862236 w_k1_KO_s014
                                 + 0.542921 w_k1_MAT_s014
                                 + 0.979113 w_k1_MSFT_s014
                                 + 0.881185 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.08299 w_k1_ADM_s014
                                 + 1.13014 w_k1_AXP_s014 + 1.6614 w_k1_DIS_s014
                                 + 1.27558 w_k1_FDX_s014
                                 + 1.09315 w_k1_IBM_s014
                                 + 1.26932 w_k1_INTC_s014
                                 + 1.43691 w_k1_KO_s014 + 1.21311 w_k1_MAT_s014
                                 + 1.33355 w_k1_MSFT_s014
                                 + 1.26884 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.821702 w_k1_ADM_s015
                                 + 0.908929 w_k1_AXP_s015
                                 + 0.795484 w_k1_DIS_s015
                                 + 0.834972 w_k1_FDX_s015
                                 + 1.23528 w_k1_IBM_s015
                                 + 0.867482 w_k1_INTC_s015
                                 + 0.948153 w_k1_KO_s015
                                 + 1.32961 w_k1_MAT_s015
                                 + 0.842465 w_k1_MSFT_s015
                                 + 1.02288 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.886342 w_k1_ADM_s015
                                 + 1.0661 w_k1_AXP_s015
                                 + 0.993654 w_k1_DIS_s015
                                 + 1.05626 w_k1_FDX_s015
                                 + 1.60121 w_k1_IBM_s015
                                 + 0.89597 w_k1_INTC_s015
                                 + 1.13678 w_k1_KO_s015 + 1.68332 w_k1_MAT_s015
                                 + 0.990604 w_k1_MSFT_s015
                                 + 1.07494 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.694385 w_k1_ADM_s015
                                 + 1.40078 w_k1_AXP_s015
                                 + 1.28109 w_k1_DIS_s015
                                 + 1.30992 w_k1_FDX_s015
                                 + 1.28055 w_k1_IBM_s015
                                 + 1.15272 w_k1_INTC_s015
                                 + 1.09521 w_k1_KO_s015 + 1.81432 w_k1_MAT_s015
                                 + 1.19329 w_k1_MSFT_s015
                                 + 1.12794 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.824989 w_k1_ADM_s015
                                 + 0.761477 w_k1_AXP_s015
                                 + 0.812277 w_k1_DIS_s015
                                 + 0.993386 w_k1_FDX_s015
                                 + 0.698494 w_k1_IBM_s015
                                 + 0.877973 w_k1_INTC_s015
                                 + 0.691055 w_k1_KO_s015
                                 + 0.541986 w_k1_MAT_s015
                                 + 0.884686 w_k1_MSFT_s015
                                 + 0.931675 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.898843 w_k1_ADM_s015
                                 + 0.824048 w_k1_AXP_s015
                                 + 1.16189 w_k1_DIS_s015
                                 + 1.09827 w_k1_FDX_s015
                                 + 1.03138 w_k1_IBM_s015
                                 + 1.14107 w_k1_INTC_s015
                                 + 1.02676 w_k1_KO_s015
                                 + 0.716828 w_k1_MAT_s015
                                 + 1.10155 w_k1_MSFT_s015
                                 + 0.995474 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.981829 w_k1_ADM_s015
                                 + 1.14041 w_k1_AXP_s015
                                 + 0.936968 w_k1_DIS_s015
                                 + 1.01775 w_k1_FDX_s015 + 0.92 w_k1_IBM_s015
                                 + 0.976105 w_k1_INTC_s015
                                 + 0.967239 w_k1_KO_s015
                                 + 0.882307 w_k1_MAT_s015
                                 + 0.991378 w_k1_MSFT_s015
                                 + 1.04095 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.59429 w_k1_ADM_s015
                                 + 0.560368 w_k1_AXP_s015
                                 + 0.568183 w_k1_DIS_s015
                                 + 0.615102 w_k1_FDX_s015
                                 + 1.07277 w_k1_IBM_s015
                                 + 0.670716 w_k1_INTC_s015
                                 + 0.740947 w_k1_KO_s015
                                 + 0.830317 w_k1_MAT_s015
                                 + 0.643481 w_k1_MSFT_s015
                                 + 0.753181 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.13291 w_k1_ADM_s015
                                 + 1.43936 w_k1_AXP_s015
                                 + 1.50748 w_k1_DIS_s015
                                 + 1.59959 w_k1_FDX_s015
                                 + 1.11548 w_k1_IBM_s015
                                 + 1.29697 w_k1_INTC_s015
                                 + 1.25231 w_k1_KO_s015 + 1.17805 w_k1_MAT_s015
                                 + 1.27506 w_k1_MSFT_s015
                                 + 1.16949 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.776209 w_k1_ADM_s016
                                 + 1.23854 w_k1_AXP_s016
                                 + 0.917419 w_k1_DIS_s016
                                 + 0.881481 w_k1_FDX_s016
                                 + 1.30653 w_k1_IBM_s016
                                 + 0.914473 w_k1_INTC_s016
                                 + 0.992265 w_k1_KO_s016
                                 + 1.52035 w_k1_MAT_s016
                                 + 0.954702 w_k1_MSFT_s016
                                 + 1.0547 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.830933 w_k1_ADM_s016
                                 + 0.80389 w_k1_AXP_s016
                                 + 0.885876 w_k1_DIS_s016
                                 + 0.896562 w_k1_FDX_s016
                                 + 0.755932 w_k1_IBM_s016
                                 + 0.855682 w_k1_INTC_s016
                                 + 0.904297 w_k1_KO_s016
                                 + 0.769874 w_k1_MAT_s016
                                 + 0.937922 w_k1_MSFT_s016
                                 + 1.04974 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.873993 w_k1_ADM_s016
                                 + 1.02481 w_k1_AXP_s016
                                 + 0.797621 w_k1_DIS_s016
                                 + 0.844336 w_k1_FDX_s016
                                 + 1.00874 w_k1_IBM_s016
                                 + 0.853272 w_k1_INTC_s016
                                 + 0.862159 w_k1_KO_s016
                                 + 1.08238 w_k1_MAT_s016
                                 + 0.894753 w_k1_MSFT_s016
                                 + 1.04769 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.33331 w_k1_ADM_s016
                                 + 1.35354 w_k1_AXP_s016
                                 + 0.982402 w_k1_DIS_s016
                                 + 1.16421 w_k1_FDX_s016
                                 + 0.87489 w_k1_IBM_s016
                                 + 1.05015 w_k1_INTC_s016
                                 + 0.860848 w_k1_KO_s016
                                 + 0.815458 w_k1_MAT_s016
                                 + 1.05854 w_k1_MSFT_s016
                                 + 0.969116 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.921072 w_k1_ADM_s016
                                 + 0.786097 w_k1_AXP_s016
                                 + 0.633004 w_k1_DIS_s016
                                 + 0.75026 w_k1_FDX_s016
                                 + 0.847099 w_k1_IBM_s016
                                 + 0.793875 w_k1_INTC_s016
                                 + 0.641139 w_k1_KO_s016
                                 + 0.848355 w_k1_MAT_s016
                                 + 0.704753 w_k1_MSFT_s016
                                 + 0.807714 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.22101 w_k1_ADM_s016
                                 + 1.47897 w_k1_AXP_s016
                                 + 1.91615 w_k1_DIS_s016
                                 + 1.36865 w_k1_FDX_s016
                                 + 1.05499 w_k1_IBM_s016
                                 + 1.14543 w_k1_INTC_s016
                                 + 1.46332 w_k1_KO_s016 + 1.02888 w_k1_MAT_s016
                                 + 1.32895 w_k1_MSFT_s016 + 1.1232 w_k1_PG_s016
                                 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.843944 w_k1_ADM_s016
                                 + 1.17945 w_k1_AXP_s016
                                 + 1.45458 w_k1_DIS_s016
                                 + 1.02427 w_k1_FDX_s016
                                 + 1.49587 w_k1_IBM_s016
                                 + 1.12086 w_k1_INTC_s016
                                 + 1.24355 w_k1_KO_s016 + 1.72468 w_k1_MAT_s016
                                 + 1.11627 w_k1_MSFT_s016
                                 + 1.28002 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.89115 w_k1_ADM_s016
                                 + 1.13892 w_k1_AXP_s016
                                 + 1.39866 w_k1_DIS_s016
                                 + 1.14575 w_k1_FDX_s016
                                 + 0.952061 w_k1_IBM_s016
                                 + 1.12791 w_k1_INTC_s016
                                 + 1.15728 w_k1_KO_s016 + 1.30066 w_k1_MAT_s016
                                 + 1.07548 w_k1_MSFT_s016
                                 + 1.10238 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.33854 w_k1_ADM_s017
                                 + 1.12669 w_k1_AXP_s017 + 1.1639 w_k1_DIS_s017
                                 + 1.11119 w_k1_FDX_s017
                                 + 1.00497 w_k1_IBM_s017
                                 + 0.907846 w_k1_INTC_s017
                                 + 1.1641 w_k1_KO_s017 + 0.795663 w_k1_MAT_s017
                                 + 1.04917 w_k1_MSFT_s017
                                 + 1.06487 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.981434 w_k1_ADM_s017
                                 + 0.855584 w_k1_AXP_s017
                                 + 0.935451 w_k1_DIS_s017
                                 + 1.08749 w_k1_FDX_s017
                                 + 0.624486 w_k1_IBM_s017
                                 + 0.933328 w_k1_INTC_s017
                                 + 0.850223 w_k1_KO_s017
                                 + 0.788123 w_k1_MAT_s017
                                 + 0.990032 w_k1_MSFT_s017
                                 + 1.00095 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.953199 w_k1_ADM_s017
                                 + 0.913418 w_k1_AXP_s017
                                 + 0.666986 w_k1_DIS_s017
                                 + 0.780861 w_k1_FDX_s017
                                 + 0.948498 w_k1_IBM_s017
                                 + 0.610374 w_k1_INTC_s017
                                 + 0.788222 w_k1_KO_s017
                                 + 1.17981 w_k1_MAT_s017
                                 + 0.736024 w_k1_MSFT_s017
                                 + 0.796332 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.15955 w_k1_ADM_s017
                                 + 0.710895 w_k1_AXP_s017
                                 + 0.772164 w_k1_DIS_s017
                                 + 0.798943 w_k1_FDX_s017
                                 + 0.894325 w_k1_IBM_s017
                                 + 0.718918 w_k1_INTC_s017
                                 + 0.85409 w_k1_KO_s017
                                 + 0.834738 w_k1_MAT_s017
                                 + 0.807246 w_k1_MSFT_s017
                                 + 0.777266 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.0423 w_k1_ADM_s017
                                 + 0.896155 w_k1_AXP_s017
                                 + 0.800622 w_k1_DIS_s017
                                 + 0.885132 w_k1_FDX_s017
                                 + 1.11325 w_k1_IBM_s017
                                 + 0.919309 w_k1_INTC_s017
                                 + 0.965261 w_k1_KO_s017
                                 + 0.781931 w_k1_MAT_s017
                                 + 0.869923 w_k1_MSFT_s017
                                 + 0.902599 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.855406 w_k1_ADM_s017
                                 + 0.894585 w_k1_AXP_s017
                                 + 1.18223 w_k1_DIS_s017
                                 + 1.00351 w_k1_FDX_s017
                                 + 1.10453 w_k1_IBM_s017
                                 + 0.897471 w_k1_INTC_s017
                                 + 1.22749 w_k1_KO_s017
                                 + 0.928403 w_k1_MAT_s017
                                 + 1.05254 w_k1_MSFT_s017
                                 + 1.01882 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.11509 w_k1_ADM_s017
                                 + 0.928284 w_k1_AXP_s017
                                 + 0.999071 w_k1_DIS_s017
                                 + 0.928787 w_k1_FDX_s017
                                 + 1.35135 w_k1_IBM_s017
                                 + 0.925657 w_k1_INTC_s017
                                 + 1.14065 w_k1_KO_s017 + 1.34314 w_k1_MAT_s017
                                 + 0.957208 w_k1_MSFT_s017
                                 + 0.965422 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.47504 w_k1_ADM_s017
                                 + 1.59952 w_k1_AXP_s017
                                 + 1.38558 w_k1_DIS_s017 + 1.278 w_k1_FDX_s017
                                 + 1.1975 w_k1_IBM_s017
                                 + 1.07253 w_k1_INTC_s017
                                 + 1.10363 w_k1_KO_s017 + 1.54214 w_k1_MAT_s017
                                 + 1.13459 w_k1_MSFT_s017
                                 + 1.22271 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.872593 w_k1_ADM_s100
                                 + 0.548491 w_k1_AXP_s100
                                 + 0.64735 w_k1_DIS_s100
                                 + 0.758545 w_k1_FDX_s100
                                 + 0.664596 w_k1_IBM_s100
                                 + 0.795143 w_k1_INTC_s100
                                 + 0.675042 w_k1_KO_s100
                                 + 0.442489 w_k1_MAT_s100
                                 + 0.797706 w_k1_MSFT_s100
                                 + 0.884702 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.0745 w_k1_ADM_s100
                                 + 1.00812 w_k1_AXP_s100
                                 + 0.95646 w_k1_DIS_s100
                                 + 0.976009 w_k1_FDX_s100
                                 + 1.17307 w_k1_IBM_s100
                                 + 0.889689 w_k1_INTC_s100
                                 + 0.913828 w_k1_KO_s100
                                 + 1.27263 w_k1_MAT_s100
                                 + 0.995209 w_k1_MSFT_s100
                                 + 0.947389 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.31618 w_k1_ADM_s100
                                 + 0.981721 w_k1_AXP_s100
                                 + 0.90321 w_k1_DIS_s100
                                 + 1.02509 w_k1_FDX_s100
                                 + 0.73219 w_k1_IBM_s100
                                 + 0.786883 w_k1_INTC_s100
                                 + 0.839643 w_k1_KO_s100
                                 + 0.893902 w_k1_MAT_s100
                                 + 1.00522 w_k1_MSFT_s100
                                 + 0.88249 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.809141 w_k1_ADM_s100
                                 + 0.844615 w_k1_AXP_s100
                                 + 0.912317 w_k1_DIS_s100
                                 + 0.777746 w_k1_FDX_s100
                                 + 1.1746 w_k1_IBM_s100
                                 + 0.886397 w_k1_INTC_s100
                                 + 0.986391 w_k1_KO_s100
                                 + 0.976754 w_k1_MAT_s100
                                 + 0.82472 w_k1_MSFT_s100
                                 + 1.01393 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.28281 w_k1_ADM_s100
                                 + 1.47493 w_k1_AXP_s100
                                 + 1.97911 w_k1_DIS_s100
                                 + 1.38994 w_k1_FDX_s100
                                 + 1.35531 w_k1_IBM_s100
                                 + 1.29538 w_k1_INTC_s100
                                 + 1.28017 w_k1_KO_s100 + 1.53142 w_k1_MAT_s100
                                 + 1.36524 w_k1_MSFT_s100
                                 + 1.09516 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.19902 w_k1_ADM_s100
                                 + 1.19793 w_k1_AXP_s100
                                 + 1.14817 w_k1_DIS_s100
                                 + 0.944438 w_k1_FDX_s100
                                 + 1.70778 w_k1_IBM_s100
                                 + 0.995618 w_k1_INTC_s100
                                 + 1.29684 w_k1_KO_s100 + 1.7663 w_k1_MAT_s100
                                 + 1.05371 w_k1_MSFT_s100
                                 + 1.11584 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.37979 w_k1_ADM_s100
                                 + 1.29845 w_k1_AXP_s100
                                 + 1.33778 w_k1_DIS_s100
                                 + 1.27108 w_k1_FDX_s100
                                 + 0.992278 w_k1_IBM_s100
                                 + 1.11196 w_k1_INTC_s100
                                 + 1.16041 w_k1_KO_s100 + 1.21565 w_k1_MAT_s100
                                 + 1.10095 w_k1_MSFT_s100
                                 + 1.18334 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.776624 w_k1_ADM_s100
                                 + 0.819151 w_k1_AXP_s100
                                 + 0.617851 w_k1_DIS_s100
                                 + 0.709664 w_k1_FDX_s100
                                 + 1.05008 w_k1_IBM_s100
                                 + 0.65649 w_k1_INTC_s100
                                 + 0.822644 w_k1_KO_s100
                                 + 1.29902 w_k1_MAT_s100
                                 + 0.73546 w_k1_MSFT_s100
                                 + 0.861088 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.944953 w_k1_ADM_s101
                                 + 1.28538 w_k1_AXP_s101
                                 + 0.993524 w_k1_DIS_s101
                                 + 1.13117 w_k1_FDX_s101
                                 + 0.919175 w_k1_IBM_s101
                                 + 1.20561 w_k1_INTC_s101
                                 + 0.891831 w_k1_KO_s101
                                 + 1.18312 w_k1_MAT_s101
                                 + 0.992434 w_k1_MSFT_s101
                                 + 0.999609 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.874 w_k1_ADM_s101
                                 + 1.02844 w_k1_AXP_s101
                                 + 0.670236 w_k1_DIS_s101
                                 + 0.858748 w_k1_FDX_s101
                                 + 0.771911 w_k1_IBM_s101
                                 + 0.999241 w_k1_INTC_s101
                                 + 0.71959 w_k1_KO_s101
                                 + 0.930837 w_k1_MAT_s101
                                 + 0.861595 w_k1_MSFT_s101
                                 + 0.875646 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.31036 w_k1_ADM_s101
                                 + 1.40413 w_k1_AXP_s101
                                 + 1.63913 w_k1_DIS_s101
                                 + 1.35325 w_k1_FDX_s101
                                 + 1.19533 w_k1_IBM_s101
                                 + 1.10966 w_k1_INTC_s101
                                 + 1.29252 w_k1_KO_s101 + 1.53925 w_k1_MAT_s101
                                 + 1.23717 w_k1_MSFT_s101
                                 + 1.23173 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.62746 w_k1_ADM_s101
                                 + 0.73505 w_k1_AXP_s101
                                 + 0.65846 w_k1_DIS_s101
                                 + 0.745146 w_k1_FDX_s101
                                 + 1.13783 w_k1_IBM_s101
                                 + 0.951549 w_k1_INTC_s101
                                 + 0.739078 w_k1_KO_s101
                                 + 1.19633 w_k1_MAT_s101
                                 + 0.806374 w_k1_MSFT_s101
                                 + 0.87406 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.13264 w_k1_ADM_s101
                                 + 1.13771 w_k1_AXP_s101
                                 + 1.27154 w_k1_DIS_s101
                                 + 1.06136 w_k1_FDX_s101
                                 + 1.46798 w_k1_IBM_s101
                                 + 1.10415 w_k1_INTC_s101
                                 + 1.11233 w_k1_KO_s101 + 1.73403 w_k1_MAT_s101
                                 + 1.13614 w_k1_MSFT_s101
                                 + 1.18859 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.908785 w_k1_ADM_s101
                                 + 0.882963 w_k1_AXP_s101
                                 + 1.04527 w_k1_DIS_s101
                                 + 0.938359 w_k1_FDX_s101
                                 + 1.03017 w_k1_IBM_s101
                                 + 1.18671 w_k1_INTC_s101
                                 + 1.07028 w_k1_KO_s101 + 1.24064 w_k1_MAT_s101
                                 + 1.00762 w_k1_MSFT_s101
                                 + 1.09146 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.930245 w_k1_ADM_s101
                                 + 1.05289 w_k1_AXP_s101
                                 + 0.788944 w_k1_DIS_s101
                                 + 0.827594 w_k1_FDX_s101
                                 + 1.12405 w_k1_IBM_s101
                                 + 0.849604 w_k1_INTC_s101
                                 + 0.865734 w_k1_KO_s101
                                 + 1.22541 w_k1_MAT_s101
                                 + 0.824863 w_k1_MSFT_s101
                                 + 1.03634 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.03494 w_k1_ADM_s101
                                 + 0.850292 w_k1_AXP_s101
                                 + 0.999301 w_k1_DIS_s101
                                 + 1.01053 w_k1_FDX_s101
                                 + 0.634729 w_k1_IBM_s101
                                 + 0.934562 w_k1_INTC_s101
                                 + 0.77595 w_k1_KO_s101
                                 + 0.542959 w_k1_MAT_s101
                                 + 0.960589 w_k1_MSFT_s101
                                 + 0.989792 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.79903 w_k1_ADM_s102
                                 + 1.06935 w_k1_AXP_s102
                                 + 0.87814 w_k1_DIS_s102
                                 + 0.877791 w_k1_FDX_s102
                                 + 1.02439 w_k1_IBM_s102
                                 + 0.707809 w_k1_INTC_s102
                                 + 1.05456 w_k1_KO_s102 + 1.48442 w_k1_MAT_s102
                                 + 0.965914 w_k1_MSFT_s102
                                 + 1.0619 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.31636 w_k1_ADM_s102
                                 + 1.25955 w_k1_AXP_s102
                                 + 1.25755 w_k1_DIS_s102
                                 + 1.23379 w_k1_FDX_s102
                                 + 1.02038 w_k1_IBM_s102
                                 + 0.97783 w_k1_INTC_s102
                                 + 1.11441 w_k1_KO_s102 + 1.19919 w_k1_MAT_s102
                                 + 1.37523 w_k1_MSFT_s102
                                 + 1.36406 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.13958 w_k1_ADM_s102
                                 + 1.37921 w_k1_AXP_s102
                                 + 1.58654 w_k1_DIS_s102
                                 + 1.18202 w_k1_FDX_s102
                                 + 1.68482 w_k1_IBM_s102
                                 + 1.18782 w_k1_INTC_s102
                                 + 1.50196 w_k1_KO_s102 + 1.54808 w_k1_MAT_s102
                                 + 1.55353 w_k1_MSFT_s102
                                 + 1.19998 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.1665 w_k1_ADM_s102
                                 + 0.875295 w_k1_AXP_s102
                                 + 0.895875 w_k1_DIS_s102
                                 + 1.01236 w_k1_FDX_s102
                                 + 0.896652 w_k1_IBM_s102
                                 + 0.798798 w_k1_INTC_s102
                                 + 0.909206 w_k1_KO_s102
                                 + 0.808953 w_k1_MAT_s102
                                 + 1.06782 w_k1_MSFT_s102
                                 + 0.927689 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.770233 w_k1_ADM_s102
                                 + 0.678644 w_k1_AXP_s102
                                 + 0.627675 w_k1_DIS_s102
                                 + 0.739624 w_k1_FDX_s102
                                 + 0.715831 w_k1_IBM_s102
                                 + 0.827922 w_k1_INTC_s102
                                 + 0.731928 w_k1_KO_s102
                                 + 0.538041 w_k1_MAT_s102
                                 + 0.891422 w_k1_MSFT_s102
                                 + 0.830507 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.990993 w_k1_ADM_s102
                                 + 1.09095 w_k1_AXP_s102
                                 + 0.99747 w_k1_DIS_s102
                                 + 0.962465 w_k1_FDX_s102
                                 + 1.3348 w_k1_IBM_s102
                                 + 0.849859 w_k1_INTC_s102
                                 + 1.19089 w_k1_KO_s102 + 1.4136 w_k1_MAT_s102
                                 + 1.06473 w_k1_MSFT_s102
                                 + 1.15177 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.961232 w_k1_ADM_s102
                                 + 1.09957 w_k1_AXP_s102
                                 + 1.00456 w_k1_DIS_s102
                                 + 1.00673 w_k1_FDX_s102
                                 + 0.724083 w_k1_IBM_s102
                                 + 0.930892 w_k1_INTC_s102
                                 + 0.889021 w_k1_KO_s102
                                 + 0.878808 w_k1_MAT_s102
                                 + 1.14212 w_k1_MSFT_s102
                                 + 1.06834 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.730378 w_k1_ADM_s102
                                 + 0.812517 w_k1_AXP_s102
                                 + 0.750457 w_k1_DIS_s102
                                 + 0.764994 w_k1_FDX_s102
                                 + 0.932929 w_k1_IBM_s102
                                 + 0.692959 w_k1_INTC_s102
                                 + 0.815976 w_k1_KO_s102
                                 + 0.862497 w_k1_MAT_s102
                                 + 0.959236 w_k1_MSFT_s102
                                 + 0.91998 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.01488 w_k1_ADM_s103
                                 + 0.935673 w_k1_AXP_s103
                                 + 1.0236 w_k1_DIS_s103 + 1.12165 w_k1_FDX_s103
                                 + 0.681164 w_k1_IBM_s103
                                 + 1.10566 w_k1_INTC_s103
                                 + 0.893041 w_k1_KO_s103
                                 + 0.650133 w_k1_MAT_s103
                                 + 0.966543 w_k1_MSFT_s103
                                 + 0.965715 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.44365 w_k1_ADM_s103
                                 + 1.5841 w_k1_AXP_s103 + 1.66339 w_k1_DIS_s103
                                 + 1.41335 w_k1_FDX_s103
                                 + 1.25931 w_k1_IBM_s103
                                 + 1.28509 w_k1_INTC_s103
                                 + 1.37096 w_k1_KO_s103 + 1.17199 w_k1_MAT_s103
                                 + 1.26437 w_k1_MSFT_s103
                                 + 1.28524 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.01599 w_k1_ADM_s103
                                 + 1.29383 w_k1_AXP_s103
                                 + 1.05378 w_k1_DIS_s103
                                 + 0.945446 w_k1_FDX_s103
                                 + 1.54489 w_k1_IBM_s103
                                 + 0.930308 w_k1_INTC_s103
                                 + 1.15531 w_k1_KO_s103 + 1.81728 w_k1_MAT_s103
                                 + 0.813335 w_k1_MSFT_s103
                                 + 1.05244 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.03584 w_k1_ADM_s103
                                 + 1.11497 w_k1_AXP_s103
                                 + 1.09422 w_k1_DIS_s103
                                 + 1.06079 w_k1_FDX_s103
                                 + 1.11087 w_k1_IBM_s103
                                 + 0.846261 w_k1_INTC_s103
                                 + 0.970044 w_k1_KO_s103
                                 + 1.06597 w_k1_MAT_s103
                                 + 0.889969 w_k1_MSFT_s103
                                 + 0.965133 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.77478 w_k1_ADM_s103
                                 + 0.789362 w_k1_AXP_s103
                                 + 0.679538 w_k1_DIS_s103
                                 + 0.791652 w_k1_FDX_s103
                                 + 0.845266 w_k1_IBM_s103
                                 + 0.848072 w_k1_INTC_s103
                                 + 0.745227 w_k1_KO_s103
                                 + 0.712631 w_k1_MAT_s103
                                 + 0.677621 w_k1_MSFT_s103
                                 + 0.736623 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.978274 w_k1_ADM_s103
                                 + 1.20443 w_k1_AXP_s103
                                 + 0.924945 w_k1_DIS_s103
                                 + 0.954667 w_k1_FDX_s103
                                 + 0.919315 w_k1_IBM_s103
                                 + 0.944816 w_k1_INTC_s103
                                 + 1.10581 w_k1_KO_s103 + 1.20537 w_k1_MAT_s103
                                 + 0.906433 w_k1_MSFT_s103
                                 + 0.926574 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.91318 w_k1_ADM_s103
                                 + 0.797807 w_k1_AXP_s103
                                 + 0.570569 w_k1_DIS_s103
                                 + 0.887809 w_k1_FDX_s103
                                 + 0.640478 w_k1_IBM_s103
                                 + 0.920729 w_k1_INTC_s103
                                 + 0.683381 w_k1_KO_s103
                                 + 0.716339 w_k1_MAT_s103
                                 + 0.811294 w_k1_MSFT_s103
                                 + 0.765106 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.33019 w_k1_ADM_s103
                                 + 1.4562 w_k1_AXP_s103 + 1.19515 w_k1_DIS_s103
                                 + 1.22454 w_k1_FDX_s103
                                 + 0.854422 w_k1_IBM_s103
                                 + 0.94493 w_k1_INTC_s103
                                 + 1.00394 w_k1_KO_s103
                                 + 0.860042 w_k1_MAT_s103
                                 + 1.03388 w_k1_MSFT_s103
                                 + 0.962564 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.05641 w_k1_ADM_s104
                                 + 1.13741 w_k1_AXP_s104
                                 + 0.770471 w_k1_DIS_s104
                                 + 1.07175 w_k1_FDX_s104
                                 + 0.724062 w_k1_IBM_s104
                                 + 0.766031 w_k1_INTC_s104
                                 + 0.817203 w_k1_KO_s104
                                 + 1.61988 w_k1_MAT_s104
                                 + 0.911372 w_k1_MSFT_s104
                                 + 0.927716 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.14595 w_k1_ADM_s104
                                 + 1.62349 w_k1_AXP_s104
                                 + 1.32231 w_k1_DIS_s104
                                 + 1.25775 w_k1_FDX_s104
                                 + 1.38208 w_k1_IBM_s104
                                 + 1.17281 w_k1_INTC_s104
                                 + 1.09623 w_k1_KO_s104 + 1.79679 w_k1_MAT_s104
                                 + 1.23141 w_k1_MSFT_s104
                                 + 1.15227 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.932117 w_k1_ADM_s104
                                 + 1.24486 w_k1_AXP_s104 + 1.1508 w_k1_DIS_s104
                                 + 1.09591 w_k1_FDX_s104
                                 + 1.25692 w_k1_IBM_s104
                                 + 1.12347 w_k1_INTC_s104
                                 + 1.03802 w_k1_KO_s104 + 1.5632 w_k1_MAT_s104
                                 + 1.14033 w_k1_MSFT_s104
                                 + 1.10698 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.95799 w_k1_ADM_s104
                                 + 1.21056 w_k1_AXP_s104
                                 + 1.56955 w_k1_DIS_s104
                                 + 1.12152 w_k1_FDX_s104
                                 + 1.38631 w_k1_IBM_s104
                                 + 1.04999 w_k1_INTC_s104
                                 + 1.32195 w_k1_KO_s104 + 1.9641 w_k1_MAT_s104
                                 + 1.14244 w_k1_MSFT_s104
                                 + 1.15295 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.865696 w_k1_ADM_s104
                                 + 1.00156 w_k1_AXP_s104
                                 + 0.728729 w_k1_DIS_s104
                                 + 0.801692 w_k1_FDX_s104
                                 + 1.34359 w_k1_IBM_s104
                                 + 0.766043 w_k1_INTC_s104
                                 + 0.92862 w_k1_KO_s104 + 1.74647 w_k1_MAT_s104
                                 + 0.729358 w_k1_MSFT_s104
                                 + 0.83588 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.739303 w_k1_ADM_s104
                                 + 0.734003 w_k1_AXP_s104
                                 + 0.787642 w_k1_DIS_s104
                                 + 0.843757 w_k1_FDX_s104
                                 + 0.840682 w_k1_IBM_s104
                                 + 0.786215 w_k1_INTC_s104
                                 + 0.759583 w_k1_KO_s104
                                 + 1.02613 w_k1_MAT_s104
                                 + 0.845334 w_k1_MSFT_s104
                                 + 0.805555 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.46823 w_k1_ADM_s104
                                 + 1.76003 w_k1_AXP_s104
                                 + 1.91794 w_k1_DIS_s104
                                 + 1.57328 w_k1_FDX_s104
                                 + 1.39894 w_k1_IBM_s104
                                 + 1.09795 w_k1_INTC_s104
                                 + 1.30079 w_k1_KO_s104 + 2.07738 w_k1_MAT_s104
                                 + 1.14004 w_k1_MSFT_s104
                                 + 1.26671 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.01195 w_k1_ADM_s104
                                 + 0.961657 w_k1_AXP_s104
                                 + 1.05109 w_k1_DIS_s104
                                 + 0.957876 w_k1_FDX_s104
                                 + 1.14473 w_k1_IBM_s104
                                 + 1.00323 w_k1_INTC_s104
                                 + 0.971096 w_k1_KO_s104
                                 + 0.97297 w_k1_MAT_s104
                                 + 0.97266 w_k1_MSFT_s104
                                 + 0.934196 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.905136 w_k1_ADM_s105
                                 + 0.890459 w_k1_AXP_s105
                                 + 0.986885 w_k1_DIS_s105
                                 + 0.980126 w_k1_FDX_s105
                                 + 1.33737 w_k1_IBM_s105
                                 + 0.875126 w_k1_INTC_s105
                                 + 1.24145 w_k1_KO_s105 + 1.69268 w_k1_MAT_s105
                                 + 1.06161 w_k1_MSFT_s105
                                 + 1.15147 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.18455 w_k1_ADM_s105
                                 + 1.16454 w_k1_AXP_s105
                                 + 1.24181 w_k1_DIS_s105
                                 + 1.15599 w_k1_FDX_s105
                                 + 1.12533 w_k1_IBM_s105
                                 + 1.22586 w_k1_INTC_s105
                                 + 1.04626 w_k1_KO_s105 + 1.66597 w_k1_MAT_s105
                                 + 1.20152 w_k1_MSFT_s105
                                 + 1.16451 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.783566 w_k1_ADM_s105
                                 + 0.765003 w_k1_AXP_s105
                                 + 0.954224 w_k1_DIS_s105
                                 + 0.898827 w_k1_FDX_s105
                                 + 0.980089 w_k1_IBM_s105
                                 + 0.98656 w_k1_INTC_s105
                                 + 0.984347 w_k1_KO_s105
                                 + 1.04944 w_k1_MAT_s105
                                 + 0.977689 w_k1_MSFT_s105
                                 + 1.01192 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.21231 w_k1_ADM_s105
                                 + 0.973153 w_k1_AXP_s105
                                 + 1.56861 w_k1_DIS_s105 + 1.1096 w_k1_FDX_s105
                                 + 1.40247 w_k1_IBM_s105
                                 + 1.15969 w_k1_INTC_s105
                                 + 1.55378 w_k1_KO_s105 + 1.42645 w_k1_MAT_s105
                                 + 1.25686 w_k1_MSFT_s105
                                 + 1.20526 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.965128 w_k1_ADM_s105
                                 + 0.722997 w_k1_AXP_s105
                                 + 0.749436 w_k1_DIS_s105
                                 + 0.803165 w_k1_FDX_s105
                                 + 0.828072 w_k1_IBM_s105
                                 + 0.713037 w_k1_INTC_s105
                                 + 0.903972 w_k1_KO_s105
                                 + 0.690034 w_k1_MAT_s105
                                 + 0.759621 w_k1_MSFT_s105
                                 + 0.935175 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.34251 w_k1_ADM_s105
                                 + 1.16404 w_k1_AXP_s105
                                 + 0.973304 w_k1_DIS_s105
                                 + 1.0505 w_k1_FDX_s105
                                 + 0.840783 w_k1_IBM_s105
                                 + 0.884553 w_k1_INTC_s105
                                 + 1.01796 w_k1_KO_s105 + 1.41849 w_k1_MAT_s105
                                 + 0.997473 w_k1_MSFT_s105
                                 + 1.09126 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.912041 w_k1_ADM_s105
                                 + 0.860239 w_k1_AXP_s105
                                 + 0.696258 w_k1_DIS_s105
                                 + 0.781693 w_k1_FDX_s105
                                 + 1.11594 w_k1_IBM_s105
                                 + 0.889989 w_k1_INTC_s105
                                 + 0.932793 w_k1_KO_s105
                                 + 1.05294 w_k1_MAT_s105
                                 + 0.760779 w_k1_MSFT_s105
                                 + 0.988277 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.857496 w_k1_ADM_s105
                                 + 0.796467 w_k1_AXP_s105
                                 + 0.691385 w_k1_DIS_s105
                                 + 0.828722 w_k1_FDX_s105
                                 + 0.628379 w_k1_IBM_s105
                                 + 0.783911 w_k1_INTC_s105
                                 + 0.786629 w_k1_KO_s105
                                 + 0.711092 w_k1_MAT_s105
                                 + 0.812449 w_k1_MSFT_s105
                                 + 0.8217 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.980294 w_k1_ADM_s106
                                 + 1.04679 w_k1_AXP_s106
                                 + 0.832969 w_k1_DIS_s106
                                 + 0.919302 w_k1_FDX_s106
                                 + 1.23721 w_k1_IBM_s106
                                 + 0.863323 w_k1_INTC_s106
                                 + 1.11365 w_k1_KO_s106 + 1.17344 w_k1_MAT_s106
                                 + 0.842671 w_k1_MSFT_s106
                                 + 0.939922 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.26824 w_k1_ADM_s106
                                 + 1.43606 w_k1_AXP_s106
                                 + 1.33476 w_k1_DIS_s106
                                 + 1.37371 w_k1_FDX_s106 + 1.2802 w_k1_IBM_s106
                                 + 1.25699 w_k1_INTC_s106
                                 + 1.30946 w_k1_KO_s106 + 1.68755 w_k1_MAT_s106
                                 + 1.23324 w_k1_MSFT_s106
                                 + 1.16895 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.14598 w_k1_ADM_s106
                                 + 1.31304 w_k1_AXP_s106
                                 + 1.28966 w_k1_DIS_s106
                                 + 1.43397 w_k1_FDX_s106
                                 + 0.944664 w_k1_IBM_s106
                                 + 1.28035 w_k1_INTC_s106
                                 + 1.09105 w_k1_KO_s106 + 1.12847 w_k1_MAT_s106
                                 + 1.24446 w_k1_MSFT_s106
                                 + 1.08209 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.21511 w_k1_ADM_s106
                                 + 1.15642 w_k1_AXP_s106
                                 + 0.961081 w_k1_DIS_s106
                                 + 1.11774 w_k1_FDX_s106
                                 + 0.893214 w_k1_IBM_s106
                                 + 1.06034 w_k1_INTC_s106
                                 + 0.960151 w_k1_KO_s106
                                 + 0.72003 w_k1_MAT_s106
                                 + 0.972555 w_k1_MSFT_s106
                                 + 1.06054 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.11598 w_k1_ADM_s106
                                 + 1.30404 w_k1_AXP_s106
                                 + 0.880978 w_k1_DIS_s106 + 1.076 w_k1_FDX_s106
                                 + 1.58407 w_k1_IBM_s106
                                 + 1.16168 w_k1_INTC_s106
                                 + 1.02874 w_k1_KO_s106 + 1.41508 w_k1_MAT_s106
                                 + 0.991248 w_k1_MSFT_s106
                                 + 1.2272 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.912936 w_k1_ADM_s106
                                 + 0.923568 w_k1_AXP_s106
                                 + 0.928204 w_k1_DIS_s106
                                 + 0.833891 w_k1_FDX_s106
                                 + 1.2131 w_k1_IBM_s106
                                 + 1.01804 w_k1_INTC_s106 + 1.0552 w_k1_KO_s106
                                 + 1.63529 w_k1_MAT_s106
                                 + 0.946794 w_k1_MSFT_s106
                                 + 0.927438 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.36804 w_k1_ADM_s106
                                 + 1.06244 w_k1_AXP_s106
                                 + 0.982342 w_k1_DIS_s106
                                 + 1.10813 w_k1_FDX_s106
                                 + 1.07844 w_k1_IBM_s106
                                 + 1.01933 w_k1_INTC_s106
                                 + 1.02114 w_k1_KO_s106 + 1.28605 w_k1_MAT_s106
                                 + 1.05086 w_k1_MSFT_s106
                                 + 1.18235 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.99761 w_k1_ADM_s106
                                 + 0.845104 w_k1_AXP_s106
                                 + 0.617229 w_k1_DIS_s106
                                 + 0.890587 w_k1_FDX_s106
                                 + 0.734283 w_k1_IBM_s106
                                 + 0.946845 w_k1_INTC_s106
                                 + 0.71844 w_k1_KO_s106
                                 + 0.717662 w_k1_MAT_s106
                                 + 0.725645 w_k1_MSFT_s106
                                 + 0.943812 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.23446 w_k1_ADM_s107
                                 + 1.27982 w_k1_AXP_s107
                                 + 1.19828 w_k1_DIS_s107
                                 + 1.06795 w_k1_FDX_s107
                                 + 1.47473 w_k1_IBM_s107
                                 + 1.03255 w_k1_INTC_s107
                                 + 1.19258 w_k1_KO_s107 + 1.31502 w_k1_MAT_s107
                                 + 1.00365 w_k1_MSFT_s107 + 1.0102 w_k1_PG_s107
                                 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.74716 w_k1_ADM_s107
                                 + 0.903227 w_k1_AXP_s107
                                 + 1.01413 w_k1_DIS_s107
                                 + 0.955555 w_k1_FDX_s107
                                 + 1.00466 w_k1_IBM_s107
                                 + 1.02207 w_k1_INTC_s107
                                 + 0.89196 w_k1_KO_s107 + 1.14019 w_k1_MAT_s107
                                 + 1.1203 w_k1_MSFT_s107 + 1.00958 w_k1_PG_s107
                                 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.783585 w_k1_ADM_s107
                                 + 0.694953 w_k1_AXP_s107
                                 + 0.788215 w_k1_DIS_s107
                                 + 0.745607 w_k1_FDX_s107
                                 + 0.943775 w_k1_IBM_s107
                                 + 0.724669 w_k1_INTC_s107
                                 + 0.88978 w_k1_KO_s107
                                 + 0.994921 w_k1_MAT_s107
                                 + 0.763069 w_k1_MSFT_s107
                                 + 0.810792 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.837997 w_k1_ADM_s107
                                 + 0.909703 w_k1_AXP_s107
                                 + 1.06899 w_k1_DIS_s107
                                 + 1.09058 w_k1_FDX_s107
                                 + 0.931565 w_k1_IBM_s107
                                 + 0.907826 w_k1_INTC_s107
                                 + 0.89445 w_k1_KO_s107 + 1.08567 w_k1_MAT_s107
                                 + 1.14208 w_k1_MSFT_s107
                                 + 1.02735 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.00258 w_k1_ADM_s107
                                 + 1.08492 w_k1_AXP_s107
                                 + 1.57676 w_k1_DIS_s107
                                 + 1.22501 w_k1_FDX_s107
                                 + 1.26703 w_k1_IBM_s107
                                 + 1.22274 w_k1_INTC_s107
                                 + 1.25055 w_k1_KO_s107 + 1.19396 w_k1_MAT_s107
                                 + 1.20951 w_k1_MSFT_s107
                                 + 1.03513 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.703833 w_k1_ADM_s107
                                 + 1.05924 w_k1_AXP_s107 + 1.383 w_k1_DIS_s107
                                 + 1.24898 w_k1_FDX_s107
                                 + 0.777079 w_k1_IBM_s107
                                 + 1.25896 w_k1_INTC_s107
                                 + 0.864055 w_k1_KO_s107
                                 + 0.793222 w_k1_MAT_s107
                                 + 1.19651 w_k1_MSFT_s107
                                 + 0.942833 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.948932 w_k1_ADM_s107
                                 + 1.15979 w_k1_AXP_s107
                                 + 1.47342 w_k1_DIS_s107
                                 + 1.35026 w_k1_FDX_s107
                                 + 0.886557 w_k1_IBM_s107
                                 + 1.12198 w_k1_INTC_s107
                                 + 0.999348 w_k1_KO_s107
                                 + 0.908567 w_k1_MAT_s107
                                 + 1.23464 w_k1_MSFT_s107
                                 + 0.989961 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.837352 w_k1_ADM_s107
                                 + 0.836105 w_k1_AXP_s107
                                 + 0.789187 w_k1_DIS_s107
                                 + 0.959307 w_k1_FDX_s107
                                 + 0.656746 w_k1_IBM_s107
                                 + 0.949325 w_k1_INTC_s107
                                 + 0.67466 w_k1_KO_s107
                                 + 0.493238 w_k1_MAT_s107
                                 + 0.909739 w_k1_MSFT_s107
                                 + 0.762076 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.915845 w_k1_ADM_s110
                                 + 0.917968 w_k1_AXP_s110
                                 + 1.2049 w_k1_DIS_s110 + 1.22846 w_k1_FDX_s110
                                 + 1.02437 w_k1_IBM_s110
                                 + 1.25316 w_k1_INTC_s110
                                 + 1.19061 w_k1_KO_s110 + 1.0657 w_k1_MAT_s110
                                 + 1.24745 w_k1_MSFT_s110
                                 + 0.943674 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.15104 w_k1_ADM_s110
                                 + 1.0905 w_k1_AXP_s110 + 1.18087 w_k1_DIS_s110
                                 + 1.22219 w_k1_FDX_s110
                                 + 0.845761 w_k1_IBM_s110
                                 + 1.10674 w_k1_INTC_s110
                                 + 1.08048 w_k1_KO_s110
                                 + 0.780181 w_k1_MAT_s110
                                 + 1.07763 w_k1_MSFT_s110
                                 + 1.06195 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.06122 w_k1_ADM_s110
                                 + 0.838331 w_k1_AXP_s110
                                 + 0.884225 w_k1_DIS_s110
                                 + 0.910224 w_k1_FDX_s110
                                 + 1.06845 w_k1_IBM_s110
                                 + 1.04337 w_k1_INTC_s110
                                 + 0.936729 w_k1_KO_s110
                                 + 0.861671 w_k1_MAT_s110
                                 + 0.973158 w_k1_MSFT_s110
                                 + 0.932144 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.773766 w_k1_ADM_s110
                                 + 0.602275 w_k1_AXP_s110
                                 + 0.683843 w_k1_DIS_s110
                                 + 0.766063 w_k1_FDX_s110
                                 + 0.824637 w_k1_IBM_s110
                                 + 0.785743 w_k1_INTC_s110
                                 + 0.780493 w_k1_KO_s110
                                 + 0.68017 w_k1_MAT_s110
                                 + 0.761866 w_k1_MSFT_s110
                                 + 0.806539 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.7452 w_k1_ADM_s110
                                 + 0.904702 w_k1_AXP_s110
                                 + 0.996299 w_k1_DIS_s110
                                 + 0.778446 w_k1_FDX_s110
                                 + 1.50954 w_k1_IBM_s110
                                 + 0.89088 w_k1_INTC_s110
                                 + 1.26634 w_k1_KO_s110 + 1.2009 w_k1_MAT_s110
                                 + 0.94245 w_k1_MSFT_s110
                                 + 0.913836 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.23143 w_k1_ADM_s110
                                 + 1.38366 w_k1_AXP_s110
                                 + 1.74997 w_k1_DIS_s110
                                 + 1.45087 w_k1_FDX_s110
                                 + 1.27417 w_k1_IBM_s110
                                 + 1.14593 w_k1_INTC_s110
                                 + 1.48234 w_k1_KO_s110 + 1.47484 w_k1_MAT_s110
                                 + 1.37129 w_k1_MSFT_s110 + 1.1525 w_k1_PG_s110
                                 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.970675 w_k1_ADM_s110
                                 + 1.11916 w_k1_AXP_s110
                                 + 1.26309 w_k1_DIS_s110
                                 + 1.08215 w_k1_FDX_s110
                                 + 1.49482 w_k1_IBM_s110
                                 + 1.06225 w_k1_INTC_s110
                                 + 1.27035 w_k1_KO_s110 + 1.06999 w_k1_MAT_s110
                                 + 0.980625 w_k1_MSFT_s110
                                 + 0.963845 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.908426 w_k1_ADM_s110
                                 + 1.10666 w_k1_AXP_s110
                                 + 0.875859 w_k1_DIS_s110
                                 + 1.09225 w_k1_FDX_s110
                                 + 0.810455 w_k1_IBM_s110
                                 + 1.12188 w_k1_INTC_s110
                                 + 0.881177 w_k1_KO_s110
                                 + 0.753422 w_k1_MAT_s110
                                 + 1.05792 w_k1_MSFT_s110
                                 + 0.842536 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.0402 w_k1_ADM_s111
                                 + 1.34164 w_k1_AXP_s111
                                 + 1.06558 w_k1_DIS_s111
                                 + 0.986816 w_k1_FDX_s111
                                 + 1.12571 w_k1_IBM_s111
                                 + 0.966314 w_k1_INTC_s111
                                 + 1.06534 w_k1_KO_s111 + 1.39921 w_k1_MAT_s111
                                 + 0.949847 w_k1_MSFT_s111
                                 + 1.16725 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.883504 w_k1_ADM_s111
                                 + 0.841951 w_k1_AXP_s111
                                 + 0.815833 w_k1_DIS_s111
                                 + 0.827559 w_k1_FDX_s111
                                 + 1.14622 w_k1_IBM_s111
                                 + 0.902511 w_k1_INTC_s111
                                 + 1.04344 w_k1_KO_s111 + 1.16246 w_k1_MAT_s111
                                 + 0.869859 w_k1_MSFT_s111
                                 + 0.912354 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.23064 w_k1_ADM_s111
                                 + 1.38357 w_k1_AXP_s111
                                 + 1.16232 w_k1_DIS_s111
                                 + 1.22262 w_k1_FDX_s111
                                 + 0.918783 w_k1_IBM_s111
                                 + 1.05885 w_k1_INTC_s111
                                 + 1.03384 w_k1_KO_s111
                                 + 0.876696 w_k1_MAT_s111
                                 + 1.06519 w_k1_MSFT_s111
                                 + 1.16616 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.714647 w_k1_ADM_s111
                                 + 0.752004 w_k1_AXP_s111
                                 + 0.580821 w_k1_DIS_s111
                                 + 0.735697 w_k1_FDX_s111
                                 + 0.746388 w_k1_IBM_s111
                                 + 0.694213 w_k1_INTC_s111
                                 + 0.711455 w_k1_KO_s111
                                 + 0.712142 w_k1_MAT_s111
                                 + 0.758376 w_k1_MSFT_s111
                                 + 0.82711 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.52844 w_k1_ADM_s111
                                 + 1.87754 w_k1_AXP_s111
                                 + 1.76707 w_k1_DIS_s111
                                 + 1.56902 w_k1_FDX_s111
                                 + 1.06816 w_k1_IBM_s111
                                 + 1.42121 w_k1_INTC_s111
                                 + 1.30838 w_k1_KO_s111
                                 + 0.960245 w_k1_MAT_s111
                                 + 1.62006 w_k1_MSFT_s111
                                 + 1.28951 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.37344 w_k1_ADM_s111
                                 + 1.34421 w_k1_AXP_s111
                                 + 1.31839 w_k1_DIS_s111
                                 + 1.04763 w_k1_FDX_s111
                                 + 1.47028 w_k1_IBM_s111
                                 + 0.951271 w_k1_INTC_s111
                                 + 1.14803 w_k1_KO_s111 + 1.37077 w_k1_MAT_s111
                                 + 1.02649 w_k1_MSFT_s111
                                 + 1.09348 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.0612 w_k1_ADM_s111
                                 + 0.912339 w_k1_AXP_s111
                                 + 1.01388 w_k1_DIS_s111 + 1.0357 w_k1_FDX_s111
                                 + 0.728439 w_k1_IBM_s111
                                 + 1.06112 w_k1_INTC_s111
                                 + 0.939522 w_k1_KO_s111
                                 + 0.750464 w_k1_MAT_s111
                                 + 1.06412 w_k1_MSFT_s111
                                 + 1.03416 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.888151 w_k1_ADM_s111
                                 + 0.887811 w_k1_AXP_s111
                                 + 0.76553 w_k1_DIS_s111
                                 + 0.661667 w_k1_FDX_s111
                                 + 1.53481 w_k1_IBM_s111
                                 + 0.892899 w_k1_INTC_s111
                                 + 1.00373 w_k1_KO_s111 + 1.02031 w_k1_MAT_s111
                                 + 0.727599 w_k1_MSFT_s111
                                 + 0.957428 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.973343 w_k1_ADM_s112
                                 + 1.32492 w_k1_AXP_s112
                                 + 1.27724 w_k1_DIS_s112
                                 + 1.26443 w_k1_FDX_s112
                                 + 0.811986 w_k1_IBM_s112
                                 + 1.45105 w_k1_INTC_s112
                                 + 0.978105 w_k1_KO_s112
                                 + 0.896396 w_k1_MAT_s112
                                 + 1.12515 w_k1_MSFT_s112 + 1.045 w_k1_PG_s112
                                 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.973571 w_k1_ADM_s112
                                 + 1.0885 w_k1_AXP_s112
                                 + 0.959518 w_k1_DIS_s112
                                 + 1.11188 w_k1_FDX_s112
                                 + 0.640538 w_k1_IBM_s112
                                 + 1.06662 w_k1_INTC_s112
                                 + 0.86381 w_k1_KO_s112
                                 + 0.606388 w_k1_MAT_s112
                                 + 1.01564 w_k1_MSFT_s112
                                 + 1.05692 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.985553 w_k1_ADM_s112
                                 + 1.3265 w_k1_AXP_s112
                                 + 0.986469 w_k1_DIS_s112
                                 + 1.10068 w_k1_FDX_s112 + 1.156 w_k1_IBM_s112
                                 + 1.29565 w_k1_INTC_s112
                                 + 1.02094 w_k1_KO_s112 + 1.21509 w_k1_MAT_s112
                                 + 0.970985 w_k1_MSFT_s112
                                 + 1.06482 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.685066 w_k1_ADM_s112
                                 + 0.752447 w_k1_AXP_s112
                                 + 0.668029 w_k1_DIS_s112
                                 + 0.751543 w_k1_FDX_s112
                                 + 0.829326 w_k1_IBM_s112
                                 + 0.75323 w_k1_INTC_s112
                                 + 0.751675 w_k1_KO_s112
                                 + 0.765757 w_k1_MAT_s112
                                 + 0.744837 w_k1_MSFT_s112
                                 + 0.817632 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.729683 w_k1_ADM_s112
                                 + 1.09765 w_k1_AXP_s112
                                 + 1.05362 w_k1_DIS_s112
                                 + 0.927609 w_k1_FDX_s112
                                 + 1.50635 w_k1_IBM_s112
                                 + 1.04252 w_k1_INTC_s112
                                 + 1.16191 w_k1_KO_s112 + 1.34206 w_k1_MAT_s112
                                 + 0.895993 w_k1_MSFT_s112
                                 + 1.02869 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.928168 w_k1_ADM_s112
                                 + 0.957064 w_k1_AXP_s112
                                 + 0.941917 w_k1_DIS_s112
                                 + 0.967901 w_k1_FDX_s112
                                 + 0.843868 w_k1_IBM_s112
                                 + 1.04435 w_k1_INTC_s112
                                 + 0.905704 w_k1_KO_s112
                                 + 0.605603 w_k1_MAT_s112
                                 + 0.894385 w_k1_MSFT_s112
                                 + 0.854141 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.11993 w_k1_ADM_s112
                                 + 1.18247 w_k1_AXP_s112
                                 + 1.51289 w_k1_DIS_s112
                                 + 1.31214 w_k1_FDX_s112
                                 + 1.25543 w_k1_IBM_s112
                                 + 1.31021 w_k1_INTC_s112
                                 + 1.30419 w_k1_KO_s112 + 1.12282 w_k1_MAT_s112
                                 + 1.28141 w_k1_MSFT_s112
                                 + 1.16291 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.83973 w_k1_ADM_s112
                                 + 1.06953 w_k1_AXP_s112 + 0.8609 w_k1_DIS_s112
                                 + 0.876577 w_k1_FDX_s112
                                 + 1.1012 w_k1_IBM_s112
                                 + 1.06872 w_k1_INTC_s112
                                 + 0.898499 w_k1_KO_s112
                                 + 0.958796 w_k1_MAT_s112
                                 + 0.818996 w_k1_MSFT_s112
                                 + 0.919685 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.24808 w_k1_ADM_s113
                                 + 1.25525 w_k1_AXP_s113
                                 + 1.40756 w_k1_DIS_s113
                                 + 1.13549 w_k1_FDX_s113
                                 + 1.01792 w_k1_IBM_s113
                                 + 1.0369 w_k1_INTC_s113 + 1.06723 w_k1_KO_s113
                                 + 1.14518 w_k1_MAT_s113
                                 + 1.09663 w_k1_MSFT_s113
                                 + 1.06503 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.17414 w_k1_ADM_s113
                                 + 1.25272 w_k1_AXP_s113
                                 + 1.08734 w_k1_DIS_s113
                                 + 1.16631 w_k1_FDX_s113
                                 + 0.767887 w_k1_IBM_s113
                                 + 1.05007 w_k1_INTC_s113
                                 + 0.834762 w_k1_KO_s113
                                 + 0.957545 w_k1_MAT_s113
                                 + 1.07342 w_k1_MSFT_s113
                                 + 1.06899 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.712047 w_k1_ADM_s113
                                 + 0.565548 w_k1_AXP_s113
                                 + 0.647176 w_k1_DIS_s113
                                 + 0.673185 w_k1_FDX_s113
                                 + 0.86546 w_k1_IBM_s113
                                 + 0.726776 w_k1_INTC_s113
                                 + 0.746272 w_k1_KO_s113
                                 + 0.762676 w_k1_MAT_s113
                                 + 0.766095 w_k1_MSFT_s113
                                 + 0.816688 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.832865 w_k1_ADM_s113
                                 + 0.953874 w_k1_AXP_s113
                                 + 0.897672 w_k1_DIS_s113
                                 + 0.869808 w_k1_FDX_s113
                                 + 0.883118 w_k1_IBM_s113
                                 + 0.87316 w_k1_INTC_s113
                                 + 0.748432 w_k1_KO_s113
                                 + 0.917848 w_k1_MAT_s113
                                 + 0.818381 w_k1_MSFT_s113
                                 + 0.893365 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.876502 w_k1_ADM_s113
                                 + 0.925693 w_k1_AXP_s113
                                 + 0.827019 w_k1_DIS_s113
                                 + 0.819674 w_k1_FDX_s113
                                 + 1.34541 w_k1_IBM_s113
                                 + 0.928676 w_k1_INTC_s113
                                 + 1.02221 w_k1_KO_s113
                                 + 0.924966 w_k1_MAT_s113
                                 + 0.834214 w_k1_MSFT_s113
                                 + 0.958941 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.27871 w_k1_ADM_s113
                                 + 1.3865 w_k1_AXP_s113 + 1.58993 w_k1_DIS_s113
                                 + 1.30957 w_k1_FDX_s113
                                 + 1.33455 w_k1_IBM_s113
                                 + 1.26809 w_k1_INTC_s113
                                 + 1.41904 w_k1_KO_s113 + 1.21825 w_k1_MAT_s113
                                 + 1.21032 w_k1_MSFT_s113
                                 + 1.11056 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.19633 w_k1_ADM_s113
                                 + 0.94606 w_k1_AXP_s113
                                 + 1.08234 w_k1_DIS_s113
                                 + 0.959558 w_k1_FDX_s113
                                 + 1.17126 w_k1_IBM_s113
                                 + 1.01347 w_k1_INTC_s113
                                 + 1.07683 w_k1_KO_s113
                                 + 0.844217 w_k1_MAT_s113
                                 + 0.970658 w_k1_MSFT_s113
                                 + 0.999619 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.844451 w_k1_ADM_s113
                                 + 0.765518 w_k1_AXP_s113
                                 + 0.770098 w_k1_DIS_s113
                                 + 0.858765 w_k1_FDX_s113
                                 + 0.649327 w_k1_IBM_s113
                                 + 0.996733 w_k1_INTC_s113
                                 + 0.81095 w_k1_KO_s113
                                 + 0.625566 w_k1_MAT_s113
                                 + 0.875899 w_k1_MSFT_s113
                                 + 0.784163 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.911794 w_k1_ADM_s114
                                 + 0.710734 w_k1_AXP_s114
                                 + 1.10279 w_k1_DIS_s114
                                 + 0.86038 w_k1_FDX_s114
                                 + 1.10905 w_k1_IBM_s114
                                 + 0.96152 w_k1_INTC_s114
                                 + 1.12967 w_k1_KO_s114 + 1.22336 w_k1_MAT_s114
                                 + 1.05248 w_k1_MSFT_s114
                                 + 0.907217 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.15266 w_k1_ADM_s114
                                 + 1.14048 w_k1_AXP_s114
                                 + 1.42324 w_k1_DIS_s114
                                 + 1.28807 w_k1_FDX_s114
                                 + 0.820126 w_k1_IBM_s114
                                 + 1.13546 w_k1_INTC_s114
                                 + 1.08709 w_k1_KO_s114
                                 + 0.914071 w_k1_MAT_s114
                                 + 1.29917 w_k1_MSFT_s114
                                 + 1.01017 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.949706 w_k1_ADM_s114
                                 + 1.19811 w_k1_AXP_s114
                                 + 1.21863 w_k1_DIS_s114
                                 + 0.994468 w_k1_FDX_s114
                                 + 1.43816 w_k1_IBM_s114
                                 + 1.01208 w_k1_INTC_s114
                                 + 1.39404 w_k1_KO_s114 + 1.40153 w_k1_MAT_s114
                                 + 1.00341 w_k1_MSFT_s114
                                 + 1.04169 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.1177 w_k1_ADM_s114
                                 + 1.34318 w_k1_AXP_s114
                                 + 1.47162 w_k1_DIS_s114
                                 + 1.27103 w_k1_FDX_s114 + 1.1375 w_k1_IBM_s114
                                 + 1.29951 w_k1_INTC_s114
                                 + 1.40499 w_k1_KO_s114 + 1.04481 w_k1_MAT_s114
                                 + 1.15867 w_k1_MSFT_s114
                                 + 1.19024 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.14817 w_k1_ADM_s114
                                 + 1.41766 w_k1_AXP_s114
                                 + 1.95045 w_k1_DIS_s114
                                 + 1.33509 w_k1_FDX_s114
                                 + 1.90308 w_k1_IBM_s114
                                 + 1.29821 w_k1_INTC_s114
                                 + 1.35521 w_k1_KO_s114 + 1.45135 w_k1_MAT_s114
                                 + 1.37325 w_k1_MSFT_s114
                                 + 1.28299 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.912396 w_k1_ADM_s114
                                 + 0.959238 w_k1_AXP_s114
                                 + 0.865103 w_k1_DIS_s114
                                 + 0.917098 w_k1_FDX_s114
                                 + 1.05943 w_k1_IBM_s114
                                 + 1.07589 w_k1_INTC_s114
                                 + 1.04025 w_k1_KO_s114 + 1.18908 w_k1_MAT_s114
                                 + 0.963914 w_k1_MSFT_s114
                                 + 1.02009 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.676194 w_k1_ADM_s114
                                 + 0.693009 w_k1_AXP_s114
                                 + 0.701868 w_k1_DIS_s114
                                 + 0.771697 w_k1_FDX_s114
                                 + 0.669429 w_k1_IBM_s114
                                 + 0.826614 w_k1_INTC_s114
                                 + 0.765073 w_k1_KO_s114
                                 + 0.778612 w_k1_MAT_s114
                                 + 0.757801 w_k1_MSFT_s114
                                 + 0.868948 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.871203 w_k1_ADM_s114
                                 + 0.772236 w_k1_AXP_s114
                                 + 1.00117 w_k1_DIS_s114
                                 + 1.05717 w_k1_FDX_s114
                                 + 0.640582 w_k1_IBM_s114
                                 + 0.998486 w_k1_INTC_s114
                                 + 0.84491 w_k1_KO_s114
                                 + 0.648545 w_k1_MAT_s114
                                 + 1.00432 w_k1_MSFT_s114
                                 + 0.928214 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.810972 w_k1_ADM_s115
                                 + 0.8692 w_k1_AXP_s115
                                 + 0.703338 w_k1_DIS_s115
                                 + 0.985264 w_k1_FDX_s115
                                 + 0.688331 w_k1_IBM_s115
                                 + 1.14085 w_k1_INTC_s115
                                 + 0.753002 w_k1_KO_s115
                                 + 0.751596 w_k1_MAT_s115
                                 + 1.03055 w_k1_MSFT_s115
                                 + 0.872658 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.06903 w_k1_ADM_s115
                                 + 1.98426 w_k1_AXP_s115
                                 + 1.52491 w_k1_DIS_s115
                                 + 1.45022 w_k1_FDX_s115
                                 + 1.17627 w_k1_IBM_s115
                                 + 1.37189 w_k1_INTC_s115
                                 + 1.15694 w_k1_KO_s115 + 1.78567 w_k1_MAT_s115
                                 + 1.43052 w_k1_MSFT_s115
                                 + 1.37357 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.0537 w_k1_ADM_s115
                                 + 1.59638 w_k1_AXP_s115
                                 + 1.40153 w_k1_DIS_s115
                                 + 1.42771 w_k1_FDX_s115
                                 + 1.10417 w_k1_IBM_s115
                                 + 1.66181 w_k1_INTC_s115
                                 + 1.12059 w_k1_KO_s115 + 0.94238 w_k1_MAT_s115
                                 + 1.38634 w_k1_MSFT_s115 + 1.1476 w_k1_PG_s115
                                 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.972059 w_k1_ADM_s115
                                 + 1.10106 w_k1_AXP_s115
                                 + 1.00311 w_k1_DIS_s115
                                 + 1.04065 w_k1_FDX_s115
                                 + 1.51303 w_k1_IBM_s115
                                 + 1.07641 w_k1_INTC_s115
                                 + 1.09219 w_k1_KO_s115 + 1.62157 w_k1_MAT_s115
                                 + 1.05004 w_k1_MSFT_s115
                                 + 1.06819 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.962956 w_k1_ADM_s115
                                 + 1.08794 w_k1_AXP_s115
                                 + 1.19445 w_k1_DIS_s115
                                 + 1.17291 w_k1_FDX_s115
                                 + 0.965765 w_k1_IBM_s115
                                 + 1.17522 w_k1_INTC_s115
                                 + 1.08508 w_k1_KO_s115 + 1.28309 w_k1_MAT_s115
                                 + 1.15898 w_k1_MSFT_s115 + 1.1179 w_k1_PG_s115
                                 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.733851 w_k1_ADM_s115
                                 + 0.682067 w_k1_AXP_s115
                                 + 0.552139 w_k1_DIS_s115
                                 + 0.757706 w_k1_FDX_s115
                                 + 0.872186 w_k1_IBM_s115
                                 + 0.868218 w_k1_INTC_s115
                                 + 0.74364 w_k1_KO_s115
                                 + 0.959941 w_k1_MAT_s115
                                 + 0.800245 w_k1_MSFT_s115
                                 + 0.796395 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.925333 w_k1_ADM_s115
                                 + 0.956299 w_k1_AXP_s115
                                 + 1.14585 w_k1_DIS_s115
                                 + 1.04485 w_k1_FDX_s115
                                 + 0.813857 w_k1_IBM_s115
                                 + 1.02212 w_k1_INTC_s115
                                 + 0.905977 w_k1_KO_s115 + 0.6337 w_k1_MAT_s115
                                 + 1.10078 w_k1_MSFT_s115
                                 + 0.994123 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.715386 w_k1_ADM_s115
                                 + 0.818208 w_k1_AXP_s115
                                 + 0.943133 w_k1_DIS_s115
                                 + 0.935178 w_k1_FDX_s115
                                 + 1.05736 w_k1_IBM_s115
                                 + 1.13817 w_k1_INTC_s115
                                 + 1.04107 w_k1_KO_s115 + 1.10117 w_k1_MAT_s115
                                 + 1.20198 w_k1_MSFT_s115
                                 + 0.979292 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.889924 w_k1_ADM_s116
                                 + 0.941441 w_k1_AXP_s116
                                 + 0.944994 w_k1_DIS_s116
                                 + 0.920513 w_k1_FDX_s116
                                 + 1.18233 w_k1_IBM_s116
                                 + 1.00211 w_k1_INTC_s116 + 1.009 w_k1_KO_s116
                                 + 1.53915 w_k1_MAT_s116
                                 + 0.929748 w_k1_MSFT_s116
                                 + 0.968372 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.57313 w_k1_ADM_s116
                                 + 1.64401 w_k1_AXP_s116
                                 + 1.30299 w_k1_DIS_s116
                                 + 1.22632 w_k1_FDX_s116
                                 + 1.32515 w_k1_IBM_s116
                                 + 1.15541 w_k1_INTC_s116
                                 + 1.37737 w_k1_KO_s116 + 1.61394 w_k1_MAT_s116
                                 + 1.17981 w_k1_MSFT_s116 + 1.2453 w_k1_PG_s116
                                 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.936063 w_k1_ADM_s116
                                 + 0.808041 w_k1_AXP_s116
                                 + 0.648943 w_k1_DIS_s116
                                 + 0.678156 w_k1_FDX_s116
                                 + 1.10002 w_k1_IBM_s116
                                 + 0.902063 w_k1_INTC_s116
                                 + 1.00048 w_k1_KO_s116 + 1.19717 w_k1_MAT_s116
                                 + 0.739655 w_k1_MSFT_s116
                                 + 0.792033 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.26853 w_k1_ADM_s116
                                 + 1.04186 w_k1_AXP_s116
                                 + 1.19427 w_k1_DIS_s116
                                 + 1.07352 w_k1_FDX_s116
                                 + 0.889975 w_k1_IBM_s116
                                 + 1.24925 w_k1_INTC_s116
                                 + 1.33562 w_k1_KO_s116 + 1.20106 w_k1_MAT_s116
                                 + 1.10629 w_k1_MSFT_s116
                                 + 1.08972 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.868717 w_k1_ADM_s116
                                 + 1.09331 w_k1_AXP_s116
                                 + 1.13389 w_k1_DIS_s116
                                 + 0.922179 w_k1_FDX_s116
                                 + 1.78866 w_k1_IBM_s116
                                 + 1.06795 w_k1_INTC_s116
                                 + 1.17976 w_k1_KO_s116 + 1.66054 w_k1_MAT_s116
                                 + 0.970662 w_k1_MSFT_s116
                                 + 1.05309 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.75653 w_k1_ADM_s116
                                 + 0.674703 w_k1_AXP_s116
                                 + 0.726107 w_k1_DIS_s116
                                 + 0.773291 w_k1_FDX_s116
                                 + 0.690364 w_k1_IBM_s116
                                 + 0.988532 w_k1_INTC_s116
                                 + 0.816297 w_k1_KO_s116
                                 + 0.808773 w_k1_MAT_s116
                                 + 0.870014 w_k1_MSFT_s116
                                 + 0.905166 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.35384 w_k1_ADM_s116
                                 + 1.2876 w_k1_AXP_s116 + 1.52456 w_k1_DIS_s116
                                 + 1.34231 w_k1_FDX_s116
                                 + 0.892172 w_k1_IBM_s116
                                 + 1.15378 w_k1_INTC_s116
                                 + 1.18213 w_k1_KO_s116 + 1.04819 w_k1_MAT_s116
                                 + 1.21212 w_k1_MSFT_s116
                                 + 1.03772 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.03153 w_k1_ADM_s116
                                 + 0.907888 w_k1_AXP_s116
                                 + 0.977541 w_k1_DIS_s116
                                 + 0.970853 w_k1_FDX_s116
                                 + 0.722595 w_k1_IBM_s116
                                 + 1.08479 w_k1_INTC_s116
                                 + 0.805616 w_k1_KO_s116
                                 + 0.631983 w_k1_MAT_s116
                                 + 0.936782 w_k1_MSFT_s116
                                 + 0.933037 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.13601 w_k1_ADM_s117
                                 + 1.57069 w_k1_AXP_s117
                                 + 1.71132 w_k1_DIS_s117
                                 + 1.38359 w_k1_FDX_s117
                                 + 1.30327 w_k1_IBM_s117
                                 + 1.3431 w_k1_INTC_s117 + 1.37121 w_k1_KO_s117
                                 + 1.44156 w_k1_MAT_s117
                                 + 1.31996 w_k1_MSFT_s117
                                 + 1.40314 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.953796 w_k1_ADM_s117
                                 + 1.30788 w_k1_AXP_s117
                                 + 1.03104 w_k1_DIS_s117
                                 + 1.10553 w_k1_FDX_s117
                                 + 1.03343 w_k1_IBM_s117
                                 + 1.20621 w_k1_INTC_s117
                                 + 0.983468 w_k1_KO_s117
                                 + 1.01631 w_k1_MAT_s117
                                 + 1.11081 w_k1_MSFT_s117 + 1.0662 w_k1_PG_s117
                                 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.0108 w_k1_ADM_s117
                                 + 1.32306 w_k1_AXP_s117
                                 + 1.10165 w_k1_DIS_s117
                                 + 1.32661 w_k1_FDX_s117
                                 + 0.76242 w_k1_IBM_s117
                                 + 1.4299 w_k1_INTC_s117
                                 + 0.853162 w_k1_KO_s117
                                 + 0.615404 w_k1_MAT_s117
                                 + 1.21549 w_k1_MSFT_s117
                                 + 1.10587 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.905875 w_k1_ADM_s117
                                 + 1.09915 w_k1_AXP_s117
                                 + 1.30519 w_k1_DIS_s117
                                 + 0.979942 w_k1_FDX_s117
                                 + 1.1677 w_k1_IBM_s117
                                 + 1.11566 w_k1_INTC_s117
                                 + 1.12361 w_k1_KO_s117
                                 + 0.887632 w_k1_MAT_s117
                                 + 1.05554 w_k1_MSFT_s117
                                 + 1.02129 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.797259 w_k1_ADM_s117
                                 + 1.05445 w_k1_AXP_s117 + 1.0861 w_k1_DIS_s117
                                 + 0.920227 w_k1_FDX_s117
                                 + 1.46469 w_k1_IBM_s117
                                 + 1.06941 w_k1_INTC_s117
                                 + 1.10911 w_k1_KO_s117 + 1.16253 w_k1_MAT_s117
                                 + 0.806494 w_k1_MSFT_s117
                                 + 1.05439 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.10101 w_k1_ADM_s117
                                 + 1.10166 w_k1_AXP_s117
                                 + 0.990588 w_k1_DIS_s117
                                 + 1.05135 w_k1_FDX_s117
                                 + 0.712414 w_k1_IBM_s117
                                 + 1.02851 w_k1_INTC_s117
                                 + 0.850058 w_k1_KO_s117
                                 + 0.487207 w_k1_MAT_s117
                                 + 0.868182 w_k1_MSFT_s117
                                 + 0.918423 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.816086 w_k1_ADM_s117
                                 + 0.831735 w_k1_AXP_s117
                                 + 0.797612 w_k1_DIS_s117
                                 + 0.810589 w_k1_FDX_s117
                                 + 0.899341 w_k1_IBM_s117
                                 + 0.91427 w_k1_INTC_s117
                                 + 0.75392 w_k1_KO_s117
                                 + 0.626288 w_k1_MAT_s117
                                 + 0.847038 w_k1_MSFT_s117
                                 + 0.781766 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.26897 w_k1_ADM_s117
                                 + 1.2725 w_k1_AXP_s117 + 1.3 w_k1_DIS_s117
                                 + 1.10746 w_k1_FDX_s117
                                 + 1.38797 w_k1_IBM_s117
                                 + 1.20832 w_k1_INTC_s117
                                 + 1.08405 w_k1_KO_s117
                                 + 0.831391 w_k1_MAT_s117
                                 + 1.05559 w_k1_MSFT_s117
                                 + 1.03069 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0272362886351708 w_k1_ADM_s000
                                 + 0.0732396440892359 w_k1_ADM_s001
                                 + 0.0271796633639969 w_k1_ADM_s002
                                 + 0.042029327829104 w_k1_ADM_s003
                                 + 0.0359354406082268 w_k1_ADM_s004
                                 + 0.0364412215430186 w_k1_ADM_s005
                                 + 0.0378101094562459 w_k1_ADM_s006
                                 + 0.0262045612187303 w_k1_ADM_s007
                                 + 0.0220473008912264 w_k1_ADM_s010
                                 + 0.0119605484312307 w_k1_ADM_s011
                                 + 0.0105022386657316 w_k1_ADM_s012
                                 + 0.0249864182754544 w_k1_ADM_s013
                                 + 0.0113035625741864 w_k1_ADM_s014
                                 + 0.0129270111741861 w_k1_ADM_s015
                                 + 0.01010176347509 w_k1_ADM_s016
                                 + 0.0177892700296403 w_k1_ADM_s017
                                 + 0.0412266748941472 w_k1_ADM_s100
                                 + 0.0268863020440262 w_k1_ADM_s101
                                 + 0.00751008462541211 w_k1_ADM_s102
                                 + 0.0090516405718831 w_k1_ADM_s103
                                 + 0.00761274907808177 w_k1_ADM_s104
                                 + 0.0301627878404176 w_k1_ADM_s105
                                 + 0.00987705026469736 w_k1_ADM_s106
                                 + 0.00708611673121902 w_k1_ADM_s107
                                 + 0.039409495228336 w_k1_ADM_s110
                                 + 0.0670791359873674 w_k1_ADM_s111
                                 + 0.065754130701806 w_k1_ADM_s112
                                 + 0.0357385474073167 w_k1_ADM_s113
                                 + 0.0936822436050627 w_k1_ADM_s114
                                 + 0.0349849712954429 w_k1_ADM_s115
                                 + 0.0708693472582805 w_k1_ADM_s116
                                 + 0.033322311736698 w_k1_ADM_s117
                                 + 0.0281261901545541 w_k1_AXP_s000
                                 + 0.0664936970418435 w_k1_AXP_s001
                                 + 0.0299599582978568 w_k1_AXP_s002
                                 + 0.0395892632535966 w_k1_AXP_s003
                                 + 0.0367020085325884 w_k1_AXP_s004
                                 + 0.0319321376788561 w_k1_AXP_s005
                                 + 0.0432271658794122 w_k1_AXP_s006
                                 + 0.0267087795182498 w_k1_AXP_s007
                                 + 0.0235935324140007 w_k1_AXP_s010
                                 + 0.0102131705509882 w_k1_AXP_s011
                                 + 0.0107339911654987 w_k1_AXP_s012
                                 + 0.0243988459978048 w_k1_AXP_s013
                                 + 0.0104561639997626 w_k1_AXP_s014
                                 + 0.0143556376292532 w_k1_AXP_s015
                                 + 0.0114304317710476 w_k1_AXP_s016
                                 + 0.0157352538320392 w_k1_AXP_s017
                                 + 0.0384743433480182 w_k1_AXP_s100
                                 + 0.0298335482778224 w_k1_AXP_s101
                                 + 0.0077304243557837 w_k1_AXP_s102
                                 + 0.00991172492279541 w_k1_AXP_s103
                                 + 0.00875011452684076 w_k1_AXP_s104
                                 + 0.0275315228326453 w_k1_AXP_s105
                                 + 0.0095710240074108 w_k1_AXP_s106
                                 + 0.00763343842381538 w_k1_AXP_s107
                                 + 0.0384202465782189 w_k1_AXP_s110
                                 + 0.0694891115689525 w_k1_AXP_s111
                                 + 0.0773309393645926 w_k1_AXP_s112
                                 + 0.0358304071118134 w_k1_AXP_s113
                                 + 0.0990618468015163 w_k1_AXP_s114
                                 + 0.0411145782703569 w_k1_AXP_s115
                                 + 0.0680120355215975 w_k1_AXP_s116
                                 + 0.038936761707622 w_k1_AXP_s117
                                 + 0.0285580760417189 w_k1_DIS_s000
                                 + 0.0667970525219321 w_k1_DIS_s001
                                 + 0.0321419363400979 w_k1_DIS_s002
                                 + 0.0388262822844947 w_k1_DIS_s003
                                 + 0.0395353785193933 w_k1_DIS_s004
                                 + 0.0353652176640991 w_k1_DIS_s005
                                 + 0.0423271449641023 w_k1_DIS_s006
                                 + 0.0260683713703562 w_k1_DIS_s007
                                 + 0.0244160180610778 w_k1_DIS_s010
                                 + 0.0115272510290851 w_k1_DIS_s011
                                 + 0.00997575910441485 w_k1_DIS_s012
                                 + 0.0236650121676481 w_k1_DIS_s013
                                 + 0.0108807255221165 w_k1_DIS_s014
                                 + 0.0147781227637125 w_k1_DIS_s015
                                 + 0.0113924938300671 w_k1_DIS_s016
                                 + 0.0156505158866619 w_k1_DIS_s017
                                 + 0.0388974995511202 w_k1_DIS_s100
                                 + 0.0279520871755686 w_k1_DIS_s101
                                 + 0.00737629808699978 w_k1_DIS_s102
                                 + 0.00869452018767427 w_k1_DIS_s103
                                 + 0.00836104656371566 w_k1_DIS_s104
                                 + 0.0294706355688689 w_k1_DIS_s105
                                 + 0.00802347509188078 w_k1_DIS_s106
                                 + 0.00879184955711259 w_k1_DIS_s107
                                 + 0.0422825158384326 w_k1_DIS_s110
                                 + 0.0634267669507911 w_k1_DIS_s111
                                 + 0.0747838370648501 w_k1_DIS_s112
                                 + 0.0369099061850219 w_k1_DIS_s113
                                 + 0.113102608614481 w_k1_DIS_s114
                                 + 0.038413575681168 w_k1_DIS_s115
                                 + 0.0697284974190339 w_k1_DIS_s116
                                 + 0.0375955465667384 w_k1_DIS_s117
                                 + 0.0285317873519686 w_k1_FDX_s000
                                 + 0.0663101917368368 w_k1_FDX_s001
                                 + 0.0301023204646019 w_k1_FDX_s002
                                 + 0.0385354734621371 w_k1_FDX_s003
                                 + 0.0373591324086382 w_k1_FDX_s004
                                 + 0.0352979733245091 w_k1_FDX_s005
                                 + 0.0404451837500596 w_k1_FDX_s006
                                 + 0.0265069484522281 w_k1_FDX_s007
                                 + 0.024358939680715 w_k1_FDX_s010
                                 + 0.0108580781600969 w_k1_FDX_s011
                                 + 0.0103649756747498 w_k1_FDX_s012
                                 + 0.0234578136039892 w_k1_FDX_s013
                                 + 0.0101966130788299 w_k1_FDX_s014
                                 + 0.0157651552704365 w_k1_FDX_s015
                                 + 0.0105178447389222 w_k1_FDX_s016
                                 + 0.0152713166916018 w_k1_FDX_s017
                                 + 0.0367389498926739 w_k1_FDX_s100
                                 + 0.0275709867359835 w_k1_FDX_s101
                                 + 0.00736718827823635 w_k1_FDX_s102
                                 + 0.00888338327805984 w_k1_FDX_s103
                                 + 0.00805935306228955 w_k1_FDX_s104
                                 + 0.0283935844745817 w_k1_FDX_s105
                                 + 0.00941038301679488 w_k1_FDX_s106
                                 + 0.00822711453169293 w_k1_FDX_s107
                                 + 0.041807374028056 w_k1_FDX_s110
                                 + 0.0618035710634954 w_k1_FDX_s111
                                 + 0.0742961090934736 w_k1_FDX_s112
                                 + 0.0341306446882914 w_k1_FDX_s113
                                 + 0.102283755913571 w_k1_FDX_s114
                                 + 0.0413290540668463 w_k1_FDX_s115
                                 + 0.065861443215093 w_k1_FDX_s116
                                 + 0.035546224382571 w_k1_FDX_s117
                                 + 0.0276422310562518 w_k1_IBM_s000
                                 + 0.0688155948600202 w_k1_IBM_s001
                                 + 0.0320126109359745 w_k1_IBM_s002
                                 + 0.0394148516842869 w_k1_IBM_s003
                                 + 0.0384601060772946 w_k1_IBM_s004
                                 + 0.0348242837717853 w_k1_IBM_s005
                                 + 0.040855669248835 w_k1_IBM_s006
                                 + 0.0272563221277526 w_k1_IBM_s007
                                 + 0.0235302752939929 w_k1_IBM_s010
                                 + 0.0110685555193053 w_k1_IBM_s011
                                 + 0.0103206984151854 w_k1_IBM_s012
                                 + 0.0232102152950066 w_k1_IBM_s013
                                 + 0.0107059043184573 w_k1_IBM_s014
                                 + 0.0152707531424233 w_k1_IBM_s015
                                 + 0.0104794593190353 w_k1_IBM_s016
                                 + 0.0157440076013602 w_k1_IBM_s017
                                 + 0.03995151014223 w_k1_IBM_s100
                                 + 0.0281742600920631 w_k1_IBM_s101
                                 + 0.00719399694013728 w_k1_IBM_s102
                                 + 0.00866933501575514 w_k1_IBM_s103
                                 + 0.00920323079546588 w_k1_IBM_s104
                                 + 0.030146842790126 w_k1_IBM_s105
                                 + 0.00887143877728166 w_k1_IBM_s106
                                 + 0.00801622576479972 w_k1_IBM_s107
                                 + 0.0435722029471755 w_k1_IBM_s110
                                 + 0.0667162131558019 w_k1_IBM_s111
                                 + 0.0741018976096708 w_k1_IBM_s112
                                 + 0.0355652515530639 w_k1_IBM_s113
                                 + 0.101797535457227 w_k1_IBM_s114
                                 + 0.0414613352732822 w_k1_IBM_s115
                                 + 0.0687064374433856 w_k1_IBM_s116
                                 + 0.0369179071868506 w_k1_IBM_s117
                                 + 0.0319383113016604 w_k1_INTC_s000
                                 + 0.066385845254685 w_k1_INTC_s001
                                 + 0.0323996201409008 w_k1_INTC_s002
                                 + 0.0379919636287956 w_k1_INTC_s003
                                 + 0.0420096680754856 w_k1_INTC_s004
                                 + 0.0350141940491516 w_k1_INTC_s005
                                 + 0.038282807136932 w_k1_INTC_s006
                                 + 0.0296787279790271 w_k1_INTC_s007
                                 + 0.025001006145921 w_k1_INTC_s010
                                 + 0.00890214690316351 w_k1_INTC_s011
                                 + 0.0122553310724004 w_k1_INTC_s012
                                 + 0.0261695354808765 w_k1_INTC_s013
                                 + 0.0106476630546216 w_k1_INTC_s014
                                 + 0.0147029641862331 w_k1_INTC_s015
                                 + 0.0103091984428103 w_k1_INTC_s016
                                 + 0.013432539686318 w_k1_INTC_s017
                                 + 0.0337201906943332 w_k1_INTC_s100
                                 + 0.0288225316586093 w_k1_INTC_s101
                                 + 0.00643745761889808 w_k1_INTC_s102
                                 + 0.00828424259346704 w_k1_INTC_s103
                                 + 0.00745467806995673 w_k1_INTC_s104
                                 + 0.0285146808469484 w_k1_INTC_s105
                                 + 0.00927017070449398 w_k1_INTC_s106
                                 + 0.00802376160202132 w_k1_INTC_s107
                                 + 0.0419577662638315 w_k1_INTC_s110
                                 + 0.0601297251389337 w_k1_INTC_s111
                                 + 0.0803278071290585 w_k1_INTC_s112
                                 + 0.0342447165059496 w_k1_INTC_s113
                                 + 0.10475086113433 w_k1_INTC_s114
                                 + 0.0447064294778818 w_k1_INTC_s115
                                 + 0.0715775106151256 w_k1_INTC_s116
                                 + 0.0383832496348456 w_k1_INTC_s117
                                 + 0.0301533644306652 w_k1_KO_s000
                                 + 0.0625925367125162 w_k1_KO_s001
                                 + 0.0339028215677841 w_k1_KO_s002
                                 + 0.0393855568510165 w_k1_KO_s003
                                 + 0.0381264009819383 w_k1_KO_s004
                                 + 0.0337066497005915 w_k1_KO_s005
                                 + 0.0407480657230177 w_k1_KO_s006
                                 + 0.0267906660539001 w_k1_KO_s007
                                 + 0.0239603248358116 w_k1_KO_s010
                                 + 0.011715202480172 w_k1_KO_s011
                                 + 0.00922506760681796 w_k1_KO_s012
                                 + 0.0241360324155381 w_k1_KO_s013
                                 + 0.0111137928864055 w_k1_KO_s014
                                 + 0.0140200548390184 w_k1_KO_s015
                                 + 0.0103418458540702 w_k1_KO_s016
                                 + 0.0155470104560626 w_k1_KO_s017
                                 + 0.0365201951461011 w_k1_KO_s100
                                 + 0.0256172980717015 w_k1_KO_s101
                                 + 0.00737850122580491 w_k1_KO_s102
                                 + 0.00855427675358193 w_k1_KO_s103
                                 + 0.00781352325739351 w_k1_KO_s104
                                 + 0.0302402799171015 w_k1_KO_s105
                                 + 0.00847732298881223 w_k1_KO_s106
                                 + 0.00762061376036249 w_k1_KO_s107
                                 + 0.0427320161358587 w_k1_KO_s110
                                 + 0.0628378826804468 w_k1_KO_s111
                                 + 0.0717933595793616 w_k1_KO_s112
                                 + 0.0339302979077714 w_k1_KO_s113
                                 + 0.110388585055269 w_k1_KO_s114
                                 + 0.0383410936928333 w_k1_KO_s115
                                 + 0.0707467335016198 w_k1_KO_s116
                                 + 0.0334959622652793 w_k1_KO_s117
                                 + 0.0218855471697866 w_k1_MAT_s000
                                 + 0.0533395698677164 w_k1_MAT_s001
                                 + 0.037121446607305 w_k1_MAT_s002
                                 + 0.0399548807168438 w_k1_MAT_s003
                                 + 0.0365300118579414 w_k1_MAT_s004
                                 + 0.0322728046387966 w_k1_MAT_s005
                                 + 0.0429156690679394 w_k1_MAT_s006
                                 + 0.0245182150650063 w_k1_MAT_s007
                                 + 0.0261477538640807 w_k1_MAT_s010
                                 + 0.010880456305192 w_k1_MAT_s011
                                 + 0.0096700795449722 w_k1_MAT_s012
                                 + 0.0254038855887908 w_k1_MAT_s013
                                 + 0.00955486861924739 w_k1_MAT_s014
                                 + 0.0143518433280664 w_k1_MAT_s015
                                 + 0.0115839630004617 w_k1_MAT_s016
                                 + 0.0158144591216568 w_k1_MAT_s017
                                 + 0.0441450952762766 w_k1_MAT_s100
                                 + 0.0335379626117175 w_k1_MAT_s101
                                 + 0.00785175486222487 w_k1_MAT_s102
                                 + 0.00905780278473607 w_k1_MAT_s103
                                 + 0.0118642484730042 w_k1_MAT_s104
                                 + 0.0368420189619183 w_k1_MAT_s105
                                 + 0.00912519095903672 w_k1_MAT_s106
                                 + 0.00791151773066179 w_k1_MAT_s107
                                 + 0.0379368324030423 w_k1_MAT_s110
                                 + 0.0628886550553451 w_k1_MAT_s111
                                 + 0.0656939141863754 w_k1_MAT_s112
                                 + 0.0325273494410505 w_k1_MAT_s113
                                 + 0.103618169944729 w_k1_MAT_s114
                                 + 0.0461253592432359 w_k1_MAT_s115
                                 + 0.079974567658665 w_k1_MAT_s116
                                 + 0.0294958841142036 w_k1_MAT_s117
                                 + 0.0275453429959434 w_k1_MSFT_s000
                                 + 0.0676712302525001 w_k1_MSFT_s001
                                 + 0.0319820488961581 w_k1_MSFT_s002
                                 + 0.0456842803711629 w_k1_MSFT_s003
                                 + 0.0393221631367015 w_k1_MSFT_s004
                                 + 0.0374048677395162 w_k1_MSFT_s005
                                 + 0.0363733775823296 w_k1_MSFT_s006
                                 + 0.0282762521073587 w_k1_MSFT_s007
                                 + 0.0237116980982289 w_k1_MSFT_s010
                                 + 0.01055577816648 w_k1_MSFT_s011
                                 + 0.0108170070365736 w_k1_MSFT_s012
                                 + 0.0261228099029163 w_k1_MSFT_s013
                                 + 0.010098586193201 w_k1_MSFT_s014
                                 + 0.0147286353684031 w_k1_MSFT_s015
                                 + 0.0104054458223306 w_k1_MSFT_s016
                                 + 0.0146035823535261 w_k1_MSFT_s017
                                 + 0.0362690797001858 w_k1_MSFT_s100
                                 + 0.0269670190781721 w_k1_MSFT_s101
                                 + 0.0083959981052671 w_k1_MSFT_s102
                                 + 0.00778251254891923 w_k1_MSFT_s103
                                 + 0.00772913289843629 w_k1_MSFT_s104
                                 + 0.0292111830964597 w_k1_MSFT_s105
                                 + 0.00836822400633646 w_k1_MSFT_s106
                                 + 0.0082462416835341 w_k1_MSFT_s107
                                 + 0.0412395203928007 w_k1_MSFT_s110
                                 + 0.061103709785797 w_k1_MSFT_s111
                                 + 0.0696624759361983 w_k1_MSFT_s112
                                 + 0.0331164790935405 w_k1_MSFT_s113
                                 + 0.102809677542448 w_k1_MSFT_s114
                                 + 0.043116517150222 w_k1_MSFT_s115
                                 + 0.0659076374122225 w_k1_MSFT_s116
                                 + 0.0343377736920535 w_k1_MSFT_s117
                                 + 0.0277241202836308 w_k1_PG_s000
                                 + 0.067561723424041 w_k1_PG_s001
                                 + 0.0314905565329987 w_k1_PG_s002
                                 + 0.0378709887549429 w_k1_PG_s003
                                 + 0.0351461183803001 w_k1_PG_s004
                                 + 0.036323598745016 w_k1_PG_s005
                                 + 0.0397420462407114 w_k1_PG_s006
                                 + 0.025986249450344 w_k1_PG_s007
                                 + 0.0233523408967313 w_k1_PG_s010
                                 + 0.010166173569686 w_k1_PG_s011
                                 + 0.01060299610543 w_k1_PG_s012
                                 + 0.0214139073386802 w_k1_PG_s013
                                 + 0.0106746297742484 w_k1_PG_s014
                                 + 0.0149336962458457 w_k1_PG_s015
                                 + 0.0109791473067175 w_k1_PG_s016
                                 + 0.0149287375043572 w_k1_PG_s017
                                 + 0.0364348100029589 w_k1_PG_s100
                                 + 0.0281601012216895 w_k1_PG_s101
                                 + 0.00808171296442471 w_k1_PG_s102
                                 + 0.0081670126736959 w_k1_PG_s103
                                 + 0.0077419676651255 w_k1_PG_s104
                                 + 0.0307668829618275 w_k1_PG_s105
                                 + 0.00922182429164236 w_k1_PG_s106
                                 + 0.00745143110550952 w_k1_PG_s107
                                 + 0.0379960157157059 w_k1_PG_s110
                                 + 0.0647949815477465 w_k1_PG_s111
                                 + 0.070537449632184 w_k1_PG_s112
                                 + 0.033440071004275 w_k1_PG_s113
                                 + 0.100046522278395 w_k1_PG_s114
                                 + 0.039950012429115 w_k1_PG_s115
                                 + 0.0663181019402103 w_k1_PG_s116
                                 + 0.0345025072660488 w_k1_PG_s117 >= 1.041
Bounds
 0 <= w_0_ADM <= 1
 0 <= w_0_AXP <= 1
 0 <= w_0_DIS <= 1
 0 <= w_0_FDX <= 1
 0 <= w_0_IBM <= 1
 0 <= w_0_INTC <= 1
 0 <= w_0_KO <= 1
 0 <= w_0_MAT <= 1
 0 <= w_0_MSFT <= 1
 0 <= w_0_PG <= 1
 0 <= w_k1_ADM_s0 <= 1
 0 <= w_k1_ADM_s1 <= 1
 0 <= w_k1_ADM_s00 <= 1
 0 <= w_k1_ADM_s01 <= 1
 0 <= w_k1_ADM_s10 <= 1
 0 <= w_k1_ADM_s11 <= 1
 0 <= w_k1_ADM_s000 <= 1
 0 <= w_k1_ADM_s001 <= 1
 0 <= w_k1_ADM_s002 <= 1
 0 <= w_k1_ADM_s003 <= 1
 0 <= w_k1_ADM_s004 <= 1
 0 <= w_k1_ADM_s005 <= 1
 0 <= w_k1_ADM_s006 <= 1
 0 <= w_k1_ADM_s007 <= 1
 0 <= w_k1_ADM_s010 <= 1
 0 <= w_k1_ADM_s011 <= 1
 0 <= w_k1_ADM_s012 <= 1
 0 <= w_k1_ADM_s013 <= 1
 0 <= w_k1_ADM_s014 <= 1
 0 <= w_k1_ADM_s015 <= 1
 0 <= w_k1_ADM_s016 <= 1
 0 <= w_k1_ADM_s017 <= 1
 0 <= w_k1_ADM_s100 <= 1
 0 <= w_k1_ADM_s101 <= 1
 0 <= w_k1_ADM_s102 <= 1
 0 <= w_k1_ADM_s103 <= 1
 0 <= w_k1_ADM_s104 <= 1
 0 <= w_k1_ADM_s105 <= 1
 0 <= w_k1_ADM_s106 <= 1
 0 <= w_k1_ADM_s107 <= 1
 0 <= w_k1_ADM_s110 <= 1
 0 <= w_k1_ADM_s111 <= 1
 0 <= w_k1_ADM_s112 <= 1
 0 <= w_k1_ADM_s113 <= 1
 0 <= w_k1_ADM_s114 <= 1
 0 <= w_k1_ADM_s115 <= 1
 0 <= w_k1_ADM_s116 <= 1
 0 <= w_k1_ADM_s117 <= 1
 0 <= w_k1_AXP_s0 <= 1
 0 <= w_k1_AXP_s1 <= 1
 0 <= w_k1_AXP_s00 <= 1
 0 <= w_k1_AXP_s01 <= 1
 0 <= w_k1_AXP_s10 <= 1
 0 <= w_k1_AXP_s11 <= 1
 0 <= w_k1_AXP_s000 <= 1
 0 <= w_k1_AXP_s001 <= 1
 0 <= w_k1_AXP_s002 <= 1
 0 <= w_k1_AXP_s003 <= 1
 0 <= w_k1_AXP_s004 <= 1
 0 <= w_k1_AXP_s005 <= 1
 0 <= w_k1_AXP_s006 <= 1
 0 <= w_k1_AXP_s007 <= 1
 0 <= w_k1_AXP_s010 <= 1
 0 <= w_k1_AXP_s011 <= 1
 0 <= w_k1_AXP_s012 <= 1
 0 <= w_k1_AXP_s013 <= 1
 0 <= w_k1_AXP_s014 <= 1
 0 <= w_k1_AXP_s015 <= 1
 0 <= w_k1_AXP_s016 <= 1
 0 <= w_k1_AXP_s017 <= 1
 0 <= w_k1_AXP_s100 <= 1
 0 <= w_k1_AXP_s101 <= 1
 0 <= w_k1_AXP_s102 <= 1
 0 <= w_k1_AXP_s103 <= 1
 0 <= w_k1_AXP_s104 <= 1
 0 <= w_k1_AXP_s105 <= 1
 0 <= w_k1_AXP_s106 <= 1
 0 <= w_k1_AXP_s107 <= 1
 0 <= w_k1_AXP_s110 <= 1
 0 <= w_k1_AXP_s111 <= 1
 0 <= w_k1_AXP_s112 <= 1
 0 <= w_k1_AXP_s113 <= 1
 0 <= w_k1_AXP_s114 <= 1
 0 <= w_k1_AXP_s115 <= 1
 0 <= w_k1_AXP_s116 <= 1
 0 <= w_k1_AXP_s117 <= 1
 0 <= w_k1_DIS_s0 <= 1
 0 <= w_k1_DIS_s1 <= 1
 0 <= w_k1_DIS_s00 <= 1
 0 <= w_k1_DIS_s01 <= 1
 0 <= w_k1_DIS_s10 <= 1
 0 <= w_k1_DIS_s11 <= 1
 0 <= w_k1_DIS_s000 <= 1
 0 <= w_k1_DIS_s001 <= 1
 0 <= w_k1_DIS_s002 <= 1
 0 <= w_k1_DIS_s003 <= 1
 0 <= w_k1_DIS_s004 <= 1
 0 <= w_k1_DIS_s005 <= 1
 0 <= w_k1_DIS_s006 <= 1
 0 <= w_k1_DIS_s007 <= 1
 0 <= w_k1_DIS_s010 <= 1
 0 <= w_k1_DIS_s011 <= 1
 0 <= w_k1_DIS_s012 <= 1
 0 <= w_k1_DIS_s013 <= 1
 0 <= w_k1_DIS_s014 <= 1
 0 <= w_k1_DIS_s015 <= 1
 0 <= w_k1_DIS_s016 <= 1
 0 <= w_k1_DIS_s017 <= 1
 0 <= w_k1_DIS_s100 <= 1
 0 <= w_k1_DIS_s101 <= 1
 0 <= w_k1_DIS_s102 <= 1
 0 <= w_k1_DIS_s103 <= 1
 0 <= w_k1_DIS_s104 <= 1
 0 <= w_k1_DIS_s105 <= 1
 0 <= w_k1_DIS_s106 <= 1
 0 <= w_k1_DIS_s107 <= 1
 0 <= w_k1_DIS_s110 <= 1
 0 <= w_k1_DIS_s111 <= 1
 0 <= w_k1_DIS_s112 <= 1
 0 <= w_k1_DIS_s113 <= 1
 0 <= w_k1_DIS_s114 <= 1
 0 <= w_k1_DIS_s115 <= 1
 0 <= w_k1_DIS_s116 <= 1
 0 <= w_k1_DIS_s117 <= 1
 0 <= w_k1_FDX_s0 <= 1
 0 <= w_k1_FDX_s1 <= 1
 0 <= w_k1_FDX_s00 <= 1
 0 <= w_k1_FDX_s01 <= 1
 0 <= w_k1_FDX_s10 <= 1
 0 <= w_k1_FDX_s11 <= 1
 0 <= w_k1_FDX_s000 <= 1
 0 <= w_k1_FDX_s001 <= 1
 0 <= w_k1_FDX_s002 <= 1
 0 <= w_k1_FDX_s003 <= 1
 0 <= w_k1_FDX_s004 <= 1
 0 <= w_k1_FDX_s005 <= 1
 0 <= w_k1_FDX_s006 <= 1
 0 <= w_k1_FDX_s007 <= 1
 0 <= w_k1_FDX_s010 <= 1
 0 <= w_k1_FDX_s011 <= 1
 0 <= w_k1_FDX_s012 <= 1
 0 <= w_k1_FDX_s013 <= 1
 0 <= w_k1_FDX_s014 <= 1
 0 <= w_k1_FDX_s015 <= 1
 0 <= w_k1_FDX_s016 <= 1
 0 <= w_k1_FDX_s017 <= 1
 0 <= w_k1_FDX_s100 <= 1
 0 <= w_k1_FDX_s101 <= 1
 0 <= w_k1_FDX_s102 <= 1
 0 <= w_k1_FDX_s103 <= 1
 0 <= w_k1_FDX_s104 <= 1
 0 <= w_k1_FDX_s105 <= 1
 0 <= w_k1_FDX_s106 <= 1
 0 <= w_k1_FDX_s107 <= 1
 0 <= w_k1_FDX_s110 <= 1
 0 <= w_k1_FDX_s111 <= 1
 0 <= w_k1_FDX_s112 <= 1
 0 <= w_k1_FDX_s113 <= 1
 0 <= w_k1_FDX_s114 <= 1
 0 <= w_k1_FDX_s115 <= 1
 0 <= w_k1_FDX_s116 <= 1
 0 <= w_k1_FDX_s117 <= 1
 0 <= w_k1_IBM_s0 <= 1
 0 <= w_k1_IBM_s1 <= 1
 0 <= w_k1_IBM_s00 <= 1
 0 <= w_k1_IBM_s01 <= 1
 0 <= w_k1_IBM_s10 <= 1
 0 <= w_k1_IBM_s11 <= 1
 0 <= w_k1_IBM_s000 <= 1
 0 <= w_k1_IBM_s001 <= 1
 0 <= w_k1_IBM_s002 <= 1
 0 <= w_k1_IBM_s003 <= 1
 0 <= w_k1_IBM_s004 <= 1
 0 <= w_k1_IBM_s005 <= 1
 0 <= w_k1_IBM_s006 <= 1
 0 <= w_k1_IBM_s007 <= 1
 0 <= w_k1_IBM_s010 <= 1
 0 <= w_k1_IBM_s011 <= 1
 0 <= w_k1_IBM_s012 <= 1
 0 <= w_k1_IBM_s013 <= 1
 0 <= w_k1_IBM_s014 <= 1
 0 <= w_k1_IBM_s015 <= 1
 0 <= w_k1_IBM_s016 <= 1
 0 <= w_k1_IBM_s017 <= 1
 0 <= w_k1_IBM_s100 <= 1
 0 <= w_k1_IBM_s101 <= 1
 0 <= w_k1_IBM_s102 <= 1
 0 <= w_k1_IBM_s103 <= 1
 0 <= w_k1_IBM_s104 <= 1
 0 <= w_k1_IBM_s105 <= 1
 0 <= w_k1_IBM_s106 <= 1
 0 <= w_k1_IBM_s107 <= 1
 0 <= w_k1_IBM_s110 <= 1
 0 <= w_k1_IBM_s111 <= 1
 0 <= w_k1_IBM_s112 <= 1
 0 <= w_k1_IBM_s113 <= 1
 0 <= w_k1_IBM_s114 <= 1
 0 <= w_k1_IBM_s115 <= 1
 0 <= w_k1_IBM_s116 <= 1
 0 <= w_k1_IBM_s117 <= 1
 0 <= w_k1_INTC_s0 <= 1
 0 <= w_k1_INTC_s1 <= 1
 0 <= w_k1_INTC_s00 <= 1
 0 <= w_k1_INTC_s01 <= 1
 0 <= w_k1_INTC_s10 <= 1
 0 <= w_k1_INTC_s11 <= 1
 0 <= w_k1_INTC_s000 <= 1
 0 <= w_k1_INTC_s001 <= 1
 0 <= w_k1_INTC_s002 <= 1
 0 <= w_k1_INTC_s003 <= 1
 0 <= w_k1_INTC_s004 <= 1
 0 <= w_k1_INTC_s005 <= 1
 0 <= w_k1_INTC_s006 <= 1
 0 <= w_k1_INTC_s007 <= 1
 0 <= w_k1_INTC_s010 <= 1
 0 <= w_k1_INTC_s011 <= 1
 0 <= w_k1_INTC_s012 <= 1
 0 <= w_k1_INTC_s013 <= 1
 0 <= w_k1_INTC_s014 <= 1
 0 <= w_k1_INTC_s015 <= 1
 0 <= w_k1_INTC_s016 <= 1
 0 <= w_k1_INTC_s017 <= 1
 0 <= w_k1_INTC_s100 <= 1
 0 <= w_k1_INTC_s101 <= 1
 0 <= w_k1_INTC_s102 <= 1
 0 <= w_k1_INTC_s103 <= 1
 0 <= w_k1_INTC_s104 <= 1
 0 <= w_k1_INTC_s105 <= 1
 0 <= w_k1_INTC_s106 <= 1
 0 <= w_k1_INTC_s107 <= 1
 0 <= w_k1_INTC_s110 <= 1
 0 <= w_k1_INTC_s111 <= 1
 0 <= w_k1_INTC_s112 <= 1
 0 <= w_k1_INTC_s113 <= 1
 0 <= w_k1_INTC_s114 <= 1
 0 <= w_k1_INTC_s115 <= 1
 0 <= w_k1_INTC_s116 <= 1
 0 <= w_k1_INTC_s117 <= 1
 0 <= w_k1_KO_s0 <= 1
 0 <= w_k1_KO_s1 <= 1
 0 <= w_k1_KO_s00 <= 1
 0 <= w_k1_KO_s01 <= 1
 0 <= w_k1_KO_s10 <= 1
 0 <= w_k1_KO_s11 <= 1
 0 <= w_k1_KO_s000 <= 1
 0 <= w_k1_KO_s001 <= 1
 0 <= w_k1_KO_s002 <= 1
 0 <= w_k1_KO_s003 <= 1
 0 <= w_k1_KO_s004 <= 1
 0 <= w_k1_KO_s005 <= 1
 0 <= w_k1_KO_s006 <= 1
 0 <= w_k1_KO_s007 <= 1
 0 <= w_k1_KO_s010 <= 1
 0 <= w_k1_KO_s011 <= 1
 0 <= w_k1_KO_s012 <= 1
 0 <= w_k1_KO_s013 <= 1
 0 <= w_k1_KO_s014 <= 1
 0 <= w_k1_KO_s015 <= 1
 0 <= w_k1_KO_s016 <= 1
 0 <= w_k1_KO_s017 <= 1
 0 <= w_k1_KO_s100 <= 1
 0 <= w_k1_KO_s101 <= 1
 0 <= w_k1_KO_s102 <= 1
 0 <= w_k1_KO_s103 <= 1
 0 <= w_k1_KO_s104 <= 1
 0 <= w_k1_KO_s105 <= 1
 0 <= w_k1_KO_s106 <= 1
 0 <= w_k1_KO_s107 <= 1
 0 <= w_k1_KO_s110 <= 1
 0 <= w_k1_KO_s111 <= 1
 0 <= w_k1_KO_s112 <= 1
 0 <= w_k1_KO_s113 <= 1
 0 <= w_k1_KO_s114 <= 1
 0 <= w_k1_KO_s115 <= 1
 0 <= w_k1_KO_s116 <= 1
 0 <= w_k1_KO_s117 <= 1
 0 <= w_k1_MAT_s0 <= 1
 0 <= w_k1_MAT_s1 <= 1
 0 <= w_k1_MAT_s00 <= 1
 0 <= w_k1_MAT_s01 <= 1
 0 <= w_k1_MAT_s10 <= 1
 0 <= w_k1_MAT_s11 <= 1
 0 <= w_k1_MAT_s000 <= 1
 0 <= w_k1_MAT_s001 <= 1
 0 <= w_k1_MAT_s002 <= 1
 0 <= w_k1_MAT_s003 <= 1
 0 <= w_k1_MAT_s004 <= 1
 0 <= w_k1_MAT_s005 <= 1
 0 <= w_k1_MAT_s006 <= 1
 0 <= w_k1_MAT_s007 <= 1
 0 <= w_k1_MAT_s010 <= 1
 0 <= w_k1_MAT_s011 <= 1
 0 <= w_k1_MAT_s012 <= 1
 0 <= w_k1_MAT_s013 <= 1
 0 <= w_k1_MAT_s014 <= 1
 0 <= w_k1_MAT_s015 <= 1
 0 <= w_k1_MAT_s016 <= 1
 0 <= w_k1_MAT_s017 <= 1
 0 <= w_k1_MAT_s100 <= 1
 0 <= w_k1_MAT_s101 <= 1
 0 <= w_k1_MAT_s102 <= 1
 0 <= w_k1_MAT_s103 <= 1
 0 <= w_k1_MAT_s104 <= 1
 0 <= w_k1_MAT_s105 <= 1
 0 <= w_k1_MAT_s106 <= 1
 0 <= w_k1_MAT_s107 <= 1
 0 <= w_k1_MAT_s110 <= 1
 0 <= w_k1_MAT_s111 <= 1
 0 <= w_k1_MAT_s112 <= 1
 0 <= w_k1_MAT_s113 <= 1
 0 <= w_k1_MAT_s114 <= 1
 0 <= w_k1_MAT_s115 <= 1
 0 <= w_k1_MAT_s116 <= 1
 0 <= w_k1_MAT_s117 <= 1
 0 <= w_k1_MSFT_s0 <= 1
 0 <= w_k1_MSFT_s1 <= 1
 0 <= w_k1_MSFT_s00 <= 1
 0 <= w_k1_MSFT_s01 <= 1
 0 <= w_k1_MSFT_s10 <= 1
 0 <= w_k1_MSFT_s11 <= 1
 0 <= w_k1_MSFT_s000 <= 1
 0 <= w_k1_MSFT_s001 <= 1
 0 <= w_k1_MSFT_s002 <= 1
 0 <= w_k1_MSFT_s003 <= 1
 0 <= w_k1_MSFT_s004 <= 1
 0 <= w_k1_MSFT_s005 <= 1
 0 <= w_k1_MSFT_s006 <= 1
 0 <= w_k1_MSFT_s007 <= 1
 0 <= w_k1_MSFT_s010 <= 1
 0 <= w_k1_MSFT_s011 <= 1
 0 <= w_k1_MSFT_s012 <= 1
 0 <= w_k1_MSFT_s013 <= 1
 0 <= w_k1_MSFT_s014 <= 1
 0 <= w_k1_MSFT_s015 <= 1
 0 <= w_k1_MSFT_s016 <= 1
 0 <= w_k1_MSFT_s017 <= 1
 0 <= w_k1_MSFT_s100 <= 1
 0 <= w_k1_MSFT_s101 <= 1
 0 <= w_k1_MSFT_s102 <= 1
 0 <= w_k1_MSFT_s103 <= 1
 0 <= w_k1_MSFT_s104 <= 1
 0 <= w_k1_MSFT_s105 <= 1
 0 <= w_k1_MSFT_s106 <= 1
 0 <= w_k1_MSFT_s107 <= 1
 0 <= w_k1_MSFT_s110 <= 1
 0 <= w_k1_MSFT_s111 <= 1
 0 <= w_k1_MSFT_s112 <= 1
 0 <= w_k1_MSFT_s113 <= 1
 0 <= w_k1_MSFT_s114 <= 1
 0 <= w_k1_MSFT_s115 <= 1
 0 <= w_k1_MSFT_s116 <= 1
 0 <= w_k1_MSFT_s117 <= 1
 0 <= w_k1_PG_s0 <= 1
 0 <= w_k1_PG_s1 <= 1
 0 <= w_k1_PG_s00 <= 1
 0 <= w_k1_PG_s01 <= 1
 0 <= w_k1_PG_s10 <= 1
 0 <= w_k1_PG_s11 <= 1
 0 <= w_k1_PG_s000 <= 1
 0 <= w_k1_PG_s001 <= 1
 0 <= w_k1_PG_s002 <= 1
 0 <= w_k1_PG_s003 <= 1
 0 <= w_k1_PG_s004 <= 1
 0 <= w_k1_PG_s005 <= 1
 0 <= w_k1_PG_s006 <= 1
 0 <= w_k1_PG_s007 <= 1
 0 <= w_k1_PG_s010 <= 1
 0 <= w_k1_PG_s011 <= 1
 0 <= w_k1_PG_s012 <= 1
 0 <= w_k1_PG_s013 <= 1
 0 <= w_k1_PG_s014 <= 1
 0 <= w_k1_PG_s015 <= 1
 0 <= w_k1_PG_s016 <= 1
 0 <= w_k1_PG_s017 <= 1
 0 <= w_k1_PG_s100 <= 1
 0 <= w_k1_PG_s101 <= 1
 0 <= w_k1_PG_s102 <= 1
 0 <= w_k1_PG_s103 <= 1
 0 <= w_k1_PG_s104 <= 1
 0 <= w_k1_PG_s105 <= 1
 0 <= w_k1_PG_s106 <= 1
 0 <= w_k1_PG_s107 <= 1
 0 <= w_k1_PG_s110 <= 1
 0 <= w_k1_PG_s111 <= 1
 0 <= w_k1_PG_s112 <= 1
 0 <= w_k1_PG_s113 <= 1
 0 <= w_k1_PG_s114 <= 1
 0 <= w_k1_PG_s115 <= 1
 0 <= w_k1_PG_s116 <= 1
 0 <= w_k1_PG_s117 <= 1
 0 <= b_0_ADM <= 0.2
 0 <= b_0_AXP <= 0.2
 0 <= b_0_DIS <= 0.2
 0 <= b_0_FDX <= 0.2
 0 <= b_0_IBM <= 0.2
 0 <= b_0_INTC <= 0.2
 0 <= b_0_KO <= 0.2
 0 <= b_0_MAT <= 0.2
 0 <= b_0_MSFT <= 0.2
 0 <= b_0_PG <= 0.2
 0 <= b_k1_ADM_s0 <= 0.2
 0 <= b_k1_ADM_s1 <= 0.2
 0 <= b_k1_ADM_s00 <= 0.2
 0 <= b_k1_ADM_s01 <= 0.2
 0 <= b_k1_ADM_s10 <= 0.2
 0 <= b_k1_ADM_s11 <= 0.2
 0 <= b_k1_ADM_s000 <= 0.2
 0 <= b_k1_ADM_s001 <= 0.2
 0 <= b_k1_ADM_s002 <= 0.2
 0 <= b_k1_ADM_s003 <= 0.2
 0 <= b_k1_ADM_s004 <= 0.2
 0 <= b_k1_ADM_s005 <= 0.2
 0 <= b_k1_ADM_s006 <= 0.2
 0 <= b_k1_ADM_s007 <= 0.2
 0 <= b_k1_ADM_s010 <= 0.2
 0 <= b_k1_ADM_s011 <= 0.2
 0 <= b_k1_ADM_s012 <= 0.2
 0 <= b_k1_ADM_s013 <= 0.2
 0 <= b_k1_ADM_s014 <= 0.2
 0 <= b_k1_ADM_s015 <= 0.2
 0 <= b_k1_ADM_s016 <= 0.2
 0 <= b_k1_ADM_s017 <= 0.2
 0 <= b_k1_ADM_s100 <= 0.2
 0 <= b_k1_ADM_s101 <= 0.2
 0 <= b_k1_ADM_s102 <= 0.2
 0 <= b_k1_ADM_s103 <= 0.2
 0 <= b_k1_ADM_s104 <= 0.2
 0 <= b_k1_ADM_s105 <= 0.2
 0 <= b_k1_ADM_s106 <= 0.2
 0 <= b_k1_ADM_s107 <= 0.2
 0 <= b_k1_ADM_s110 <= 0.2
 0 <= b_k1_ADM_s111 <= 0.2
 0 <= b_k1_ADM_s112 <= 0.2
 0 <= b_k1_ADM_s113 <= 0.2
 0 <= b_k1_ADM_s114 <= 0.2
 0 <= b_k1_ADM_s115 <= 0.2
 0 <= b_k1_ADM_s116 <= 0.2
 0 <= b_k1_ADM_s117 <= 0.2
 0 <= b_k1_AXP_s0 <= 0.2
 0 <= b_k1_AXP_s1 <= 0.2
 0 <= b_k1_AXP_s00 <= 0.2
 0 <= b_k1_AXP_s01 <= 0.2
 0 <= b_k1_AXP_s10 <= 0.2
 0 <= b_k1_AXP_s11 <= 0.2
 0 <= b_k1_AXP_s000 <= 0.2
 0 <= b_k1_AXP_s001 <= 0.2
 0 <= b_k1_AXP_s002 <= 0.2
 0 <= b_k1_AXP_s003 <= 0.2
 0 <= b_k1_AXP_s004 <= 0.2
 0 <= b_k1_AXP_s005 <= 0.2
 0 <= b_k1_AXP_s006 <= 0.2
 0 <= b_k1_AXP_s007 <= 0.2
 0 <= b_k1_AXP_s010 <= 0.2
 0 <= b_k1_AXP_s011 <= 0.2
 0 <= b_k1_AXP_s012 <= 0.2
 0 <= b_k1_AXP_s013 <= 0.2
 0 <= b_k1_AXP_s014 <= 0.2
 0 <= b_k1_AXP_s015 <= 0.2
 0 <= b_k1_AXP_s016 <= 0.2
 0 <= b_k1_AXP_s017 <= 0.2
 0 <= b_k1_AXP_s100 <= 0.2
 0 <= b_k1_AXP_s101 <= 0.2
 0 <= b_k1_AXP_s102 <= 0.2
 0 <= b_k1_AXP_s103 <= 0.2
 0 <= b_k1_AXP_s104 <= 0.2
 0 <= b_k1_AXP_s105 <= 0.2
 0 <= b_k1_AXP_s106 <= 0.2
 0 <= b_k1_AXP_s107 <= 0.2
 0 <= b_k1_AXP_s110 <= 0.2
 0 <= b_k1_AXP_s111 <= 0.2
 0 <= b_k1_AXP_s112 <= 0.2
 0 <= b_k1_AXP_s113 <= 0.2
 0 <= b_k1_AXP_s114 <= 0.2
 0 <= b_k1_AXP_s115 <= 0.2
 0 <= b_k1_AXP_s116 <= 0.2
 0 <= b_k1_AXP_s117 <= 0.2
 0 <= b_k1_DIS_s0 <= 0.2
 0 <= b_k1_DIS_s1 <= 0.2
 0 <= b_k1_DIS_s00 <= 0.2
 0 <= b_k1_DIS_s01 <= 0.2
 0 <= b_k1_DIS_s10 <= 0.2
 0 <= b_k1_DIS_s11 <= 0.2
 0 <= b_k1_DIS_s000 <= 0.2
 0 <= b_k1_DIS_s001 <= 0.2
 0 <= b_k1_DIS_s002 <= 0.2
 0 <= b_k1_DIS_s003 <= 0.2
 0 <= b_k1_DIS_s004 <= 0.2
 0 <= b_k1_DIS_s005 <= 0.2
 0 <= b_k1_DIS_s006 <= 0.2
 0 <= b_k1_DIS_s007 <= 0.2
 0 <= b_k1_DIS_s010 <= 0.2
 0 <= b_k1_DIS_s011 <= 0.2
 0 <= b_k1_DIS_s012 <= 0.2
 0 <= b_k1_DIS_s013 <= 0.2
 0 <= b_k1_DIS_s014 <= 0.2
 0 <= b_k1_DIS_s015 <= 0.2
 0 <= b_k1_DIS_s016 <= 0.2
 0 <= b_k1_DIS_s017 <= 0.2
 0 <= b_k1_DIS_s100 <= 0.2
 0 <= b_k1_DIS_s101 <= 0.2
 0 <= b_k1_DIS_s102 <= 0.2
 0 <= b_k1_DIS_s103 <= 0.2
 0 <= b_k1_DIS_s104 <= 0.2
 0 <= b_k1_DIS_s105 <= 0.2
 0 <= b_k1_DIS_s106 <= 0.2
 0 <= b_k1_DIS_s107 <= 0.2
 0 <= b_k1_DIS_s110 <= 0.2
 0 <= b_k1_DIS_s111 <= 0.2
 0 <= b_k1_DIS_s112 <= 0.2
 0 <= b_k1_DIS_s113 <= 0.2
 0 <= b_k1_DIS_s114 <= 0.2
 0 <= b_k1_DIS_s115 <= 0.2
 0 <= b_k1_DIS_s116 <= 0.2
 0 <= b_k1_DIS_s117 <= 0.2
 0 <= b_k1_FDX_s0 <= 0.2
 0 <= b_k1_FDX_s1 <= 0.2
 0 <= b_k1_FDX_s00 <= 0.2
 0 <= b_k1_FDX_s01 <= 0.2
 0 <= b_k1_FDX_s10 <= 0.2
 0 <= b_k1_FDX_s11 <= 0.2
 0 <= b_k1_FDX_s000 <= 0.2
 0 <= b_k1_FDX_s001 <= 0.2
 0 <= b_k1_FDX_s002 <= 0.2
 0 <= b_k1_FDX_s003 <= 0.2
 0 <= b_k1_FDX_s004 <= 0.2
 0 <= b_k1_FDX_s005 <= 0.2
 0 <= b_k1_FDX_s006 <= 0.2
 0 <= b_k1_FDX_s007 <= 0.2
 0 <= b_k1_FDX_s010 <= 0.2
 0 <= b_k1_FDX_s011 <= 0.2
 0 <= b_k1_FDX_s012 <= 0.2
 0 <= b_k1_FDX_s013 <= 0.2
 0 <= b_k1_FDX_s014 <= 0.2
 0 <= b_k1_FDX_s015 <= 0.2
 0 <= b_k1_FDX_s016 <= 0.2
 0 <= b_k1_FDX_s017 <= 0.2
 0 <= b_k1_FDX_s100 <= 0.2
 0 <= b_k1_FDX_s101 <= 0.2
 0 <= b_k1_FDX_s102 <= 0.2
 0 <= b_k1_FDX_s103 <= 0.2
 0 <= b_k1_FDX_s104 <= 0.2
 0 <= b_k1_FDX_s105 <= 0.2
 0 <= b_k1_FDX_s106 <= 0.2
 0 <= b_k1_FDX_s107 <= 0.2
 0 <= b_k1_FDX_s110 <= 0.2
 0 <= b_k1_FDX_s111 <= 0.2
 0 <= b_k1_FDX_s112 <= 0.2
 0 <= b_k1_FDX_s113 <= 0.2
 0 <= b_k1_FDX_s114 <= 0.2
 0 <= b_k1_FDX_s115 <= 0.2
 0 <= b_k1_FDX_s116 <= 0.2
 0 <= b_k1_FDX_s117 <= 0.2
 0 <= b_k1_IBM_s0 <= 0.2
 0 <= b_k1_IBM_s1 <= 0.2
 0 <= b_k1_IBM_s00 <= 0.2
 0 <= b_k1_IBM_s01 <= 0.2
 0 <= b_k1_IBM_s10 <= 0.2
 0 <= b_k1_IBM_s11 <= 0.2
 0 <= b_k1_IBM_s000 <= 0.2
 0 <= b_k1_IBM_s001 <= 0.2
 0 <= b_k1_IBM_s002 <= 0.2
 0 <= b_k1_IBM_s003 <= 0.2
 0 <= b_k1_IBM_s004 <= 0.2
 0 <= b_k1_IBM_s005 <= 0.2
 0 <= b_k1_IBM_s006 <= 0.2
 0 <= b_k1_IBM_s007 <= 0.2
 0 <= b_k1_IBM_s010 <= 0.2
 0 <= b_k1_IBM_s011 <= 0.2
 0 <= b_k1_IBM_s012 <= 0.2
 0 <= b_k1_IBM_s013 <= 0.2
 0 <= b_k1_IBM_s014 <= 0.2
 0 <= b_k1_IBM_s015 <= 0.2
 0 <= b_k1_IBM_s016 <= 0.2
 0 <= b_k1_IBM_s017 <= 0.2
 0 <= b_k1_IBM_s100 <= 0.2
 0 <= b_k1_IBM_s101 <= 0.2
 0 <= b_k1_IBM_s102 <= 0.2
 0 <= b_k1_IBM_s103 <= 0.2
 0 <= b_k1_IBM_s104 <= 0.2
 0 <= b_k1_IBM_s105 <= 0.2
 0 <= b_k1_IBM_s106 <= 0.2
 0 <= b_k1_IBM_s107 <= 0.2
 0 <= b_k1_IBM_s110 <= 0.2
 0 <= b_k1_IBM_s111 <= 0.2
 0 <= b_k1_IBM_s112 <= 0.2
 0 <= b_k1_IBM_s113 <= 0.2
 0 <= b_k1_IBM_s114 <= 0.2
 0 <= b_k1_IBM_s115 <= 0.2
 0 <= b_k1_IBM_s116 <= 0.2
 0 <= b_k1_IBM_s117 <= 0.2
 0 <= b_k1_INTC_s0 <= 0.2
 0 <= b_k1_INTC_s1 <= 0.2
 0 <= b_k1_INTC_s00 <= 0.2
 0 <= b_k1_INTC_s01 <= 0.2
 0 <= b_k1_INTC_s10 <= 0.2
 0 <= b_k1_INTC_s11 <= 0.2
 0 <= b_k1_INTC_s000 <= 0.2
 0 <= b_k1_INTC_s001 <= 0.2
 0 <= b_k1_INTC_s002 <= 0.2
 0 <= b_k1_INTC_s003 <= 0.2
 0 <= b_k1_INTC_s004 <= 0.2
 0 <= b_k1_INTC_s005 <= 0.2
 0 <= b_k1_INTC_s006 <= 0.2
 0 <= b_k1_INTC_s007 <= 0.2
 0 <= b_k1_INTC_s010 <= 0.2
 0 <= b_k1_INTC_s011 <= 0.2
 0 <= b_k1_INTC_s012 <= 0.2
 0 <= b_k1_INTC_s013 <= 0.2
 0 <= b_k1_INTC_s014 <= 0.2
 0 <= b_k1_INTC_s015 <= 0.2
 0 <= b_k1_INTC_s016 <= 0.2
 0 <= b_k1_INTC_s017 <= 0.2
 0 <= b_k1_INTC_s100 <= 0.2
 0 <= b_k1_INTC_s101 <= 0.2
 0 <= b_k1_INTC_s102 <= 0.2
 0 <= b_k1_INTC_s103 <= 0.2
 0 <= b_k1_INTC_s104 <= 0.2
 0 <= b_k1_INTC_s105 <= 0.2
 0 <= b_k1_INTC_s106 <= 0.2
 0 <= b_k1_INTC_s107 <= 0.2
 0 <= b_k1_INTC_s110 <= 0.2
 0 <= b_k1_INTC_s111 <= 0.2
 0 <= b_k1_INTC_s112 <= 0.2
 0 <= b_k1_INTC_s113 <= 0.2
 0 <= b_k1_INTC_s114 <= 0.2
 0 <= b_k1_INTC_s115 <= 0.2
 0 <= b_k1_INTC_s116 <= 0.2
 0 <= b_k1_INTC_s117 <= 0.2
 0 <= b_k1_KO_s0 <= 0.2
 0 <= b_k1_KO_s1 <= 0.2
 0 <= b_k1_KO_s00 <= 0.2
 0 <= b_k1_KO_s01 <= 0.2
 0 <= b_k1_KO_s10 <= 0.2
 0 <= b_k1_KO_s11 <= 0.2
 0 <= b_k1_KO_s000 <= 0.2
 0 <= b_k1_KO_s001 <= 0.2
 0 <= b_k1_KO_s002 <= 0.2
 0 <= b_k1_KO_s003 <= 0.2
 0 <= b_k1_KO_s004 <= 0.2
 0 <= b_k1_KO_s005 <= 0.2
 0 <= b_k1_KO_s006 <= 0.2
 0 <= b_k1_KO_s007 <= 0.2
 0 <= b_k1_KO_s010 <= 0.2
 0 <= b_k1_KO_s011 <= 0.2
 0 <= b_k1_KO_s012 <= 0.2
 0 <= b_k1_KO_s013 <= 0.2
 0 <= b_k1_KO_s014 <= 0.2
 0 <= b_k1_KO_s015 <= 0.2
 0 <= b_k1_KO_s016 <= 0.2
 0 <= b_k1_KO_s017 <= 0.2
 0 <= b_k1_KO_s100 <= 0.2
 0 <= b_k1_KO_s101 <= 0.2
 0 <= b_k1_KO_s102 <= 0.2
 0 <= b_k1_KO_s103 <= 0.2
 0 <= b_k1_KO_s104 <= 0.2
 0 <= b_k1_KO_s105 <= 0.2
 0 <= b_k1_KO_s106 <= 0.2
 0 <= b_k1_KO_s107 <= 0.2
 0 <= b_k1_KO_s110 <= 0.2
 0 <= b_k1_KO_s111 <= 0.2
 0 <= b_k1_KO_s112 <= 0.2
 0 <= b_k1_KO_s113 <= 0.2
 0 <= b_k1_KO_s114 <= 0.2
 0 <= b_k1_KO_s115 <= 0.2
 0 <= b_k1_KO_s116 <= 0.2
 0 <= b_k1_KO_s117 <= 0.2
 0 <= b_k1_MAT_s0 <= 0.2
 0 <= b_k1_MAT_s1 <= 0.2
 0 <= b_k1_MAT_s00 <= 0.2
 0 <= b_k1_MAT_s01 <= 0.2
 0 <= b_k1_MAT_s10 <= 0.2
 0 <= b_k1_MAT_s11 <= 0.2
 0 <= b_k1_MAT_s000 <= 0.2
 0 <= b_k1_MAT_s001 <= 0.2
 0 <= b_k1_MAT_s002 <= 0.2
 0 <= b_k1_MAT_s003 <= 0.2
 0 <= b_k1_MAT_s004 <= 0.2
 0 <= b_k1_MAT_s005 <= 0.2
 0 <= b_k1_MAT_s006 <= 0.2
 0 <= b_k1_MAT_s007 <= 0.2
 0 <= b_k1_MAT_s010 <= 0.2
 0 <= b_k1_MAT_s011 <= 0.2
 0 <= b_k1_MAT_s012 <= 0.2
 0 <= b_k1_MAT_s013 <= 0.2
 0 <= b_k1_MAT_s014 <= 0.2
 0 <= b_k1_MAT_s015 <= 0.2
 0 <= b_k1_MAT_s016 <= 0.2
 0 <= b_k1_MAT_s017 <= 0.2
 0 <= b_k1_MAT_s100 <= 0.2
 0 <= b_k1_MAT_s101 <= 0.2
 0 <= b_k1_MAT_s102 <= 0.2
 0 <= b_k1_MAT_s103 <= 0.2
 0 <= b_k1_MAT_s104 <= 0.2
 0 <= b_k1_MAT_s105 <= 0.2
 0 <= b_k1_MAT_s106 <= 0.2
 0 <= b_k1_MAT_s107 <= 0.2
 0 <= b_k1_MAT_s110 <= 0.2
 0 <= b_k1_MAT_s111 <= 0.2
 0 <= b_k1_MAT_s112 <= 0.2
 0 <= b_k1_MAT_s113 <= 0.2
 0 <= b_k1_MAT_s114 <= 0.2
 0 <= b_k1_MAT_s115 <= 0.2
 0 <= b_k1_MAT_s116 <= 0.2
 0 <= b_k1_MAT_s117 <= 0.2
 0 <= b_k1_MSFT_s0 <= 0.2
 0 <= b_k1_MSFT_s1 <= 0.2
 0 <= b_k1_MSFT_s00 <= 0.2
 0 <= b_k1_MSFT_s01 <= 0.2
 0 <= b_k1_MSFT_s10 <= 0.2
 0 <= b_k1_MSFT_s11 <= 0.2
 0 <= b_k1_MSFT_s000 <= 0.2
 0 <= b_k1_MSFT_s001 <= 0.2
 0 <= b_k1_MSFT_s002 <= 0.2
 0 <= b_k1_MSFT_s003 <= 0.2
 0 <= b_k1_MSFT_s004 <= 0.2
 0 <= b_k1_MSFT_s005 <= 0.2
 0 <= b_k1_MSFT_s006 <= 0.2
 0 <= b_k1_MSFT_s007 <= 0.2
 0 <= b_k1_MSFT_s010 <= 0.2
 0 <= b_k1_MSFT_s011 <= 0.2
 0 <= b_k1_MSFT_s012 <= 0.2
 0 <= b_k1_MSFT_s013 <= 0.2
 0 <= b_k1_MSFT_s014 <= 0.2
 0 <= b_k1_MSFT_s015 <= 0.2
 0 <= b_k1_MSFT_s016 <= 0.2
 0 <= b_k1_MSFT_s017 <= 0.2
 0 <= b_k1_MSFT_s100 <= 0.2
 0 <= b_k1_MSFT_s101 <= 0.2
 0 <= b_k1_MSFT_s102 <= 0.2
 0 <= b_k1_MSFT_s103 <= 0.2
 0 <= b_k1_MSFT_s104 <= 0.2
 0 <= b_k1_MSFT_s105 <= 0.2
 0 <= b_k1_MSFT_s106 <= 0.2
 0 <= b_k1_MSFT_s107 <= 0.2
 0 <= b_k1_MSFT_s110 <= 0.2
 0 <= b_k1_MSFT_s111 <= 0.2
 0 <= b_k1_MSFT_s112 <= 0.2
 0 <= b_k1_MSFT_s113 <= 0.2
 0 <= b_k1_MSFT_s114 <= 0.2
 0 <= b_k1_MSFT_s115 <= 0.2
 0 <= b_k1_MSFT_s116 <= 0.2
 0 <= b_k1_MSFT_s117 <= 0.2
 0 <= b_k1_PG_s0 <= 0.2
 0 <= b_k1_PG_s1 <= 0.2
 0 <= b_k1_PG_s00 <= 0.2
 0 <= b_k1_PG_s01 <= 0.2
 0 <= b_k1_PG_s10 <= 0.2
 0 <= b_k1_PG_s11 <= 0.2
 0 <= b_k1_PG_s000 <= 0.2
 0 <= b_k1_PG_s001 <= 0.2
 0 <= b_k1_PG_s002 <= 0.2
 0 <= b_k1_PG_s003 <= 0.2
 0 <= b_k1_PG_s004 <= 0.2
 0 <= b_k1_PG_s005 <= 0.2
 0 <= b_k1_PG_s006 <= 0.2
 0 <= b_k1_PG_s007 <= 0.2
 0 <= b_k1_PG_s010 <= 0.2
 0 <= b_k1_PG_s011 <= 0.2
 0 <= b_k1_PG_s012 <= 0.2
 0 <= b_k1_PG_s013 <= 0.2
 0 <= b_k1_PG_s014 <= 0.2
 0 <= b_k1_PG_s015 <= 0.2
 0 <= b_k1_PG_s016 <= 0.2
 0 <= b_k1_PG_s017 <= 0.2
 0 <= b_k1_PG_s100 <= 0.2
 0 <= b_k1_PG_s101 <= 0.2
 0 <= b_k1_PG_s102 <= 0.2
 0 <= b_k1_PG_s103 <= 0.2
 0 <= b_k1_PG_s104 <= 0.2
 0 <= b_k1_PG_s105 <= 0.2
 0 <= b_k1_PG_s106 <= 0.2
 0 <= b_k1_PG_s107 <= 0.2
 0 <= b_k1_PG_s110 <= 0.2
 0 <= b_k1_PG_s111 <= 0.2
 0 <= b_k1_PG_s112 <= 0.2
 0 <= b_k1_PG_s113 <= 0.2
 0 <= b_k1_PG_s114 <= 0.2
 0 <= b_k1_PG_s115 <= 0.2
 0 <= b_k1_PG_s116 <= 0.2
 0 <= b_k1_PG_s117 <= 0.2
 0 <= s_0_ADM <= 0.2
 0 <= s_0_AXP <= 0.2
 0 <= s_0_DIS <= 0.2
 0 <= s_0_FDX <= 0.2
 0 <= s_0_IBM <= 0.2
 0 <= s_0_INTC <= 0.2
 0 <= s_0_KO <= 0.2
 0 <= s_0_MAT <= 0.2
 0 <= s_0_MSFT <= 0.2
 0 <= s_0_PG <= 0.2
 0 <= s_k1_ADM_s0 <= 0.2
 0 <= s_k1_ADM_s1 <= 0.2
 0 <= s_k1_ADM_s00 <= 0.2
 0 <= s_k1_ADM_s01 <= 0.2
 0 <= s_k1_ADM_s10 <= 0.2
 0 <= s_k1_ADM_s11 <= 0.2
 0 <= s_k1_ADM_s000 <= 0.2
 0 <= s_k1_ADM_s001 <= 0.2
 0 <= s_k1_ADM_s002 <= 0.2
 0 <= s_k1_ADM_s003 <= 0.2
 0 <= s_k1_ADM_s004 <= 0.2
 0 <= s_k1_ADM_s005 <= 0.2
 0 <= s_k1_ADM_s006 <= 0.2
 0 <= s_k1_ADM_s007 <= 0.2
 0 <= s_k1_ADM_s010 <= 0.2
 0 <= s_k1_ADM_s011 <= 0.2
 0 <= s_k1_ADM_s012 <= 0.2
 0 <= s_k1_ADM_s013 <= 0.2
 0 <= s_k1_ADM_s014 <= 0.2
 0 <= s_k1_ADM_s015 <= 0.2
 0 <= s_k1_ADM_s016 <= 0.2
 0 <= s_k1_ADM_s017 <= 0.2
 0 <= s_k1_ADM_s100 <= 0.2
 0 <= s_k1_ADM_s101 <= 0.2
 0 <= s_k1_ADM_s102 <= 0.2
 0 <= s_k1_ADM_s103 <= 0.2
 0 <= s_k1_ADM_s104 <= 0.2
 0 <= s_k1_ADM_s105 <= 0.2
 0 <= s_k1_ADM_s106 <= 0.2
 0 <= s_k1_ADM_s107 <= 0.2
 0 <= s_k1_ADM_s110 <= 0.2
 0 <= s_k1_ADM_s111 <= 0.2
 0 <= s_k1_ADM_s112 <= 0.2
 0 <= s_k1_ADM_s113 <= 0.2
 0 <= s_k1_ADM_s114 <= 0.2
 0 <= s_k1_ADM_s115 <= 0.2
 0 <= s_k1_ADM_s116 <= 0.2
 0 <= s_k1_ADM_s117 <= 0.2
 0 <= s_k1_AXP_s0 <= 0.2
 0 <= s_k1_AXP_s1 <= 0.2
 0 <= s_k1_AXP_s00 <= 0.2
 0 <= s_k1_AXP_s01 <= 0.2
 0 <= s_k1_AXP_s10 <= 0.2
 0 <= s_k1_AXP_s11 <= 0.2
 0 <= s_k1_AXP_s000 <= 0.2
 0 <= s_k1_AXP_s001 <= 0.2
 0 <= s_k1_AXP_s002 <= 0.2
 0 <= s_k1_AXP_s003 <= 0.2
 0 <= s_k1_AXP_s004 <= 0.2
 0 <= s_k1_AXP_s005 <= 0.2
 0 <= s_k1_AXP_s006 <= 0.2
 0 <= s_k1_AXP_s007 <= 0.2
 0 <= s_k1_AXP_s010 <= 0.2
 0 <= s_k1_AXP_s011 <= 0.2
 0 <= s_k1_AXP_s012 <= 0.2
 0 <= s_k1_AXP_s013 <= 0.2
 0 <= s_k1_AXP_s014 <= 0.2
 0 <= s_k1_AXP_s015 <= 0.2
 0 <= s_k1_AXP_s016 <= 0.2
 0 <= s_k1_AXP_s017 <= 0.2
 0 <= s_k1_AXP_s100 <= 0.2
 0 <= s_k1_AXP_s101 <= 0.2
 0 <= s_k1_AXP_s102 <= 0.2
 0 <= s_k1_AXP_s103 <= 0.2
 0 <= s_k1_AXP_s104 <= 0.2
 0 <= s_k1_AXP_s105 <= 0.2
 0 <= s_k1_AXP_s106 <= 0.2
 0 <= s_k1_AXP_s107 <= 0.2
 0 <= s_k1_AXP_s110 <= 0.2
 0 <= s_k1_AXP_s111 <= 0.2
 0 <= s_k1_AXP_s112 <= 0.2
 0 <= s_k1_AXP_s113 <= 0.2
 0 <= s_k1_AXP_s114 <= 0.2
 0 <= s_k1_AXP_s115 <= 0.2
 0 <= s_k1_AXP_s116 <= 0.2
 0 <= s_k1_AXP_s117 <= 0.2
 0 <= s_k1_DIS_s0 <= 0.2
 0 <= s_k1_DIS_s1 <= 0.2
 0 <= s_k1_DIS_s00 <= 0.2
 0 <= s_k1_DIS_s01 <= 0.2
 0 <= s_k1_DIS_s10 <= 0.2
 0 <= s_k1_DIS_s11 <= 0.2
 0 <= s_k1_DIS_s000 <= 0.2
 0 <= s_k1_DIS_s001 <= 0.2
 0 <= s_k1_DIS_s002 <= 0.2
 0 <= s_k1_DIS_s003 <= 0.2
 0 <= s_k1_DIS_s004 <= 0.2
 0 <= s_k1_DIS_s005 <= 0.2
 0 <= s_k1_DIS_s006 <= 0.2
 0 <= s_k1_DIS_s007 <= 0.2
 0 <= s_k1_DIS_s010 <= 0.2
 0 <= s_k1_DIS_s011 <= 0.2
 0 <= s_k1_DIS_s012 <= 0.2
 0 <= s_k1_DIS_s013 <= 0.2
 0 <= s_k1_DIS_s014 <= 0.2
 0 <= s_k1_DIS_s015 <= 0.2
 0 <= s_k1_DIS_s016 <= 0.2
 0 <= s_k1_DIS_s017 <= 0.2
 0 <= s_k1_DIS_s100 <= 0.2
 0 <= s_k1_DIS_s101 <= 0.2
 0 <= s_k1_DIS_s102 <= 0.2
 0 <= s_k1_DIS_s103 <= 0.2
 0 <= s_k1_DIS_s104 <= 0.2
 0 <= s_k1_DIS_s105 <= 0.2
 0 <= s_k1_DIS_s106 <= 0.2
 0 <= s_k1_DIS_s107 <= 0.2
 0 <= s_k1_DIS_s110 <= 0.2
 0 <= s_k1_DIS_s111 <= 0.2
 0 <= s_k1_DIS_s112 <= 0.2
 0 <= s_k1_DIS_s113 <= 0.2
 0 <= s_k1_DIS_s114 <= 0.2
 0 <= s_k1_DIS_s115 <= 0.2
 0 <= s_k1_DIS_s116 <= 0.2
 0 <= s_k1_DIS_s117 <= 0.2
 0 <= s_k1_FDX_s0 <= 0.2
 0 <= s_k1_FDX_s1 <= 0.2
 0 <= s_k1_FDX_s00 <= 0.2
 0 <= s_k1_FDX_s01 <= 0.2
 0 <= s_k1_FDX_s10 <= 0.2
 0 <= s_k1_FDX_s11 <= 0.2
 0 <= s_k1_FDX_s000 <= 0.2
 0 <= s_k1_FDX_s001 <= 0.2
 0 <= s_k1_FDX_s002 <= 0.2
 0 <= s_k1_FDX_s003 <= 0.2
 0 <= s_k1_FDX_s004 <= 0.2
 0 <= s_k1_FDX_s005 <= 0.2
 0 <= s_k1_FDX_s006 <= 0.2
 0 <= s_k1_FDX_s007 <= 0.2
 0 <= s_k1_FDX_s010 <= 0.2
 0 <= s_k1_FDX_s011 <= 0.2
 0 <= s_k1_FDX_s012 <= 0.2
 0 <= s_k1_FDX_s013 <= 0.2
 0 <= s_k1_FDX_s014 <= 0.2
 0 <= s_k1_FDX_s015 <= 0.2
 0 <= s_k1_FDX_s016 <= 0.2
 0 <= s_k1_FDX_s017 <= 0.2
 0 <= s_k1_FDX_s100 <= 0.2
 0 <= s_k1_FDX_s101 <= 0.2
 0 <= s_k1_FDX_s102 <= 0.2
 0 <= s_k1_FDX_s103 <= 0.2
 0 <= s_k1_FDX_s104 <= 0.2
 0 <= s_k1_FDX_s105 <= 0.2
 0 <= s_k1_FDX_s106 <= 0.2
 0 <= s_k1_FDX_s107 <= 0.2
 0 <= s_k1_FDX_s110 <= 0.2
 0 <= s_k1_FDX_s111 <= 0.2
 0 <= s_k1_FDX_s112 <= 0.2
 0 <= s_k1_FDX_s113 <= 0.2
 0 <= s_k1_FDX_s114 <= 0.2
 0 <= s_k1_FDX_s115 <= 0.2
 0 <= s_k1_FDX_s116 <= 0.2
 0 <= s_k1_FDX_s117 <= 0.2
 0 <= s_k1_IBM_s0 <= 0.2
 0 <= s_k1_IBM_s1 <= 0.2
 0 <= s_k1_IBM_s00 <= 0.2
 0 <= s_k1_IBM_s01 <= 0.2
 0 <= s_k1_IBM_s10 <= 0.2
 0 <= s_k1_IBM_s11 <= 0.2
 0 <= s_k1_IBM_s000 <= 0.2
 0 <= s_k1_IBM_s001 <= 0.2
 0 <= s_k1_IBM_s002 <= 0.2
 0 <= s_k1_IBM_s003 <= 0.2
 0 <= s_k1_IBM_s004 <= 0.2
 0 <= s_k1_IBM_s005 <= 0.2
 0 <= s_k1_IBM_s006 <= 0.2
 0 <= s_k1_IBM_s007 <= 0.2
 0 <= s_k1_IBM_s010 <= 0.2
 0 <= s_k1_IBM_s011 <= 0.2
 0 <= s_k1_IBM_s012 <= 0.2
 0 <= s_k1_IBM_s013 <= 0.2
 0 <= s_k1_IBM_s014 <= 0.2
 0 <= s_k1_IBM_s015 <= 0.2
 0 <= s_k1_IBM_s016 <= 0.2
 0 <= s_k1_IBM_s017 <= 0.2
 0 <= s_k1_IBM_s100 <= 0.2
 0 <= s_k1_IBM_s101 <= 0.2
 0 <= s_k1_IBM_s102 <= 0.2
 0 <= s_k1_IBM_s103 <= 0.2
 0 <= s_k1_IBM_s104 <= 0.2
 0 <= s_k1_IBM_s105 <= 0.2
 0 <= s_k1_IBM_s106 <= 0.2
 0 <= s_k1_IBM_s107 <= 0.2
 0 <= s_k1_IBM_s110 <= 0.2
 0 <= s_k1_IBM_s111 <= 0.2
 0 <= s_k1_IBM_s112 <= 0.2
 0 <= s_k1_IBM_s113 <= 0.2
 0 <= s_k1_IBM_s114 <= 0.2
 0 <= s_k1_IBM_s115 <= 0.2
 0 <= s_k1_IBM_s116 <= 0.2
 0 <= s_k1_IBM_s117 <= 0.2
 0 <= s_k1_INTC_s0 <= 0.2
 0 <= s_k1_INTC_s1 <= 0.2
 0 <= s_k1_INTC_s00 <= 0.2
 0 <= s_k1_INTC_s01 <= 0.2
 0 <= s_k1_INTC_s10 <= 0.2
 0 <= s_k1_INTC_s11 <= 0.2
 0 <= s_k1_INTC_s000 <= 0.2
 0 <= s_k1_INTC_s001 <= 0.2
 0 <= s_k1_INTC_s002 <= 0.2
 0 <= s_k1_INTC_s003 <= 0.2
 0 <= s_k1_INTC_s004 <= 0.2
 0 <= s_k1_INTC_s005 <= 0.2
 0 <= s_k1_INTC_s006 <= 0.2
 0 <= s_k1_INTC_s007 <= 0.2
 0 <= s_k1_INTC_s010 <= 0.2
 0 <= s_k1_INTC_s011 <= 0.2
 0 <= s_k1_INTC_s012 <= 0.2
 0 <= s_k1_INTC_s013 <= 0.2
 0 <= s_k1_INTC_s014 <= 0.2
 0 <= s_k1_INTC_s015 <= 0.2
 0 <= s_k1_INTC_s016 <= 0.2
 0 <= s_k1_INTC_s017 <= 0.2
 0 <= s_k1_INTC_s100 <= 0.2
 0 <= s_k1_INTC_s101 <= 0.2
 0 <= s_k1_INTC_s102 <= 0.2
 0 <= s_k1_INTC_s103 <= 0.2
 0 <= s_k1_INTC_s104 <= 0.2
 0 <= s_k1_INTC_s105 <= 0.2
 0 <= s_k1_INTC_s106 <= 0.2
 0 <= s_k1_INTC_s107 <= 0.2
 0 <= s_k1_INTC_s110 <= 0.2
 0 <= s_k1_INTC_s111 <= 0.2
 0 <= s_k1_INTC_s112 <= 0.2
 0 <= s_k1_INTC_s113 <= 0.2
 0 <= s_k1_INTC_s114 <= 0.2
 0 <= s_k1_INTC_s115 <= 0.2
 0 <= s_k1_INTC_s116 <= 0.2
 0 <= s_k1_INTC_s117 <= 0.2
 0 <= s_k1_KO_s0 <= 0.2
 0 <= s_k1_KO_s1 <= 0.2
 0 <= s_k1_KO_s00 <= 0.2
 0 <= s_k1_KO_s01 <= 0.2
 0 <= s_k1_KO_s10 <= 0.2
 0 <= s_k1_KO_s11 <= 0.2
 0 <= s_k1_KO_s000 <= 0.2
 0 <= s_k1_KO_s001 <= 0.2
 0 <= s_k1_KO_s002 <= 0.2
 0 <= s_k1_KO_s003 <= 0.2
 0 <= s_k1_KO_s004 <= 0.2
 0 <= s_k1_KO_s005 <= 0.2
 0 <= s_k1_KO_s006 <= 0.2
 0 <= s_k1_KO_s007 <= 0.2
 0 <= s_k1_KO_s010 <= 0.2
 0 <= s_k1_KO_s011 <= 0.2
 0 <= s_k1_KO_s012 <= 0.2
 0 <= s_k1_KO_s013 <= 0.2
 0 <= s_k1_KO_s014 <= 0.2
 0 <= s_k1_KO_s015 <= 0.2
 0 <= s_k1_KO_s016 <= 0.2
 0 <= s_k1_KO_s017 <= 0.2
 0 <= s_k1_KO_s100 <= 0.2
 0 <= s_k1_KO_s101 <= 0.2
 0 <= s_k1_KO_s102 <= 0.2
 0 <= s_k1_KO_s103 <= 0.2
 0 <= s_k1_KO_s104 <= 0.2
 0 <= s_k1_KO_s105 <= 0.2
 0 <= s_k1_KO_s106 <= 0.2
 0 <= s_k1_KO_s107 <= 0.2
 0 <= s_k1_KO_s110 <= 0.2
 0 <= s_k1_KO_s111 <= 0.2
 0 <= s_k1_KO_s112 <= 0.2
 0 <= s_k1_KO_s113 <= 0.2
 0 <= s_k1_KO_s114 <= 0.2
 0 <= s_k1_KO_s115 <= 0.2
 0 <= s_k1_KO_s116 <= 0.2
 0 <= s_k1_KO_s117 <= 0.2
 0 <= s_k1_MAT_s0 <= 0.2
 0 <= s_k1_MAT_s1 <= 0.2
 0 <= s_k1_MAT_s00 <= 0.2
 0 <= s_k1_MAT_s01 <= 0.2
 0 <= s_k1_MAT_s10 <= 0.2
 0 <= s_k1_MAT_s11 <= 0.2
 0 <= s_k1_MAT_s000 <= 0.2
 0 <= s_k1_MAT_s001 <= 0.2
 0 <= s_k1_MAT_s002 <= 0.2
 0 <= s_k1_MAT_s003 <= 0.2
 0 <= s_k1_MAT_s004 <= 0.2
 0 <= s_k1_MAT_s005 <= 0.2
 0 <= s_k1_MAT_s006 <= 0.2
 0 <= s_k1_MAT_s007 <= 0.2
 0 <= s_k1_MAT_s010 <= 0.2
 0 <= s_k1_MAT_s011 <= 0.2
 0 <= s_k1_MAT_s012 <= 0.2
 0 <= s_k1_MAT_s013 <= 0.2
 0 <= s_k1_MAT_s014 <= 0.2
 0 <= s_k1_MAT_s015 <= 0.2
 0 <= s_k1_MAT_s016 <= 0.2
 0 <= s_k1_MAT_s017 <= 0.2
 0 <= s_k1_MAT_s100 <= 0.2
 0 <= s_k1_MAT_s101 <= 0.2
 0 <= s_k1_MAT_s102 <= 0.2
 0 <= s_k1_MAT_s103 <= 0.2
 0 <= s_k1_MAT_s104 <= 0.2
 0 <= s_k1_MAT_s105 <= 0.2
 0 <= s_k1_MAT_s106 <= 0.2
 0 <= s_k1_MAT_s107 <= 0.2
 0 <= s_k1_MAT_s110 <= 0.2
 0 <= s_k1_MAT_s111 <= 0.2
 0 <= s_k1_MAT_s112 <= 0.2
 0 <= s_k1_MAT_s113 <= 0.2
 0 <= s_k1_MAT_s114 <= 0.2
 0 <= s_k1_MAT_s115 <= 0.2
 0 <= s_k1_MAT_s116 <= 0.2
 0 <= s_k1_MAT_s117 <= 0.2
 0 <= s_k1_MSFT_s0 <= 0.2
 0 <= s_k1_MSFT_s1 <= 0.2
 0 <= s_k1_MSFT_s00 <= 0.2
 0 <= s_k1_MSFT_s01 <= 0.2
 0 <= s_k1_MSFT_s10 <= 0.2
 0 <= s_k1_MSFT_s11 <= 0.2
 0 <= s_k1_MSFT_s000 <= 0.2
 0 <= s_k1_MSFT_s001 <= 0.2
 0 <= s_k1_MSFT_s002 <= 0.2
 0 <= s_k1_MSFT_s003 <= 0.2
 0 <= s_k1_MSFT_s004 <= 0.2
 0 <= s_k1_MSFT_s005 <= 0.2
 0 <= s_k1_MSFT_s006 <= 0.2
 0 <= s_k1_MSFT_s007 <= 0.2
 0 <= s_k1_MSFT_s010 <= 0.2
 0 <= s_k1_MSFT_s011 <= 0.2
 0 <= s_k1_MSFT_s012 <= 0.2
 0 <= s_k1_MSFT_s013 <= 0.2
 0 <= s_k1_MSFT_s014 <= 0.2
 0 <= s_k1_MSFT_s015 <= 0.2
 0 <= s_k1_MSFT_s016 <= 0.2
 0 <= s_k1_MSFT_s017 <= 0.2
 0 <= s_k1_MSFT_s100 <= 0.2
 0 <= s_k1_MSFT_s101 <= 0.2
 0 <= s_k1_MSFT_s102 <= 0.2
 0 <= s_k1_MSFT_s103 <= 0.2
 0 <= s_k1_MSFT_s104 <= 0.2
 0 <= s_k1_MSFT_s105 <= 0.2
 0 <= s_k1_MSFT_s106 <= 0.2
 0 <= s_k1_MSFT_s107 <= 0.2
 0 <= s_k1_MSFT_s110 <= 0.2
 0 <= s_k1_MSFT_s111 <= 0.2
 0 <= s_k1_MSFT_s112 <= 0.2
 0 <= s_k1_MSFT_s113 <= 0.2
 0 <= s_k1_MSFT_s114 <= 0.2
 0 <= s_k1_MSFT_s115 <= 0.2
 0 <= s_k1_MSFT_s116 <= 0.2
 0 <= s_k1_MSFT_s117 <= 0.2
 0 <= s_k1_PG_s0 <= 0.2
 0 <= s_k1_PG_s1 <= 0.2
 0 <= s_k1_PG_s00 <= 0.2
 0 <= s_k1_PG_s01 <= 0.2
 0 <= s_k1_PG_s10 <= 0.2
 0 <= s_k1_PG_s11 <= 0.2
 0 <= s_k1_PG_s000 <= 0.2
 0 <= s_k1_PG_s001 <= 0.2
 0 <= s_k1_PG_s002 <= 0.2
 0 <= s_k1_PG_s003 <= 0.2
 0 <= s_k1_PG_s004 <= 0.2
 0 <= s_k1_PG_s005 <= 0.2
 0 <= s_k1_PG_s006 <= 0.2
 0 <= s_k1_PG_s007 <= 0.2
 0 <= s_k1_PG_s010 <= 0.2
 0 <= s_k1_PG_s011 <= 0.2
 0 <= s_k1_PG_s012 <= 0.2
 0 <= s_k1_PG_s013 <= 0.2
 0 <= s_k1_PG_s014 <= 0.2
 0 <= s_k1_PG_s015 <= 0.2
 0 <= s_k1_PG_s016 <= 0.2
 0 <= s_k1_PG_s017 <= 0.2
 0 <= s_k1_PG_s100 <= 0.2
 0 <= s_k1_PG_s101 <= 0.2
 0 <= s_k1_PG_s102 <= 0.2
 0 <= s_k1_PG_s103 <= 0.2
 0 <= s_k1_PG_s104 <= 0.2
 0 <= s_k1_PG_s105 <= 0.2
 0 <= s_k1_PG_s106 <= 0.2
 0 <= s_k1_PG_s107 <= 0.2
 0 <= s_k1_PG_s110 <= 0.2
 0 <= s_k1_PG_s111 <= 0.2
 0 <= s_k1_PG_s112 <= 0.2
 0 <= s_k1_PG_s113 <= 0.2
 0 <= s_k1_PG_s114 <= 0.2
 0 <= s_k1_PG_s115 <= 0.2
 0 <= s_k1_PG_s116 <= 0.2
 0 <= s_k1_PG_s117 <= 0.2
End
