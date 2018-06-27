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
 w_k1_ADM_s0_return_constraint:  1.09176 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.882782 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.963532 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.10738 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.01457 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.15911 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.992578 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.39126 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.31888 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.991826 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.50558 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.802622 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.826017 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.15182 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.883786 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.86511 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.41692 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.830567 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.05962 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.26142 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.2866 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.968282 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.079 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.918286 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.956743 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.777817 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.14783 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.53092 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.786066 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.653012 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.937162 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.14483 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.74539 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.34128 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.984865 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.22033 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.816575 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.09344 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.19234 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.887724 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.859764 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.10838 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.93784 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.15815 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.964853 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.70626 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.24598 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.00302 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.57816 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.752258 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.72776 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.20979 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.921366 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.816905 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.67557 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.975778 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.07787 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 2.1588 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.36449 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.964104 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.27353 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.98466 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.991324 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.624223 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.23108 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.74524 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.570792 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.675646 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.863061 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.32318 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.47258 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.56504 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.985566 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.17486 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.651549 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.824282 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.26227 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.811488 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.827183 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.21194 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.01002 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.20513 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.10862 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.95416 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.61835 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.813664 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.56962 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.565623 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.806031 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.21084 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.69645 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.792046 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.6328 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.635749 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.05821 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.9066 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.29805 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.934039 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.2643 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.849816 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.796086 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.612199 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.996532 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.87282 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.733907 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.557603 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.819686 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.1198 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.93345 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.58235 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.26304 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.08256 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.59733 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.1423 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.15331 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.881693 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.915115 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.08316 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.956011 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.2588 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.18503 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.51293 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.36796 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.915763 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.58852 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.67403 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.876963 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.12947 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.814365 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.846656 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.54628 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.708838 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.15911 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.4873 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.18545 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.891859 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.20561 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.02315 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.866933 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.846909 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.06551 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.496 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.725892 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.686854 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.898034 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.11439 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.45012 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.23305 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.07192 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.934528 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.701215 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.974783 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.13962 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.944134 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.734851 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.2281 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.08101 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.768358 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.814886 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.57562 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.941761 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.13924 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.91677 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.92725 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.847866 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.40175 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.923488 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.763926 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.954268 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.16866 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.789059 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.73556 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.26994 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.48601 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.27159 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.629335 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.15462 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.447916 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.954036 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.48869 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.937363 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.608668 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.817224 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.02282 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.2989 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.60448 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.12727 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.70891 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.699648 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.925413 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.07655 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.931702 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.992333 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.2454 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.05384 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.15659 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.00672 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.29241 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.26372 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.953685 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.25843 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.75514 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.951088 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.21839 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.780775 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.785085 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.10419 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.641959 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.900667 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.09277 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.983471 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.769273 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.04567 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.837146 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.844807 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.690756 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.04101 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.29463 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.657722 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.715987 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.896651 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.06565 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.26038 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.00344 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.09776 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.883985 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.661473 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.876337 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.1539 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.876637 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.799142 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.1629 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.00988 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.96089 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.947573 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.80018 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.23831 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.08856 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.17108 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.821307 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.922066 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.3623 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.831316 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.91295 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.04866 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.938635 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.91928 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.49537 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.27452 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.14877 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.31755 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.811763 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.939914 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.585142 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.947842 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.38556 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.870272 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.600892 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.83728 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.14983 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.59958 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.46784 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.19885 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.40128 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.660318 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.00551 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.15482 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.862673 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.767702 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.31149 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.22899 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.853185 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.923411 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.7222 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.946755 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.36033 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.03911 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.810083 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.916456 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.28943 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.834499 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.701896 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.959108 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.30102 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.690074 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.95988 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.48575 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.30105 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.08183 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.692333 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.984589 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.435301 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.871322 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.4995 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.722871 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.510257 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.03902 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.09341 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.65825 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 2.23123 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.11048 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.82107 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.69677 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.991417 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.15388 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.897808 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.864831 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.03771 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.07666 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.09535 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.18358 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.41385 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.39832 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.921219 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.47761 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.739295 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.998357 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.13904 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.803829 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.995528 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.34106 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.768136 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.14846 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.35906 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.20283 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.942117 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.07862 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.893464 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.7333 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.777778 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.945784 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.26974 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.702153 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.648252 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.963129 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.13954 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.43754 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.2631 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.17633 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.932357 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.749295 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.09707 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.12597 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.901512 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.849943 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.13416 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.977862 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.0382 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.04755 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.32386 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.19681 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.04157 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.19228 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.783196 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.905059 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.22475 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.778757 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.826973 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.01838 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.865905 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.989114 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.241 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.04263 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.990859 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.17891 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.922194 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.00102 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.862247 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.095 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.30657 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.861772 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.826794 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.885845 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.07012 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.30462 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.29133 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.11003 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.20712 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.761871 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.07466 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.358001556081713 z_k1_s0000
                                 - 0.673000740570698 z_k1_s0001
                                 - 0.450001362840383 z_k1_s0002
                                 - 0.570999723860497 z_k1_s0003
                                 - 0.457001502006741 z_k1_s0004
                                 - 0.413999130414012 z_k1_s0005
                                 - 0.288999931513428 z_k1_s0006
                                 - 0.326999928631108 z_k1_s0007
                                 - 0.179999860059406 z_k1_s0010
                                 - 0.0489999070884649 z_k1_s0011
                                 - 0.0969996604712442 z_k1_s0012
                                 - 0.101000275293987 z_k1_s0013
                                 - 0.0649999444825699 z_k1_s0014
                                 - 0.148999580177346 z_k1_s0015
                                 - 0.193999320942488 z_k1_s0016
                                 - 0.0619999318649324 z_k1_s0017
                                 - 0.0939999486474684 z_k1_s0020
                                 - 0.270999546487924 z_k1_s0021
                                 - 0.089999984773107 z_k1_s0022
                                 - 0.141999714539799 z_k1_s0023
                                 - 0.0650000111583544 z_k1_s0024
                                 - 0.151999624225703 z_k1_s0025
                                 - 0.104000057733366 z_k1_s0026
                                 - 0.0789999844186162 z_k1_s0027
                                 - 0.318000897272207 z_k1_s0030
                                 - 0.556999760441418 z_k1_s0031
                                 - 0.622000108500273 z_k1_s0032
                                 - 0.324998660292948 z_k1_s0033
                                 - 0.47099996331768 z_k1_s0034
                                 - 0.139000152002528 z_k1_s0035
                                 - 0.417999300438757 z_k1_s0036
                                 - 0.23000002328466 z_k1_s0037
                                 - 0.233000048428578 z_k1_s0040
                                 - 0.0820000217199285 z_k1_s0041
                                 - 0.0969999991174782 z_k1_s0042
                                 - 0.225999368896434 z_k1_s0043
                                 - 0.122000052246722 z_k1_s0044
                                 - 0.297000424151425 z_k1_s0045
                                 - 0.199000444700776 z_k1_s0046
                                 - 0.10599995831601 z_k1_s0047
                                 - 0.288999798255963 z_k1_s0050
                                 - 0.329000937848817 z_k1_s0051
                                 - 0.732000151087864 z_k1_s0052
                                 - 0.206999906668412 z_k1_s0053
                                 - 0.367999498744145 z_k1_s0054
                                 - 0.171999890373782 z_k1_s0055
                                 - 0.714999772390806 z_k1_s0056
                                 - 0.205999742368713 z_k1_s0057
                                 - 0.317999991111826 z_k1_s0060
                                 - 0.930001303345296 z_k1_s0061
                                 - 0.345999649266483 z_k1_s0062
                                 - 0.517997732359291 z_k1_s0063
                                 - 0.668997673085372 z_k1_s0064
                                 - 0.722001158769076 z_k1_s0065
                                 - 0.300999847910945 z_k1_s0066
                                 - 0.465997696215236 z_k1_s0067
                                 - 0.421001088818818 z_k1_s0070
                                 - 0.340999747422875 z_k1_s0071
                                 - 0.43900002224909 z_k1_s0072
                                 - 0.369999736936834 z_k1_s0073
                                 - 0.856000931528208 z_k1_s0074
                                 - 0.421001088818818 z_k1_s0075
                                 - 0.673001152829641 z_k1_s0076
                                 - 0.227999736571975 z_k1_s0077
                                 - 0.304999782779823 z_k1_s0100
                                 - 0.377999510559255 z_k1_s0101
                                 - 0.587998415915613 z_k1_s0102
                                 - 0.912996407003301 z_k1_s0103
                                 - 0.603997617240966 z_k1_s0104
                                 - 0.810998192934071 z_k1_s0105
                                 - 1.30600025946004 z_k1_s0106
                                 - 0.383999343281066 z_k1_s0107
                                 - 0.674997553149828 z_k1_s0110
                                 - 0.554999417853336 z_k1_s0111
                                 - 0.722997797066217 z_k1_s0112
                                 - 0.585000188924699 z_k1_s0113
                                 - 0.987996359881657 z_k1_s0114
                                 - 0.243999507762174 z_k1_s0115
                                 - 0.89999838719609 z_k1_s0116
                                 - 0.27899925258134 z_k1_s0117
                                 - 0.644000089535947 z_k1_s0120
                                 - 0.223000166607754 z_k1_s0121
                                 - 0.250000184269677 z_k1_s0122
                                 - 0.735001258297488 z_k1_s0123
                                 - 0.183000181908646 z_k1_s0124
                                 - 0.28599927481808 z_k1_s0125
                                 - 0.187000236358606 z_k1_s0126
                                 - 0.291001092257084 z_k1_s0127
                                 - 0.0890001028038909 z_k1_s0130
                                 - 0.398000583080375 z_k1_s0131
                                 - 0.284000058504434 z_k1_s0132
                                 - 0.140999996117328 z_k1_s0133
                                 - 0.394999873775442 z_k1_s0134
                                 - 0.246999442413803 z_k1_s0135
                                 - 0.154000020270709 z_k1_s0136
                                 - 0.32499958733409 z_k1_s0137
                                 - 0.576000932168215 z_k1_s0140
                                 - 0.446999480691813 z_k1_s0141
                                 - 0.709997303723633 z_k1_s0142
                                 - 0.461999450505652 z_k1_s0143
                                 - 0.521997428563908 z_k1_s0144
                                 - 0.563001497002067 z_k1_s0145
                                 - 0.480998113118496 z_k1_s0146
                                 - 0.99199711646322 z_k1_s0147
                                 - 0.178000049829722 z_k1_s0150
                                 - 0.164000060786949 z_k1_s0151
                                 - 0.0880001391833403 z_k1_s0152
                                 - 0.110000140879144 z_k1_s0153
                                 - 0.0940001155730826 z_k1_s0154
                                 - 0.232000764138267 z_k1_s0155
                                 - 0.216999830162986 z_k1_s0156
                                 - 0.241000000522202 z_k1_s0157
                                 - 0.52999999730824 z_k1_s0160
                                 - 0.337000219139459 z_k1_s0161
                                 - 0.909000611089228 z_k1_s0162
                                 - 0.247999931815911 z_k1_s0163
                                 - 0.259000246305424 z_k1_s0164
                                 - 0.20199993874199 z_k1_s0165
                                 - 0.601000318185481 z_k1_s0166
                                 - 0.461000185047414 z_k1_s0167
                                 - 0.848003692879054 z_k1_s0170
                                 - 0.351001372684073 z_k1_s0171
                                 - 0.313000928653159 z_k1_s0172
                                 - 0.520000341192813 z_k1_s0173
                                 - 0.354000380985618 z_k1_s0174
                                 - 0.451000654179998 z_k1_s0175
                                 - 0.485001365971896 z_k1_s0176
                                 - 0.193000525453703 z_k1_s0177
                                 - 1.47899865201413 z_k1_s1000
                                 - 0.429000326027146 z_k1_s1001
                                 - 1.54600615074396 z_k1_s1002
                                 - 1.66099890869243 z_k1_s1003
                                 - 1.03899706570295 z_k1_s1004
                                 - 0.662000300516041 z_k1_s1005
                                 - 2.00400465051463 z_k1_s1006
                                 - 0.94500103229089 z_k1_s1007
                                 - 0.36400038782014 z_k1_s1010
                                 - 0.231000209241606 z_k1_s1011
                                 - 0.368000295153272 z_k1_s1012
                                 - 0.572999189228402 z_k1_s1013
                                 - 0.916001789300607 z_k1_s1014
                                 - 0.290000184656083 z_k1_s1015
                                 - 0.707001837391673 z_k1_s1016
                                 - 0.386000645152812 z_k1_s1017
                                 - 0.926002064198997 z_k1_s1020
                                 - 1.1250019430214 z_k1_s1021
                                 - 1.92299764453992 z_k1_s1022
                                 - 0.572000166200118 z_k1_s1023
                                 - 0.455000081981906 z_k1_s1024
                                 - 1.10199792238317 z_k1_s1025
                                 - 0.591000028779126 z_k1_s1026
                                 - 0.672000257068677 z_k1_s1027
                                 - 0.199999934732727 z_k1_s1030
                                 - 0.323000056093382 z_k1_s1031
                                 - 0.123000121360655 z_k1_s1032
                                 - 0.31600030487778 z_k1_s1033
                                 - 0.355000436650687 z_k1_s1034
                                 - 0.336000808351141 z_k1_s1035
                                 - 0.251000793589725 z_k1_s1036
                                 - 0.221000038379683 z_k1_s1037
                                 - 0.513001029429956 z_k1_s1040
                                 - 0.250000409485379 z_k1_s1041
                                 - 0.998004397368453 z_k1_s1042
                                 - 0.873000893219531 z_k1_s1043
                                 - 0.862002872442747 z_k1_s1044
                                 - 1.1490017236233 z_k1_s1045
                                 - 0.451000787102916 z_k1_s1046
                                 - 0.403001025224721 z_k1_s1047
                                 - 0.354000087206481 z_k1_s1050
                                 - 0.617001293542093 z_k1_s1051
                                 - 0.949001461891739 z_k1_s1052
                                 - 0.674998380926276 z_k1_s1053
                                 - 0.640998767493181 z_k1_s1054
                                 - 0.688998503398938 z_k1_s1055
                                 - 0.666001722196409 z_k1_s1056
                                 - 0.197000013364032 z_k1_s1057
                                 - 0.464998094805465 z_k1_s1060
                                 - 0.543001324064032 z_k1_s1061
                                 - 0.776001181590411 z_k1_s1062
                                 - 0.672998496159616 z_k1_s1063
                                 - 0.246999963482928 z_k1_s1064
                                 - 0.320000017506239 z_k1_s1065
                                 - 0.647999039987388 z_k1_s1066
                                 - 0.432999611905216 z_k1_s1067
                                 - 0.69400002092339 z_k1_s1070
                                 - 0.766999845761166 z_k1_s1071
                                 - 0.527000472732859 z_k1_s1072
                                 - 0.238000093169687 z_k1_s1073
                                 - 0.174000011695489 z_k1_s1074
                                 - 0.578000759126345 z_k1_s1075
                                 - 0.476999826268684 z_k1_s1076
                                 - 0.27499982555323 z_k1_s1077
                                 - 0.508000838522415 z_k1_s1100
                                 - 0.549998416905471 z_k1_s1101
                                 - 0.178999857865407 z_k1_s1102
                                 - 0.271999869996552 z_k1_s1103
                                 - 0.259999951173136 z_k1_s1104
                                 - 0.292999799187493 z_k1_s1105
                                 - 0.347999070878334 z_k1_s1106
                                 - 0.43999987352379 z_k1_s1107
                                 - 0.226999972546089 z_k1_s1110
                                 - 0.147999530277727 z_k1_s1111
                                 - 0.0899998529577761 z_k1_s1112
                                 - 0.201000055064866 z_k1_s1113
                                 - 0.163000175669234 z_k1_s1114
                                 - 0.207000036022072 z_k1_s1115
                                 - 0.249999206550215 z_k1_s1116
                                 - 0.0999998212197847 z_k1_s1117
                                 - 0.0730000934814808 z_k1_s1120
                                 - 0.0449999679832096 z_k1_s1121
                                 - 0.159000058733806 z_k1_s1122
                                 - 0.0559999204969295 z_k1_s1123
                                 - 0.0609999169395084 z_k1_s1124
                                 - 0.0950001307087657 z_k1_s1125
                                 - 0.0439999422547248 z_k1_s1126
                                 - 0.12799985605008 z_k1_s1127
                                 - 0.10400007782977 z_k1_s1130
                                 - 0.179000171033895 z_k1_s1131
                                 - 0.0569999165952811 z_k1_s1132
                                 - 0.0779999619724181 z_k1_s1133
                                 - 0.0789999182285156 z_k1_s1134
                                 - 0.218999385276884 z_k1_s1135
                                 - 0.202999892379507 z_k1_s1136
                                 - 0.0449999595225664 z_k1_s1137
                                 - 0.118999971646868 z_k1_s1140
                                 - 0.0620001091772784 z_k1_s1141
                                 - 0.0750001606176567 z_k1_s1142
                                 - 0.0840000146918842 z_k1_s1143
                                 - 0.0630001585188161 z_k1_s1144
                                 - 0.0970000661322625 z_k1_s1145
                                 - 0.0240000041976812 z_k1_s1146
                                 - 0.0659997165438176 z_k1_s1147
                                 - 0.136999643754704 z_k1_s1150
                                 - 0.0519999726491904 z_k1_s1151
                                 - 0.0749997289460344 z_k1_s1152
                                 - 0.0469999690194629 z_k1_s1153
                                 - 0.0839998005795196 z_k1_s1154
                                 - 0.115999693609824 z_k1_s1155
                                 - 0.0870002584571856 z_k1_s1156
                                 - 0.052999973375136 z_k1_s1157
                                 - 0.187999784571835 z_k1_s1160
                                 - 0.290999780799817 z_k1_s1161
                                 - 0.335998510513335 z_k1_s1162
                                 - 0.294998633665086 z_k1_s1163
                                 - 0.0819999326961782 z_k1_s1164
                                 - 0.162999794176678 z_k1_s1165
                                 - 0.265998895406257 z_k1_s1166
                                 - 0.168999684471707 z_k1_s1167
                                 - 0.0459999518806623 z_k1_s1170
                                 - 0.0809998227247031 z_k1_s1171
                                 - 0.162999652816372 z_k1_s1172
                                 - 0.117000087935464 z_k1_s1173
                                 - 0.170999711752096 z_k1_s1174
                                 - 0.0449999445136967 z_k1_s1175
                                 - 0.106000006898843 z_k1_s1176
                                 - 0.0449999445136967 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.4888 w_k1_ADM_s000
                                 + 1.87062 w_k1_AXP_s000
                                 + 1.31898 w_k1_DIS_s000
                                 + 1.24493 w_k1_FDX_s000
                                 + 0.993815 w_k1_IBM_s000
                                 + 1.23247 w_k1_INTC_s000
                                 + 1.02683 w_k1_KO_s000 + 1.11961 w_k1_MAT_s000
                                 + 1.20173 w_k1_MSFT_s000
                                 + 1.09293 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.04945 w_k1_ADM_s000
                                 + 1.34601 w_k1_AXP_s000
                                 + 0.985017 w_k1_DIS_s000
                                 + 0.979341 w_k1_FDX_s000
                                 + 1.15869 w_k1_IBM_s000
                                 + 1.10827 w_k1_INTC_s000 + 1.0131 w_k1_KO_s000
                                 + 1.20515 w_k1_MAT_s000
                                 + 0.96791 w_k1_MSFT_s000
                                 + 0.958232 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.13712 w_k1_ADM_s000
                                 + 0.936916 w_k1_AXP_s000
                                 + 1.04594 w_k1_DIS_s000
                                 + 0.931736 w_k1_FDX_s000
                                 + 0.979866 w_k1_IBM_s000
                                 + 1.04979 w_k1_INTC_s000
                                 + 1.19272 w_k1_KO_s000
                                 + 0.798466 w_k1_MAT_s000
                                 + 0.985201 w_k1_MSFT_s000
                                 + 1.10462 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.04149 w_k1_ADM_s000
                                 + 0.839419 w_k1_AXP_s000
                                 + 0.732666 w_k1_DIS_s000
                                 + 0.848189 w_k1_FDX_s000
                                 + 0.878134 w_k1_IBM_s000
                                 + 0.81268 w_k1_INTC_s000
                                 + 0.866555 w_k1_KO_s000
                                 + 0.885005 w_k1_MAT_s000
                                 + 0.930715 w_k1_MSFT_s000
                                 + 0.984885 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.90471 w_k1_ADM_s000
                                 + 0.818796 w_k1_AXP_s000
                                 + 0.691415 w_k1_DIS_s000
                                 + 0.772515 w_k1_FDX_s000
                                 + 0.767106 w_k1_IBM_s000
                                 + 0.759931 w_k1_INTC_s000
                                 + 0.765274 w_k1_KO_s000
                                 + 0.749052 w_k1_MAT_s000
                                 + 0.715354 w_k1_MSFT_s000
                                 + 0.788833 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.22785 w_k1_ADM_s000
                                 + 1.37179 w_k1_AXP_s000 + 1.5775 w_k1_DIS_s000
                                 + 1.11069 w_k1_FDX_s000
                                 + 1.57883 w_k1_IBM_s000
                                 + 1.05772 w_k1_INTC_s000
                                 + 1.44935 w_k1_KO_s000 + 1.2138 w_k1_MAT_s000
                                 + 1.13947 w_k1_MSFT_s000
                                 + 1.17459 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.34072 w_k1_ADM_s000
                                 + 0.999785 w_k1_AXP_s000
                                 + 0.761429 w_k1_DIS_s000
                                 + 1.0078 w_k1_FDX_s000
                                 + 0.594247 w_k1_IBM_s000
                                 + 0.978809 w_k1_INTC_s000
                                 + 0.803735 w_k1_KO_s000
                                 + 0.480873 w_k1_MAT_s000
                                 + 0.987163 w_k1_MSFT_s000
                                 + 0.976133 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.29875 w_k1_ADM_s000
                                 + 1.06373 w_k1_AXP_s000
                                 + 1.36474 w_k1_DIS_s000
                                 + 1.17469 w_k1_FDX_s000
                                 + 1.02218 w_k1_IBM_s000
                                 + 1.08786 w_k1_INTC_s000 + 1.2412 w_k1_KO_s000
                                 + 1.02216 w_k1_MAT_s000
                                 + 1.15961 w_k1_MSFT_s000
                                 + 1.15374 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.09638 w_k1_ADM_s001
                                 + 0.753953 w_k1_AXP_s001
                                 + 0.679766 w_k1_DIS_s001
                                 + 0.815843 w_k1_FDX_s001
                                 + 0.73599 w_k1_IBM_s001
                                 + 0.851491 w_k1_INTC_s001
                                 + 0.661559 w_k1_KO_s001
                                 + 0.513404 w_k1_MAT_s001
                                 + 0.769323 w_k1_MSFT_s001
                                 + 0.90347 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.19279 w_k1_ADM_s001
                                 + 1.19585 w_k1_AXP_s001
                                 + 1.10494 w_k1_DIS_s001
                                 + 1.14668 w_k1_FDX_s001
                                 + 1.13979 w_k1_IBM_s001
                                 + 1.03642 w_k1_INTC_s001
                                 + 0.975853 w_k1_KO_s001
                                 + 1.10446 w_k1_MAT_s001
                                 + 1.20625 w_k1_MSFT_s001 + 1.0851 w_k1_PG_s001
                                 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.20386 w_k1_ADM_s001
                                 + 1.18722 w_k1_AXP_s001
                                 + 1.15327 w_k1_DIS_s001 + 1.0738 w_k1_FDX_s001
                                 + 1.74231 w_k1_IBM_s001
                                 + 0.998407 w_k1_INTC_s001
                                 + 1.08848 w_k1_KO_s001 + 1.42989 w_k1_MAT_s001
                                 + 1.12312 w_k1_MSFT_s001
                                 + 1.17087 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.992943 w_k1_ADM_s001
                                 + 0.989056 w_k1_AXP_s001
                                 + 0.993123 w_k1_DIS_s001
                                 + 1.10306 w_k1_FDX_s001
                                 + 0.798209 w_k1_IBM_s001
                                 + 1.2081 w_k1_INTC_s001
                                 + 0.858595 w_k1_KO_s001
                                 + 1.01544 w_k1_MAT_s001
                                 + 1.13101 w_k1_MSFT_s001
                                 + 1.07333 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.12385 w_k1_ADM_s001
                                 + 0.783319 w_k1_AXP_s001
                                 + 1.15369 w_k1_DIS_s001
                                 + 0.967093 w_k1_FDX_s001
                                 + 0.942685 w_k1_IBM_s001
                                 + 1.02695 w_k1_INTC_s001
                                 + 1.03996 w_k1_KO_s001
                                 + 0.785815 w_k1_MAT_s001
                                 + 1.07972 w_k1_MSFT_s001 + 1.041 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.42064 w_k1_ADM_s001
                                 + 1.45574 w_k1_AXP_s001
                                 + 1.54454 w_k1_DIS_s001
                                 + 1.30525 w_k1_FDX_s001
                                 + 1.16959 w_k1_IBM_s001
                                 + 1.37827 w_k1_INTC_s001
                                 + 1.15644 w_k1_KO_s001 + 1.17514 w_k1_MAT_s001
                                 + 1.33819 w_k1_MSFT_s001
                                 + 1.27148 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.04887 w_k1_ADM_s001
                                 + 0.753429 w_k1_AXP_s001
                                 + 0.620928 w_k1_DIS_s001
                                 + 0.730505 w_k1_FDX_s001
                                 + 1.18097 w_k1_IBM_s001
                                 + 0.897776 w_k1_INTC_s001
                                 + 0.807153 w_k1_KO_s001
                                 + 0.817095 w_k1_MAT_s001
                                 + 0.750212 w_k1_MSFT_s001
                                 + 0.94213 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.900944 w_k1_ADM_s001
                                 + 0.88681 w_k1_AXP_s001
                                 + 1.25342 w_k1_DIS_s001
                                 + 1.00589 w_k1_FDX_s001
                                 + 1.02936 w_k1_IBM_s001
                                 + 1.01439 w_k1_INTC_s001
                                 + 1.01046 w_k1_KO_s001 + 1.1297 w_k1_MAT_s001
                                 + 1.17229 w_k1_MSFT_s001
                                 + 1.15723 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.16538 w_k1_ADM_s002
                                 + 1.19106 w_k1_AXP_s002
                                 + 0.95061 w_k1_DIS_s002
                                 + 1.01645 w_k1_FDX_s002
                                 + 1.36685 w_k1_IBM_s002
                                 + 0.950409 w_k1_INTC_s002
                                 + 1.04805 w_k1_KO_s002 + 1.66595 w_k1_MAT_s002
                                 + 0.949512 w_k1_MSFT_s002
                                 + 1.13292 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.05358 w_k1_ADM_s002
                                 + 0.871755 w_k1_AXP_s002
                                 + 0.844648 w_k1_DIS_s002
                                 + 0.976666 w_k1_FDX_s002
                                 + 0.849725 w_k1_IBM_s002
                                 + 0.738331 w_k1_INTC_s002
                                 + 0.849508 w_k1_KO_s002
                                 + 0.657115 w_k1_MAT_s002
                                 + 0.843697 w_k1_MSFT_s002
                                 + 0.925166 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.41383 w_k1_ADM_s002
                                 + 1.64954 w_k1_AXP_s002
                                 + 1.39788 w_k1_DIS_s002
                                 + 1.27116 w_k1_FDX_s002
                                 + 1.11259 w_k1_IBM_s002
                                 + 1.03486 w_k1_INTC_s002
                                 + 1.06396 w_k1_KO_s002 + 1.22115 w_k1_MAT_s002
                                 + 1.05906 w_k1_MSFT_s002
                                 + 1.09728 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.788423 w_k1_ADM_s002
                                 + 0.679002 w_k1_AXP_s002
                                 + 1.01747 w_k1_DIS_s002
                                 + 0.925142 w_k1_FDX_s002
                                 + 1.03288 w_k1_IBM_s002
                                 + 1.09768 w_k1_INTC_s002
                                 + 0.95315 w_k1_KO_s002
                                 + 0.670818 w_k1_MAT_s002
                                 + 0.958634 w_k1_MSFT_s002
                                 + 0.892248 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.757795 w_k1_ADM_s002
                                 + 0.693722 w_k1_AXP_s002
                                 + 0.604155 w_k1_DIS_s002
                                 + 0.744061 w_k1_FDX_s002
                                 + 0.828597 w_k1_IBM_s002
                                 + 0.837123 w_k1_INTC_s002
                                 + 0.812007 w_k1_KO_s002
                                 + 0.776991 w_k1_MAT_s002
                                 + 0.68358 w_k1_MSFT_s002
                                 + 0.862707 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.16067 w_k1_ADM_s002
                                 + 1.20647 w_k1_AXP_s002
                                 + 1.29072 w_k1_DIS_s002
                                 + 1.17741 w_k1_FDX_s002
                                 + 1.22561 w_k1_IBM_s002
                                 + 1.08332 w_k1_INTC_s002
                                 + 1.21576 w_k1_KO_s002
                                 + 0.943356 w_k1_MAT_s002
                                 + 1.1381 w_k1_MSFT_s002 + 1.06569 w_k1_PG_s002
                                 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.871551 w_k1_ADM_s002
                                 + 0.788698 w_k1_AXP_s002
                                 + 0.790898 w_k1_DIS_s002
                                 + 0.796566 w_k1_FDX_s002
                                 + 1.30743 w_k1_IBM_s002
                                 + 0.739078 w_k1_INTC_s002
                                 + 1.08189 w_k1_KO_s002 + 1.27202 w_k1_MAT_s002
                                 + 0.814426 w_k1_MSFT_s002
                                 + 1.13252 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.19514 w_k1_ADM_s002
                                 + 1.13261 w_k1_AXP_s002
                                 + 1.44407 w_k1_DIS_s002
                                 + 1.13588 w_k1_FDX_s002
                                 + 1.68485 w_k1_IBM_s002
                                 + 1.16311 w_k1_INTC_s002
                                 + 1.43819 w_k1_KO_s002 + 1.22098 w_k1_MAT_s002
                                 + 1.2074 w_k1_MSFT_s002 + 1.2721 w_k1_PG_s002
                                 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.920604 w_k1_ADM_s003
                                 + 1.13641 w_k1_AXP_s003
                                 + 0.947251 w_k1_DIS_s003
                                 + 1.06578 w_k1_FDX_s003
                                 + 0.638397 w_k1_IBM_s003
                                 + 0.796846 w_k1_INTC_s003
                                 + 0.695243 w_k1_KO_s003
                                 + 0.760951 w_k1_MAT_s003
                                 + 0.95866 w_k1_MSFT_s003
                                 + 0.873717 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.721382 w_k1_ADM_s003
                                 + 0.945449 w_k1_AXP_s003
                                 + 0.981883 w_k1_DIS_s003
                                 + 0.919051 w_k1_FDX_s003
                                 + 1.23853 w_k1_IBM_s003
                                 + 0.911449 w_k1_INTC_s003
                                 + 0.955508 w_k1_KO_s003
                                 + 1.08277 w_k1_MAT_s003
                                 + 0.986316 w_k1_MSFT_s003
                                 + 0.940528 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.22855 w_k1_ADM_s003
                                 + 1.16221 w_k1_AXP_s003
                                 + 1.31996 w_k1_DIS_s003
                                 + 1.16991 w_k1_FDX_s003
                                 + 0.831385 w_k1_IBM_s003
                                 + 0.966559 w_k1_INTC_s003
                                 + 0.924409 w_k1_KO_s003
                                 + 0.79441 w_k1_MAT_s003
                                 + 1.25259 w_k1_MSFT_s003
                                 + 1.00304 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.00008 w_k1_ADM_s003
                                 + 1.12838 w_k1_AXP_s003
                                 + 1.43112 w_k1_DIS_s003
                                 + 1.14071 w_k1_FDX_s003
                                 + 1.15797 w_k1_IBM_s003
                                 + 0.984291 w_k1_INTC_s003
                                 + 1.09698 w_k1_KO_s003 + 1.071 w_k1_MAT_s003
                                 + 1.1173 w_k1_MSFT_s003
                                 + 0.982253 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.749349 w_k1_ADM_s003
                                 + 0.703536 w_k1_AXP_s003
                                 + 0.797932 w_k1_DIS_s003
                                 + 0.770762 w_k1_FDX_s003
                                 + 0.83099 w_k1_IBM_s003
                                 + 0.759938 w_k1_INTC_s003
                                 + 0.859813 w_k1_KO_s003
                                 + 0.784667 w_k1_MAT_s003
                                 + 0.850979 w_k1_MSFT_s003
                                 + 0.814257 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.18742 w_k1_ADM_s003
                                 + 1.66301 w_k1_AXP_s003
                                 + 1.91545 w_k1_DIS_s003
                                 + 1.42785 w_k1_FDX_s003
                                 + 1.21858 w_k1_IBM_s003
                                 + 1.01858 w_k1_INTC_s003
                                 + 1.19124 w_k1_KO_s003 + 1.35373 w_k1_MAT_s003
                                 + 1.19706 w_k1_MSFT_s003
                                 + 1.22387 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.02285 w_k1_ADM_s003
                                 + 1.16134 w_k1_AXP_s003
                                 + 0.924288 w_k1_DIS_s003
                                 + 0.98277 w_k1_FDX_s003
                                 + 0.923772 w_k1_IBM_s003
                                 + 0.940996 w_k1_INTC_s003
                                 + 0.895655 w_k1_KO_s003
                                 + 0.759817 w_k1_MAT_s003
                                 + 0.912386 w_k1_MSFT_s003
                                 + 0.970135 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.889323 w_k1_ADM_s003
                                 + 1.3055 w_k1_AXP_s003 + 1.44626 w_k1_DIS_s003
                                 + 1.04825 w_k1_FDX_s003
                                 + 1.73966 w_k1_IBM_s003
                                 + 0.996556 w_k1_INTC_s003
                                 + 1.33609 w_k1_KO_s003 + 1.65236 w_k1_MAT_s003
                                 + 1.09242 w_k1_MSFT_s003
                                 + 1.16876 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.03929 w_k1_ADM_s004
                                 + 1.26773 w_k1_AXP_s004
                                 + 1.40292 w_k1_DIS_s004
                                 + 1.20219 w_k1_FDX_s004
                                 + 1.03765 w_k1_IBM_s004
                                 + 1.25536 w_k1_INTC_s004 + 1.1496 w_k1_KO_s004
                                 + 1.08469 w_k1_MAT_s004
                                 + 1.23866 w_k1_MSFT_s004
                                 + 1.27666 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.644301 w_k1_ADM_s004
                                 + 0.930335 w_k1_AXP_s004
                                 + 0.657503 w_k1_DIS_s004
                                 + 0.766925 w_k1_FDX_s004
                                 + 1.24741 w_k1_IBM_s004
                                 + 0.930586 w_k1_INTC_s004
                                 + 0.976423 w_k1_KO_s004
                                 + 1.39996 w_k1_MAT_s004
                                 + 0.839271 w_k1_MSFT_s004
                                 + 0.954401 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.665652 w_k1_ADM_s004
                                 + 0.65575 w_k1_AXP_s004
                                 + 0.560772 w_k1_DIS_s004
                                 + 0.69411 w_k1_FDX_s004
                                 + 0.718555 w_k1_IBM_s004
                                 + 0.983612 w_k1_INTC_s004
                                 + 0.772499 w_k1_KO_s004
                                 + 0.767388 w_k1_MAT_s004
                                 + 0.768515 w_k1_MSFT_s004
                                 + 0.894444 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.942442 w_k1_ADM_s004
                                 + 0.959813 w_k1_AXP_s004
                                 + 0.974982 w_k1_DIS_s004
                                 + 0.901304 w_k1_FDX_s004
                                 + 1.26173 w_k1_IBM_s004
                                 + 0.984292 w_k1_INTC_s004
                                 + 1.14336 w_k1_KO_s004 + 1.27148 w_k1_MAT_s004
                                 + 0.929563 w_k1_MSFT_s004
                                 + 1.05148 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.79223 w_k1_ADM_s004
                                 + 0.769427 w_k1_AXP_s004
                                 + 0.758371 w_k1_DIS_s004
                                 + 0.832774 w_k1_FDX_s004
                                 + 0.614384 w_k1_IBM_s004
                                 + 0.857506 w_k1_INTC_s004
                                 + 0.840969 w_k1_KO_s004
                                 + 0.741872 w_k1_MAT_s004
                                 + 0.810764 w_k1_MSFT_s004
                                 + 0.925008 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.988647 w_k1_ADM_s004
                                 + 0.952062 w_k1_AXP_s004
                                 + 1.0845 w_k1_DIS_s004 + 1.03393 w_k1_FDX_s004
                                 + 0.901106 w_k1_IBM_s004
                                 + 1.21831 w_k1_INTC_s004
                                 + 1.12668 w_k1_KO_s004
                                 + 0.688333 w_k1_MAT_s004
                                 + 1.04592 w_k1_MSFT_s004 + 1.0629 w_k1_PG_s004
                                 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.00342 w_k1_ADM_s004
                                 + 1.31484 w_k1_AXP_s004
                                 + 0.961826 w_k1_DIS_s004
                                 + 0.951183 w_k1_FDX_s004
                                 + 1.1755 w_k1_IBM_s004
                                 + 1.06841 w_k1_INTC_s004 + 1.0691 w_k1_KO_s004
                                 + 1.18933 w_k1_MAT_s004
                                 + 0.947506 w_k1_MSFT_s004
                                 + 1.20628 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.41407 w_k1_ADM_s004
                                 + 1.48053 w_k1_AXP_s004
                                 + 1.67047 w_k1_DIS_s004
                                 + 1.48646 w_k1_FDX_s004
                                 + 1.25322 w_k1_IBM_s004
                                 + 1.51535 w_k1_INTC_s004
                                 + 1.43826 w_k1_KO_s004 + 1.23769 w_k1_MAT_s004
                                 + 1.3876 w_k1_MSFT_s004 + 1.32322 w_k1_PG_s004
                                 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.754649 w_k1_ADM_s005
                                 + 0.755068 w_k1_AXP_s005
                                 + 0.751056 w_k1_DIS_s005
                                 + 0.937101 w_k1_FDX_s005
                                 + 0.728065 w_k1_IBM_s005
                                 + 0.825583 w_k1_INTC_s005
                                 + 0.71824 w_k1_KO_s005 + 0.68499 w_k1_MAT_s005
                                 + 0.858799 w_k1_MSFT_s005
                                 + 0.926303 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.72844 w_k1_ADM_s005
                                 + 0.696817 w_k1_AXP_s005
                                 + 0.739856 w_k1_DIS_s005
                                 + 0.731423 w_k1_FDX_s005
                                 + 0.953771 w_k1_IBM_s005
                                 + 0.775845 w_k1_INTC_s005
                                 + 0.755761 w_k1_KO_s005
                                 + 0.794731 w_k1_MAT_s005
                                 + 0.741391 w_k1_MSFT_s005
                                 + 0.940456 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.884566 w_k1_ADM_s005
                                 + 0.827595 w_k1_AXP_s005
                                 + 1.01268 w_k1_DIS_s005
                                 + 1.03636 w_k1_FDX_s005
                                 + 0.945045 w_k1_IBM_s005
                                 + 1.05255 w_k1_INTC_s005 + 1.0088 w_k1_KO_s005
                                 + 1.04348 w_k1_MAT_s005
                                 + 1.1601 w_k1_MSFT_s005 + 1.01251 w_k1_PG_s005
                                 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.914137 w_k1_ADM_s005
                                 + 0.948241 w_k1_AXP_s005
                                 + 0.882408 w_k1_DIS_s005
                                 + 1.12966 w_k1_FDX_s005
                                 + 0.575519 w_k1_IBM_s005
                                 + 1.0096 w_k1_INTC_s005
                                 + 0.685319 w_k1_KO_s005
                                 + 0.734308 w_k1_MAT_s005
                                 + 0.984842 w_k1_MSFT_s005
                                 + 0.926866 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.2954 w_k1_ADM_s005
                                 + 1.22658 w_k1_AXP_s005
                                 + 1.37519 w_k1_DIS_s005
                                 + 1.35996 w_k1_FDX_s005 + 0.8401 w_k1_IBM_s005
                                 + 1.22081 w_k1_INTC_s005 + 1.0294 w_k1_KO_s005
                                 + 0.838722 w_k1_MAT_s005
                                 + 1.20477 w_k1_MSFT_s005
                                 + 1.16791 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.733655 w_k1_ADM_s005
                                 + 0.54669 w_k1_AXP_s005
                                 + 0.577604 w_k1_DIS_s005
                                 + 0.773938 w_k1_FDX_s005
                                 + 0.562312 w_k1_IBM_s005
                                 + 0.79311 w_k1_INTC_s005
                                 + 0.610659 w_k1_KO_s005
                                 + 0.446093 w_k1_MAT_s005
                                 + 0.830619 w_k1_MSFT_s005
                                 + 0.918897 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.893209 w_k1_ADM_s005
                                 + 1.07263 w_k1_AXP_s005
                                 + 1.04322 w_k1_DIS_s005
                                 + 1.03365 w_k1_FDX_s005
                                 + 1.39142 w_k1_IBM_s005
                                 + 1.16422 w_k1_INTC_s005
                                 + 1.18256 w_k1_KO_s005 + 1.46444 w_k1_MAT_s005
                                 + 1.01714 w_k1_MSFT_s005
                                 + 1.18946 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.19551 w_k1_ADM_s005
                                 + 1.43632 w_k1_AXP_s005
                                 + 1.56287 w_k1_DIS_s005
                                 + 1.27461 w_k1_FDX_s005
                                 + 1.84625 w_k1_IBM_s005
                                 + 1.21525 w_k1_INTC_s005
                                 + 1.47616 w_k1_KO_s005 + 1.51155 w_k1_MAT_s005
                                 + 1.39934 w_k1_MSFT_s005
                                 + 1.43433 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.27543 w_k1_ADM_s006
                                 + 1.17182 w_k1_AXP_s006
                                 + 1.48031 w_k1_DIS_s006
                                 + 1.13618 w_k1_FDX_s006 + 1.5257 w_k1_IBM_s006
                                 + 0.957957 w_k1_INTC_s006
                                 + 1.35113 w_k1_KO_s006 + 0.91319 w_k1_MAT_s006
                                 + 1.14129 w_k1_MSFT_s006
                                 + 1.11871 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.81702 w_k1_ADM_s006
                                 + 0.76532 w_k1_AXP_s006
                                 + 0.801132 w_k1_DIS_s006
                                 + 0.853075 w_k1_FDX_s006
                                 + 1.03719 w_k1_IBM_s006
                                 + 0.917283 w_k1_INTC_s006
                                 + 0.976147 w_k1_KO_s006
                                 + 0.720533 w_k1_MAT_s006
                                 + 0.832013 w_k1_MSFT_s006
                                 + 0.873852 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.27076 w_k1_ADM_s006
                                 + 1.60785 w_k1_AXP_s006
                                 + 1.55977 w_k1_DIS_s006
                                 + 1.38212 w_k1_FDX_s006
                                 + 1.21302 w_k1_IBM_s006
                                 + 1.23059 w_k1_INTC_s006
                                 + 1.14497 w_k1_KO_s006 + 0.95027 w_k1_MAT_s006
                                 + 1.18197 w_k1_MSFT_s006
                                 + 1.32023 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.96612 w_k1_ADM_s006
                                 + 1.06227 w_k1_AXP_s006
                                 + 1.24532 w_k1_DIS_s006
                                 + 0.992506 w_k1_FDX_s006
                                 + 1.4826 w_k1_IBM_s006
                                 + 0.940325 w_k1_INTC_s006
                                 + 1.24731 w_k1_KO_s006 + 1.07102 w_k1_MAT_s006
                                 + 1.06237 w_k1_MSFT_s006 + 1.0537 w_k1_PG_s006
                                 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.22029 w_k1_ADM_s006
                                 + 1.11075 w_k1_AXP_s006
                                 + 0.786842 w_k1_DIS_s006
                                 + 1.05836 w_k1_FDX_s006
                                 + 0.825335 w_k1_IBM_s006
                                 + 0.843946 w_k1_INTC_s006
                                 + 0.822609 w_k1_KO_s006
                                 + 0.854645 w_k1_MAT_s006
                                 + 0.963031 w_k1_MSFT_s006
                                 + 1.06605 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.705518 w_k1_ADM_s006
                                 + 0.825678 w_k1_AXP_s006
                                 + 0.756738 w_k1_DIS_s006
                                 + 0.848756 w_k1_FDX_s006
                                 + 0.690845 w_k1_IBM_s006
                                 + 0.795039 w_k1_INTC_s006
                                 + 0.814796 w_k1_KO_s006
                                 + 0.807052 w_k1_MAT_s006
                                 + 0.841285 w_k1_MSFT_s006
                                 + 0.897906 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.18185 w_k1_ADM_s006
                                 + 1.17933 w_k1_AXP_s006
                                 + 1.22178 w_k1_DIS_s006
                                 + 1.29469 w_k1_FDX_s006
                                 + 0.732343 w_k1_IBM_s006
                                 + 0.998699 w_k1_INTC_s006
                                 + 0.915404 w_k1_KO_s006
                                 + 0.674179 w_k1_MAT_s006
                                 + 1.06251 w_k1_MSFT_s006
                                 + 1.11645 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.2716 w_k1_ADM_s006
                                 + 1.1119 w_k1_AXP_s006 + 1.26254 w_k1_DIS_s006
                                 + 1.11534 w_k1_FDX_s006
                                 + 0.908675 w_k1_IBM_s006
                                 + 1.02031 w_k1_INTC_s006
                                 + 0.939061 w_k1_KO_s006
                                 + 0.538703 w_k1_MAT_s006
                                 + 1.01968 w_k1_MSFT_s006
                                 + 1.03442 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.812092 w_k1_ADM_s007
                                 + 0.752717 w_k1_AXP_s007
                                 + 1.09878 w_k1_DIS_s007
                                 + 0.800879 w_k1_FDX_s007
                                 + 1.31058 w_k1_IBM_s007
                                 + 0.784836 w_k1_INTC_s007
                                 + 1.10511 w_k1_KO_s007 + 1.096 w_k1_MAT_s007
                                 + 1.03248 w_k1_MSFT_s007 + 1.0164 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.755915 w_k1_ADM_s007
                                 + 0.850715 w_k1_AXP_s007
                                 + 0.666501 w_k1_DIS_s007
                                 + 0.797226 w_k1_FDX_s007
                                 + 1.04791 w_k1_IBM_s007
                                 + 0.70301 w_k1_INTC_s007
                                 + 0.799365 w_k1_KO_s007
                                 + 1.29035 w_k1_MAT_s007
                                 + 0.719163 w_k1_MSFT_s007
                                 + 0.871046 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.972031 w_k1_ADM_s007
                                 + 0.933662 w_k1_AXP_s007
                                 + 0.826261 w_k1_DIS_s007
                                 + 0.935159 w_k1_FDX_s007
                                 + 0.694945 w_k1_IBM_s007
                                 + 0.864963 w_k1_INTC_s007
                                 + 0.705635 w_k1_KO_s007
                                 + 0.624844 w_k1_MAT_s007
                                 + 0.899213 w_k1_MSFT_s007
                                 + 0.791646 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.70736 w_k1_ADM_s007
                                 + 0.773068 w_k1_AXP_s007
                                 + 0.865284 w_k1_DIS_s007
                                 + 0.846369 w_k1_FDX_s007
                                 + 0.855975 w_k1_IBM_s007
                                 + 0.773812 w_k1_INTC_s007
                                 + 0.884649 w_k1_KO_s007
                                 + 0.933799 w_k1_MAT_s007
                                 + 0.955815 w_k1_MSFT_s007
                                 + 0.878382 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.952344 w_k1_ADM_s007
                                 + 1.20696 w_k1_AXP_s007
                                 + 1.07432 w_k1_DIS_s007
                                 + 1.04554 w_k1_FDX_s007
                                 + 1.28503 w_k1_IBM_s007
                                 + 1.12138 w_k1_INTC_s007
                                 + 1.06076 w_k1_KO_s007 + 1.3068 w_k1_MAT_s007
                                 + 1.12581 w_k1_MSFT_s007 + 1.0399 w_k1_PG_s007
                                 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.33772 w_k1_ADM_s007
                                 + 1.43085 w_k1_AXP_s007
                                 + 1.69773 w_k1_DIS_s007
                                 + 1.37606 w_k1_FDX_s007
                                 + 1.14798 w_k1_IBM_s007
                                 + 1.03432 w_k1_INTC_s007
                                 + 1.17746 w_k1_KO_s007 + 1.17752 w_k1_MAT_s007
                                 + 1.28422 w_k1_MSFT_s007 + 1.2209 w_k1_PG_s007
                                 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.981167 w_k1_ADM_s007
                                 + 1.03466 w_k1_AXP_s007
                                 + 1.14609 w_k1_DIS_s007
                                 + 1.11448 w_k1_FDX_s007
                                 + 0.816035 w_k1_IBM_s007
                                 + 0.852448 w_k1_INTC_s007
                                 + 0.845056 w_k1_KO_s007
                                 + 1.07571 w_k1_MAT_s007
                                 + 1.05305 w_k1_MSFT_s007
                                 + 0.939352 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.637378 w_k1_ADM_s007
                                 + 0.759027 w_k1_AXP_s007
                                 + 0.588703 w_k1_DIS_s007
                                 + 0.690402 w_k1_FDX_s007
                                 + 0.706836 w_k1_IBM_s007
                                 + 0.598595 w_k1_INTC_s007
                                 + 0.626708 w_k1_KO_s007
                                 + 0.817421 w_k1_MAT_s007
                                 + 0.70512 w_k1_MSFT_s007
                                 + 0.735798 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.0678 w_k1_ADM_s010
                                 + 1.60686 w_k1_AXP_s010
                                 + 1.99894 w_k1_DIS_s010
                                 + 1.60949 w_k1_FDX_s010
                                 + 1.27221 w_k1_IBM_s010
                                 + 1.52392 w_k1_INTC_s010
                                 + 1.43793 w_k1_KO_s010 + 1.59345 w_k1_MAT_s010
                                 + 1.5209 w_k1_MSFT_s010 + 1.3911 w_k1_PG_s010
                                 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.611038 w_k1_ADM_s010
                                 + 0.565841 w_k1_AXP_s010
                                 + 0.510577 w_k1_DIS_s010
                                 + 0.631313 w_k1_FDX_s010
                                 + 0.731775 w_k1_IBM_s010
                                 + 0.686825 w_k1_INTC_s010
                                 + 0.652834 w_k1_KO_s010
                                 + 0.690065 w_k1_MAT_s010
                                 + 0.675024 w_k1_MSFT_s010
                                 + 0.71372 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.74867 w_k1_ADM_s010
                                 + 0.724624 w_k1_AXP_s010
                                 + 0.739877 w_k1_DIS_s010
                                 + 0.768613 w_k1_FDX_s010
                                 + 1.15083 w_k1_IBM_s010
                                 + 0.756891 w_k1_INTC_s010
                                 + 0.957671 w_k1_KO_s010
                                 + 0.957157 w_k1_MAT_s010
                                 + 0.810286 w_k1_MSFT_s010
                                 + 0.960464 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.00934 w_k1_ADM_s010
                                 + 0.797801 w_k1_AXP_s010
                                 + 0.766408 w_k1_DIS_s010
                                 + 0.876678 w_k1_FDX_s010
                                 + 0.831505 w_k1_IBM_s010
                                 + 0.912235 w_k1_INTC_s010
                                 + 0.859387 w_k1_KO_s010
                                 + 0.962928 w_k1_MAT_s010
                                 + 0.935422 w_k1_MSFT_s010
                                 + 1.02099 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.20843 w_k1_ADM_s010
                                 + 1.0886 w_k1_AXP_s010 + 1.14854 w_k1_DIS_s010
                                 + 1.24124 w_k1_FDX_s010
                                 + 0.852133 w_k1_IBM_s010
                                 + 1.10018 w_k1_INTC_s010
                                 + 0.960444 w_k1_KO_s010
                                 + 0.945152 w_k1_MAT_s010
                                 + 1.24169 w_k1_MSFT_s010
                                 + 1.02316 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.02157 w_k1_ADM_s010
                                 + 0.844991 w_k1_AXP_s010
                                 + 1.1223 w_k1_DIS_s010 + 1.02529 w_k1_FDX_s010
                                 + 1.08142 w_k1_IBM_s010
                                 + 0.995514 w_k1_INTC_s010
                                 + 1.03063 w_k1_KO_s010
                                 + 0.757759 w_k1_MAT_s010
                                 + 1.17303 w_k1_MSFT_s010
                                 + 1.01471 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.23605 w_k1_ADM_s010
                                 + 1.20038 w_k1_AXP_s010
                                 + 1.35724 w_k1_DIS_s010 + 1.2313 w_k1_FDX_s010
                                 + 1.29936 w_k1_IBM_s010
                                 + 1.28257 w_k1_INTC_s010
                                 + 1.31283 w_k1_KO_s010 + 1.59728 w_k1_MAT_s010
                                 + 1.31286 w_k1_MSFT_s010
                                 + 1.17482 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.858962 w_k1_ADM_s010
                                 + 0.736863 w_k1_AXP_s010
                                 + 0.783601 w_k1_DIS_s010
                                 + 0.900585 w_k1_FDX_s010
                                 + 0.636837 w_k1_IBM_s010
                                 + 0.940089 w_k1_INTC_s010
                                 + 0.70926 w_k1_KO_s010
                                 + 0.551736 w_k1_MAT_s010
                                 + 0.935633 w_k1_MSFT_s010
                                 + 0.839807 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.879516 w_k1_ADM_s011
                                 + 0.716834 w_k1_AXP_s011
                                 + 0.776962 w_k1_DIS_s011
                                 + 0.829692 w_k1_FDX_s011
                                 + 1.15716 w_k1_IBM_s011
                                 + 0.938557 w_k1_INTC_s011
                                 + 0.888457 w_k1_KO_s011 + 1.0117 w_k1_MAT_s011
                                 + 0.777749 w_k1_MSFT_s011
                                 + 0.937977 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.1907 w_k1_ADM_s011
                                 + 1.14557 w_k1_AXP_s011
                                 + 1.18181 w_k1_DIS_s011
                                 + 1.29335 w_k1_FDX_s011
                                 + 0.843363 w_k1_IBM_s011
                                 + 1.33821 w_k1_INTC_s011
                                 + 0.868102 w_k1_KO_s011
                                 + 0.605604 w_k1_MAT_s011
                                 + 1.07795 w_k1_MSFT_s011
                                 + 1.03382 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.961756 w_k1_ADM_s011
                                 + 0.87742 w_k1_AXP_s011
                                 + 0.925854 w_k1_DIS_s011
                                 + 0.93399 w_k1_FDX_s011
                                 + 0.759895 w_k1_IBM_s011
                                 + 0.92861 w_k1_INTC_s011
                                 + 0.776138 w_k1_KO_s011
                                 + 0.891901 w_k1_MAT_s011
                                 + 0.862682 w_k1_MSFT_s011
                                 + 0.92657 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.05292 w_k1_ADM_s011
                                 + 1.10867 w_k1_AXP_s011
                                 + 1.03114 w_k1_DIS_s011
                                 + 1.06495 w_k1_FDX_s011
                                 + 0.997367 w_k1_IBM_s011
                                 + 1.06026 w_k1_INTC_s011
                                 + 0.94626 w_k1_KO_s011 + 1.04846 w_k1_MAT_s011
                                 + 0.997709 w_k1_MSFT_s011
                                 + 0.995486 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.852436 w_k1_ADM_s011
                                 + 0.646782 w_k1_AXP_s011
                                 + 0.643185 w_k1_DIS_s011
                                 + 0.78246 w_k1_FDX_s011
                                 + 0.700791 w_k1_IBM_s011
                                 + 0.817659 w_k1_INTC_s011
                                 + 0.582215 w_k1_KO_s011
                                 + 0.690011 w_k1_MAT_s011
                                 + 0.751903 w_k1_MSFT_s011
                                 + 0.83547 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.951899 w_k1_ADM_s011
                                 + 1.03566 w_k1_AXP_s011
                                 + 1.03981 w_k1_DIS_s011
                                 + 0.931675 w_k1_FDX_s011
                                 + 1.33105 w_k1_IBM_s011
                                 + 0.898828 w_k1_INTC_s011
                                 + 0.990355 w_k1_KO_s011
                                 + 1.62781 w_k1_MAT_s011
                                 + 0.97102 w_k1_MSFT_s011
                                 + 1.10964 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.19762 w_k1_ADM_s011
                                 + 1.34739 w_k1_AXP_s011
                                 + 1.49885 w_k1_DIS_s011
                                 + 1.21208 w_k1_FDX_s011
                                 + 1.41587 w_k1_IBM_s011
                                 + 1.10514 w_k1_INTC_s011
                                 + 1.22428 w_k1_KO_s011 + 1.45021 w_k1_MAT_s011
                                 + 1.09652 w_k1_MSFT_s011
                                 + 1.15485 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.27496 w_k1_ADM_s011
                                 + 1.64492 w_k1_AXP_s011
                                 + 1.32182 w_k1_DIS_s011
                                 + 1.27771 w_k1_FDX_s011
                                 + 1.15711 w_k1_IBM_s011
                                 + 1.32336 w_k1_INTC_s011
                                 + 1.00549 w_k1_KO_s011 + 1.40806 w_k1_MAT_s011
                                 + 0.986985 w_k1_MSFT_s011
                                 + 1.14277 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.16787 w_k1_ADM_s012
                                 + 1.35749 w_k1_AXP_s012
                                 + 1.19501 w_k1_DIS_s012
                                 + 1.15338 w_k1_FDX_s012
                                 + 1.31056 w_k1_IBM_s012
                                 + 1.21821 w_k1_INTC_s012
                                 + 1.27769 w_k1_KO_s012 + 1.60462 w_k1_MAT_s012
                                 + 1.13915 w_k1_MSFT_s012
                                 + 1.26998 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.715291 w_k1_ADM_s012
                                 + 0.640211 w_k1_AXP_s012
                                 + 0.686732 w_k1_DIS_s012
                                 + 0.76791 w_k1_FDX_s012
                                 + 0.614801 w_k1_IBM_s012
                                 + 0.898232 w_k1_INTC_s012
                                 + 0.793044 w_k1_KO_s012
                                 + 0.611543 w_k1_MAT_s012
                                 + 0.908251 w_k1_MSFT_s012
                                 + 0.85673 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.951217 w_k1_ADM_s012
                                 + 0.989513 w_k1_AXP_s012
                                 + 0.768694 w_k1_DIS_s012
                                 + 0.941304 w_k1_FDX_s012
                                 + 0.73601 w_k1_IBM_s012
                                 + 0.869149 w_k1_INTC_s012
                                 + 0.865809 w_k1_KO_s012 + 1.0189 w_k1_MAT_s012
                                 + 0.936144 w_k1_MSFT_s012
                                 + 1.07763 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.798393 w_k1_ADM_s012
                                 + 0.751251 w_k1_AXP_s012
                                 + 0.931549 w_k1_DIS_s012
                                 + 0.880247 w_k1_FDX_s012
                                 + 0.951816 w_k1_IBM_s012
                                 + 1.03833 w_k1_INTC_s012
                                 + 1.10647 w_k1_KO_s012 + 1.04047 w_k1_MAT_s012
                                 + 0.972731 w_k1_MSFT_s012
                                 + 1.05165 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.803456 w_k1_ADM_s012
                                 + 0.744766 w_k1_AXP_s012
                                 + 0.762364 w_k1_DIS_s012
                                 + 0.749477 w_k1_FDX_s012
                                 + 0.772286 w_k1_IBM_s012
                                 + 0.7143 w_k1_INTC_s012
                                 + 0.911008 w_k1_KO_s012
                                 + 0.763053 w_k1_MAT_s012
                                 + 0.832599 w_k1_MSFT_s012
                                 + 0.926816 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.06774 w_k1_ADM_s012
                                 + 1.06277 w_k1_AXP_s012
                                 + 1.75002 w_k1_DIS_s012
                                 + 1.31918 w_k1_FDX_s012
                                 + 0.992576 w_k1_IBM_s012
                                 + 1.16057 w_k1_INTC_s012
                                 + 1.58799 w_k1_KO_s012 + 1.34239 w_k1_MAT_s012
                                 + 1.34343 w_k1_MSFT_s012
                                 + 1.35232 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.829217 w_k1_ADM_s012
                                 + 0.872956 w_k1_AXP_s012
                                 + 0.797522 w_k1_DIS_s012
                                 + 0.789309 w_k1_FDX_s012
                                 + 1.34842 w_k1_IBM_s012
                                 + 0.998022 w_k1_INTC_s012
                                 + 1.31073 w_k1_KO_s012 + 1.46514 w_k1_MAT_s012
                                 + 0.957017 w_k1_MSFT_s012
                                 + 1.07779 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.866277 w_k1_ADM_s012
                                 + 0.945179 w_k1_AXP_s012
                                 + 1.24231 w_k1_DIS_s012
                                 + 1.12769 w_k1_FDX_s012
                                 + 0.851409 w_k1_IBM_s012
                                 + 1.34849 w_k1_INTC_s012
                                 + 1.15759 w_k1_KO_s012 + 1.04055 w_k1_MAT_s012
                                 + 1.27153 w_k1_MSFT_s012
                                 + 1.21085 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.672554 w_k1_ADM_s013
                                 + 0.469866 w_k1_AXP_s013
                                 + 0.463809 w_k1_DIS_s013
                                 + 0.632021 w_k1_FDX_s013
                                 + 0.628488 w_k1_IBM_s013
                                 + 0.807174 w_k1_INTC_s013
                                 + 0.570141 w_k1_KO_s013
                                 + 0.514352 w_k1_MAT_s013
                                 + 0.713981 w_k1_MSFT_s013
                                 + 0.748826 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.1891 w_k1_ADM_s013
                                 + 1.27296 w_k1_AXP_s013
                                 + 1.36119 w_k1_DIS_s013
                                 + 1.33602 w_k1_FDX_s013
                                 + 1.22687 w_k1_IBM_s013
                                 + 1.3686 w_k1_INTC_s013 + 1.26673 w_k1_KO_s013
                                 + 1.02322 w_k1_MAT_s013
                                 + 1.24404 w_k1_MSFT_s013
                                 + 1.00184 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.887707 w_k1_ADM_s013
                                 + 0.640768 w_k1_AXP_s013
                                 + 0.628858 w_k1_DIS_s013
                                 + 0.799816 w_k1_FDX_s013
                                 + 0.729255 w_k1_IBM_s013
                                 + 0.862295 w_k1_INTC_s013
                                 + 0.645559 w_k1_KO_s013
                                 + 0.62107 w_k1_MAT_s013
                                 + 0.836479 w_k1_MSFT_s013
                                 + 0.825945 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.05498 w_k1_ADM_s013
                                 + 1.18918 w_k1_AXP_s013
                                 + 1.04563 w_k1_DIS_s013
                                 + 1.10346 w_k1_FDX_s013
                                 + 0.978183 w_k1_IBM_s013
                                 + 1.10645 w_k1_INTC_s013
                                 + 0.920075 w_k1_KO_s013
                                 + 1.00945 w_k1_MAT_s013
                                 + 1.05286 w_k1_MSFT_s013
                                 + 0.944261 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.907522 w_k1_ADM_s013
                                 + 0.770264 w_k1_AXP_s013
                                 + 0.826296 w_k1_DIS_s013
                                 + 0.821716 w_k1_FDX_s013
                                 + 0.981318 w_k1_IBM_s013
                                 + 1.09249 w_k1_INTC_s013
                                 + 0.965315 w_k1_KO_s013
                                 + 0.888287 w_k1_MAT_s013
                                 + 0.893191 w_k1_MSFT_s013
                                 + 0.889036 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.33807 w_k1_ADM_s013
                                 + 1.21671 w_k1_AXP_s013
                                 + 1.07114 w_k1_DIS_s013
                                 + 1.28923 w_k1_FDX_s013
                                 + 0.801085 w_k1_IBM_s013
                                 + 1.3604 w_k1_INTC_s013 + 1.04796 w_k1_KO_s013
                                 + 0.974084 w_k1_MAT_s013
                                 + 1.35183 w_k1_MSFT_s013
                                 + 1.06494 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.43016 w_k1_ADM_s013
                                 + 1.63069 w_k1_AXP_s013
                                 + 1.87689 w_k1_DIS_s013
                                 + 1.60967 w_k1_FDX_s013
                                 + 1.33018 w_k1_IBM_s013
                                 + 1.40106 w_k1_INTC_s013
                                 + 1.35596 w_k1_KO_s013 + 1.36048 w_k1_MAT_s013
                                 + 1.37631 w_k1_MSFT_s013
                                 + 1.27081 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.01858 w_k1_ADM_s013
                                 + 0.858141 w_k1_AXP_s013
                                 + 0.907732 w_k1_DIS_s013
                                 + 0.878438 w_k1_FDX_s013
                                 + 1.28761 w_k1_IBM_s013
                                 + 0.891149 w_k1_INTC_s013
                                 + 0.937424 w_k1_KO_s013
                                 + 1.15682 w_k1_MAT_s013
                                 + 0.885105 w_k1_MSFT_s013
                                 + 1.04128 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.20102 w_k1_ADM_s014
                                 + 1.48734 w_k1_AXP_s014
                                 + 1.61838 w_k1_DIS_s014
                                 + 1.32549 w_k1_FDX_s014
                                 + 1.37361 w_k1_IBM_s014
                                 + 1.11158 w_k1_INTC_s014
                                 + 1.40594 w_k1_KO_s014 + 1.74854 w_k1_MAT_s014
                                 + 1.26247 w_k1_MSFT_s014
                                 + 1.21223 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.18429 w_k1_ADM_s014
                                 + 0.878592 w_k1_AXP_s014
                                 + 1.39331 w_k1_DIS_s014
                                 + 1.05625 w_k1_FDX_s014
                                 + 1.04156 w_k1_IBM_s014
                                 + 1.10026 w_k1_INTC_s014
                                 + 1.24873 w_k1_KO_s014 + 1.40773 w_k1_MAT_s014
                                 + 1.08321 w_k1_MSFT_s014
                                 + 1.05064 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.94175 w_k1_ADM_s014
                                 + 0.732178 w_k1_AXP_s014
                                 + 0.785103 w_k1_DIS_s014
                                 + 0.830322 w_k1_FDX_s014
                                 + 0.652509 w_k1_IBM_s014
                                 + 0.895978 w_k1_INTC_s014
                                 + 0.821346 w_k1_KO_s014
                                 + 0.583676 w_k1_MAT_s014
                                 + 0.826257 w_k1_MSFT_s014
                                 + 0.824559 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.42662 w_k1_ADM_s014
                                 + 1.30137 w_k1_AXP_s014
                                 + 1.04267 w_k1_DIS_s014
                                 + 1.02446 w_k1_FDX_s014
                                 + 1.45127 w_k1_IBM_s014
                                 + 0.930955 w_k1_INTC_s014
                                 + 1.28798 w_k1_KO_s014 + 1.83962 w_k1_MAT_s014
                                 + 0.883905 w_k1_MSFT_s014
                                 + 1.06221 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.861013 w_k1_ADM_s014
                                 + 0.819272 w_k1_AXP_s014
                                 + 0.636272 w_k1_DIS_s014
                                 + 0.769847 w_k1_FDX_s014
                                 + 0.723313 w_k1_IBM_s014
                                 + 0.867727 w_k1_INTC_s014
                                 + 0.724802 w_k1_KO_s014
                                 + 1.26735 w_k1_MAT_s014
                                 + 0.760967 w_k1_MSFT_s014
                                 + 0.808745 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.09594 w_k1_ADM_s014
                                 + 1.01541 w_k1_AXP_s014
                                 + 1.02156 w_k1_DIS_s014
                                 + 1.14398 w_k1_FDX_s014
                                 + 0.748419 w_k1_IBM_s014
                                 + 1.10847 w_k1_INTC_s014
                                 + 0.892144 w_k1_KO_s014
                                 + 0.889326 w_k1_MAT_s014
                                 + 1.03658 w_k1_MSFT_s014
                                 + 0.915208 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.914883 w_k1_ADM_s014
                                 + 0.909363 w_k1_AXP_s014
                                 + 0.812676 w_k1_DIS_s014
                                 + 0.781447 w_k1_FDX_s014
                                 + 1.09162 w_k1_IBM_s014
                                 + 0.899635 w_k1_INTC_s014
                                 + 0.98315 w_k1_KO_s014 + 1.35895 w_k1_MAT_s014
                                 + 0.841152 w_k1_MSFT_s014
                                 + 0.892093 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.97427 w_k1_ADM_s014
                                 + 0.961885 w_k1_AXP_s014
                                 + 0.987813 w_k1_DIS_s014
                                 + 0.919848 w_k1_FDX_s014
                                 + 0.994035 w_k1_IBM_s014
                                 + 1.0372 w_k1_INTC_s014 + 1.01366 w_k1_KO_s014
                                 + 1.25602 w_k1_MAT_s014
                                 + 1.00478 w_k1_MSFT_s014
                                 + 1.00927 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.969729 w_k1_ADM_s015
                                 + 0.899602 w_k1_AXP_s015
                                 + 0.782145 w_k1_DIS_s015
                                 + 0.792342 w_k1_FDX_s015
                                 + 1.5289 w_k1_IBM_s015
                                 + 1.23589 w_k1_INTC_s015
                                 + 1.20449 w_k1_KO_s015 + 1.72983 w_k1_MAT_s015
                                 + 0.882231 w_k1_MSFT_s015
                                 + 0.990205 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.740001 w_k1_ADM_s015
                                 + 0.573481 w_k1_AXP_s015
                                 + 0.657001 w_k1_DIS_s015
                                 + 0.809203 w_k1_FDX_s015
                                 + 0.720342 w_k1_IBM_s015
                                 + 0.930168 w_k1_INTC_s015
                                 + 0.746002 w_k1_KO_s015
                                 + 0.679187 w_k1_MAT_s015
                                 + 0.839544 w_k1_MSFT_s015
                                 + 0.831301 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.09721 w_k1_ADM_s015
                                 + 1.50154 w_k1_AXP_s015
                                 + 0.903885 w_k1_DIS_s015
                                 + 1.18818 w_k1_FDX_s015
                                 + 1.24911 w_k1_IBM_s015
                                 + 1.35002 w_k1_INTC_s015
                                 + 1.14443 w_k1_KO_s015 + 1.41631 w_k1_MAT_s015
                                 + 1.04497 w_k1_MSFT_s015 + 1.0917 w_k1_PG_s015
                                 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.19062 w_k1_ADM_s015
                                 + 1.46593 w_k1_AXP_s015
                                 + 1.93515 w_k1_DIS_s015
                                 + 1.50663 w_k1_FDX_s015
                                 + 0.918576 w_k1_IBM_s015
                                 + 1.83139 w_k1_INTC_s015
                                 + 1.53277 w_k1_KO_s015 + 1.46325 w_k1_MAT_s015
                                 + 1.58764 w_k1_MSFT_s015
                                 + 1.40194 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.99775 w_k1_ADM_s015
                                 + 0.999158 w_k1_AXP_s015
                                 + 0.982379 w_k1_DIS_s015
                                 + 1.12438 w_k1_FDX_s015
                                 + 0.689233 w_k1_IBM_s015
                                 + 1.09571 w_k1_INTC_s015
                                 + 0.995717 w_k1_KO_s015
                                 + 0.946391 w_k1_MAT_s015
                                 + 0.99877 w_k1_MSFT_s015
                                 + 1.03023 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.05272 w_k1_ADM_s015
                                 + 0.878455 w_k1_AXP_s015
                                 + 1.09787 w_k1_DIS_s015
                                 + 1.13688 w_k1_FDX_s015
                                 + 0.914938 w_k1_IBM_s015
                                 + 1.2047 w_k1_INTC_s015 + 1.13389 w_k1_KO_s015
                                 + 0.866217 w_k1_MAT_s015
                                 + 1.15824 w_k1_MSFT_s015
                                 + 0.87741 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.25761 w_k1_ADM_s015
                                 + 0.997218 w_k1_AXP_s015
                                 + 1.28636 w_k1_DIS_s015
                                 + 0.961776 w_k1_FDX_s015
                                 + 1.54671 w_k1_IBM_s015
                                 + 1.10764 w_k1_INTC_s015
                                 + 1.34533 w_k1_KO_s015 + 1.39467 w_k1_MAT_s015
                                 + 0.991757 w_k1_MSFT_s015
                                 + 1.26186 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.754838 w_k1_ADM_s015
                                 + 0.756943 w_k1_AXP_s015
                                 + 0.725379 w_k1_DIS_s015
                                 + 0.889872 w_k1_FDX_s015
                                 + 0.955083 w_k1_IBM_s015
                                 + 1.10094 w_k1_INTC_s015
                                 + 0.95952 w_k1_KO_s015 + 1.14911 w_k1_MAT_s015
                                 + 0.907632 w_k1_MSFT_s015
                                 + 1.01765 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.982654 w_k1_ADM_s016
                                 + 0.922543 w_k1_AXP_s016
                                 + 0.828467 w_k1_DIS_s016
                                 + 0.89288 w_k1_FDX_s016
                                 + 0.839316 w_k1_IBM_s016
                                 + 1.09712 w_k1_INTC_s016
                                 + 0.929185 w_k1_KO_s016
                                 + 0.584053 w_k1_MAT_s016
                                 + 0.912995 w_k1_MSFT_s016
                                 + 0.985507 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.705142 w_k1_ADM_s016
                                 + 0.771115 w_k1_AXP_s016
                                 + 0.562633 w_k1_DIS_s016
                                 + 0.767391 w_k1_FDX_s016
                                 + 0.741575 w_k1_IBM_s016
                                 + 0.754728 w_k1_INTC_s016
                                 + 0.6478 w_k1_KO_s016 + 0.649531 w_k1_MAT_s016
                                 + 0.690822 w_k1_MSFT_s016
                                 + 0.900828 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.09019 w_k1_ADM_s016
                                 + 1.05806 w_k1_AXP_s016 + 1.3878 w_k1_DIS_s016
                                 + 1.09079 w_k1_FDX_s016
                                 + 1.16806 w_k1_IBM_s016
                                 + 1.30792 w_k1_INTC_s016
                                 + 1.20881 w_k1_KO_s016
                                 + 0.871273 w_k1_MAT_s016
                                 + 1.15623 w_k1_MSFT_s016
                                 + 1.11874 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.72642 w_k1_ADM_s016
                                 + 0.730968 w_k1_AXP_s016
                                 + 0.908577 w_k1_DIS_s016
                                 + 0.823515 w_k1_FDX_s016
                                 + 0.682944 w_k1_IBM_s016
                                 + 1.0353 w_k1_INTC_s016
                                 + 0.843337 w_k1_KO_s016
                                 + 0.768161 w_k1_MAT_s016
                                 + 0.970533 w_k1_MSFT_s016
                                 + 0.83499 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.93374 w_k1_ADM_s016
                                 + 1.17212 w_k1_AXP_s016 + 1.006 w_k1_DIS_s016
                                 + 1.13436 w_k1_FDX_s016
                                 + 0.973626 w_k1_IBM_s016
                                 + 1.0412 w_k1_INTC_s016
                                 + 0.838018 w_k1_KO_s016
                                 + 0.908318 w_k1_MAT_s016
                                 + 0.950244 w_k1_MSFT_s016
                                 + 1.00107 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.863355 w_k1_ADM_s016
                                 + 1.31635 w_k1_AXP_s016
                                 + 0.93435 w_k1_DIS_s016
                                 + 1.07782 w_k1_FDX_s016
                                 + 1.20345 w_k1_IBM_s016
                                 + 0.985402 w_k1_INTC_s016
                                 + 0.901858 w_k1_KO_s016
                                 + 1.16446 w_k1_MAT_s016
                                 + 0.893717 w_k1_MSFT_s016
                                 + 1.0239 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.658374 w_k1_ADM_s016
                                 + 1.0957 w_k1_AXP_s016 + 1.01339 w_k1_DIS_s016
                                 + 0.939158 w_k1_FDX_s016
                                 + 1.42431 w_k1_IBM_s016
                                 + 1.14139 w_k1_INTC_s016 + 1.056 w_k1_KO_s016
                                 + 1.46212 w_k1_MAT_s016
                                 + 0.849049 w_k1_MSFT_s016
                                 + 1.0574 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.977964 w_k1_ADM_s016
                                 + 1.24303 w_k1_AXP_s016
                                 + 1.42997 w_k1_DIS_s016
                                 + 1.28355 w_k1_FDX_s016
                                 + 0.946405 w_k1_IBM_s016
                                 + 1.22651 w_k1_INTC_s016
                                 + 1.04988 w_k1_KO_s016
                                 + 0.972118 w_k1_MAT_s016
                                 + 1.20725 w_k1_MSFT_s016
                                 + 1.06012 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.666418 w_k1_ADM_s017
                                 + 0.787285 w_k1_AXP_s017
                                 + 0.765781 w_k1_DIS_s017
                                 + 0.812043 w_k1_FDX_s017
                                 + 0.979618 w_k1_IBM_s017
                                 + 1.03671 w_k1_INTC_s017
                                 + 0.921924 w_k1_KO_s017
                                 + 0.94775 w_k1_MAT_s017
                                 + 0.987866 w_k1_MSFT_s017
                                 + 0.810674 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.906615 w_k1_ADM_s017
                                 + 0.860758 w_k1_AXP_s017
                                 + 0.92926 w_k1_DIS_s017
                                 + 0.945995 w_k1_FDX_s017
                                 + 0.675226 w_k1_IBM_s017
                                 + 0.948151 w_k1_INTC_s017
                                 + 0.762264 w_k1_KO_s017
                                 + 0.773195 w_k1_MAT_s017
                                 + 1.06797 w_k1_MSFT_s017
                                 + 0.881813 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.08787 w_k1_ADM_s017
                                 + 1.39696 w_k1_AXP_s017
                                 + 1.46768 w_k1_DIS_s017 + 1.1005 w_k1_FDX_s017
                                 + 1.85406 w_k1_IBM_s017
                                 + 1.26678 w_k1_INTC_s017
                                 + 1.46278 w_k1_KO_s017 + 2.1242 w_k1_MAT_s017
                                 + 1.12393 w_k1_MSFT_s017
                                 + 1.25159 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.17202 w_k1_ADM_s017
                                 + 1.06067 w_k1_AXP_s017
                                 + 1.41425 w_k1_DIS_s017
                                 + 1.15323 w_k1_FDX_s017
                                 + 1.18274 w_k1_IBM_s017
                                 + 1.33052 w_k1_INTC_s017
                                 + 1.21197 w_k1_KO_s017 + 1.20231 w_k1_MAT_s017
                                 + 1.36396 w_k1_MSFT_s017
                                 + 1.10618 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.01304 w_k1_ADM_s017
                                 + 1.12413 w_k1_AXP_s017 + 1.0164 w_k1_DIS_s017
                                 + 1.16133 w_k1_FDX_s017
                                 + 0.78954 w_k1_IBM_s017
                                 + 1.09162 w_k1_INTC_s017
                                 + 0.964875 w_k1_KO_s017
                                 + 0.880077 w_k1_MAT_s017
                                 + 1.10439 w_k1_MSFT_s017
                                 + 0.866328 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.06317 w_k1_ADM_s017
                                 + 1.27111 w_k1_AXP_s017
                                 + 1.16631 w_k1_DIS_s017
                                 + 1.09097 w_k1_FDX_s017
                                 + 0.977286 w_k1_IBM_s017
                                 + 1.12044 w_k1_INTC_s017
                                 + 1.17844 w_k1_KO_s017 + 1.28054 w_k1_MAT_s017
                                 + 1.19861 w_k1_MSFT_s017
                                 + 0.959041 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.942001 w_k1_ADM_s017
                                 + 1.19812 w_k1_AXP_s017
                                 + 1.03483 w_k1_DIS_s017
                                 + 1.03422 w_k1_FDX_s017 + 1.3608 w_k1_IBM_s017
                                 + 1.18378 w_k1_INTC_s017
                                 + 1.14325 w_k1_KO_s017 + 1.39001 w_k1_MAT_s017
                                 + 0.954267 w_k1_MSFT_s017
                                 + 1.12649 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.739819 w_k1_ADM_s017
                                 + 0.685996 w_k1_AXP_s017
                                 + 0.544637 w_k1_DIS_s017
                                 + 0.667958 w_k1_FDX_s017
                                 + 0.647644 w_k1_IBM_s017
                                 + 0.775713 w_k1_INTC_s017
                                 + 0.593331 w_k1_KO_s017
                                 + 0.65945 w_k1_MAT_s017
                                 + 0.734214 w_k1_MSFT_s017
                                 + 0.669951 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.946159 w_k1_ADM_s100
                                 + 0.80311 w_k1_AXP_s100
                                 + 0.745448 w_k1_DIS_s100
                                 + 0.760385 w_k1_FDX_s100
                                 + 0.96744 w_k1_IBM_s100
                                 + 0.913757 w_k1_INTC_s100
                                 + 0.91579 w_k1_KO_s100 + 0.92886 w_k1_MAT_s100
                                 + 0.898198 w_k1_MSFT_s100
                                 + 0.886063 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.681624 w_k1_ADM_s100
                                 + 0.592036 w_k1_AXP_s100
                                 + 0.634472 w_k1_DIS_s100
                                 + 0.731891 w_k1_FDX_s100
                                 + 0.580419 w_k1_IBM_s100
                                 + 0.909024 w_k1_INTC_s100
                                 + 0.649731 w_k1_KO_s100
                                 + 0.550179 w_k1_MAT_s100
                                 + 0.702552 w_k1_MSFT_s100
                                 + 0.731917 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.876014 w_k1_ADM_s100
                                 + 1.05568 w_k1_AXP_s100
                                 + 1.20998 w_k1_DIS_s100
                                 + 0.871145 w_k1_FDX_s100
                                 + 1.66948 w_k1_IBM_s100
                                 + 0.908549 w_k1_INTC_s100
                                 + 1.18106 w_k1_KO_s100 + 1.90188 w_k1_MAT_s100
                                 + 0.905822 w_k1_MSFT_s100
                                 + 1.07597 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.980892 w_k1_ADM_s100
                                 + 0.938696 w_k1_AXP_s100
                                 + 1.11191 w_k1_DIS_s100
                                 + 1.05884 w_k1_FDX_s100
                                 + 0.652802 w_k1_IBM_s100
                                 + 1.15043 w_k1_INTC_s100
                                 + 0.798075 w_k1_KO_s100
                                 + 0.854088 w_k1_MAT_s100
                                 + 1.12365 w_k1_MSFT_s100
                                 + 1.01663 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.16469 w_k1_ADM_s100
                                 + 1.2677 w_k1_AXP_s100 + 1.37886 w_k1_DIS_s100
                                 + 1.11427 w_k1_FDX_s100
                                 + 1.15789 w_k1_IBM_s100
                                 + 0.887035 w_k1_INTC_s100
                                 + 1.31944 w_k1_KO_s100 + 1.47025 w_k1_MAT_s100
                                 + 1.06399 w_k1_MSFT_s100
                                 + 1.18515 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.30463 w_k1_ADM_s100
                                 + 1.44845 w_k1_AXP_s100
                                 + 2.01633 w_k1_DIS_s100
                                 + 1.34909 w_k1_FDX_s100
                                 + 1.16735 w_k1_IBM_s100
                                 + 1.3996 w_k1_INTC_s100 + 1.22407 w_k1_KO_s100
                                 + 1.44362 w_k1_MAT_s100
                                 + 1.33959 w_k1_MSFT_s100
                                 + 1.18765 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.05548 w_k1_ADM_s100
                                 + 1.0863 w_k1_AXP_s100 + 1.14917 w_k1_DIS_s100
                                 + 1.05039 w_k1_FDX_s100
                                 + 0.980401 w_k1_IBM_s100
                                 + 0.98378 w_k1_INTC_s100
                                 + 1.01639 w_k1_KO_s100 + 1.34169 w_k1_MAT_s100
                                 + 1.01382 w_k1_MSFT_s100
                                 + 1.17107 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.841803 w_k1_ADM_s100
                                 + 0.739865 w_k1_AXP_s100
                                 + 0.920512 w_k1_DIS_s100
                                 + 0.89625 w_k1_FDX_s100
                                 + 0.602103 w_k1_IBM_s100
                                 + 0.907785 w_k1_INTC_s100
                                 + 0.829478 w_k1_KO_s100
                                 + 0.682815 w_k1_MAT_s100
                                 + 0.893938 w_k1_MSFT_s100
                                 + 0.866398 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.38594 w_k1_ADM_s101
                                 + 1.16014 w_k1_AXP_s101 + 1.56 w_k1_DIS_s101
                                 + 1.23282 w_k1_FDX_s101
                                 + 1.03741 w_k1_IBM_s101
                                 + 1.28249 w_k1_INTC_s101
                                 + 1.19128 w_k1_KO_s101 + 1.11302 w_k1_MAT_s101
                                 + 1.48378 w_k1_MSFT_s101
                                 + 1.11256 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.0317 w_k1_ADM_s101
                                 + 1.5291 w_k1_AXP_s101 + 1.54417 w_k1_DIS_s101
                                 + 1.24814 w_k1_FDX_s101
                                 + 1.75783 w_k1_IBM_s101
                                 + 1.30219 w_k1_INTC_s101
                                 + 1.49817 w_k1_KO_s101 + 1.69795 w_k1_MAT_s101
                                 + 1.32903 w_k1_MSFT_s101
                                 + 1.37999 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.854489 w_k1_ADM_s101
                                 + 0.676275 w_k1_AXP_s101
                                 + 0.565416 w_k1_DIS_s101
                                 + 0.741612 w_k1_FDX_s101
                                 + 0.860502 w_k1_IBM_s101
                                 + 0.819571 w_k1_INTC_s101
                                 + 0.848362 w_k1_KO_s101
                                 + 0.837407 w_k1_MAT_s101
                                 + 0.792332 w_k1_MSFT_s101
                                 + 0.885031 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.996508 w_k1_ADM_s101
                                 + 0.642072 w_k1_AXP_s101
                                 + 0.949361 w_k1_DIS_s101
                                 + 0.869997 w_k1_FDX_s101
                                 + 0.846157 w_k1_IBM_s101
                                 + 1.02426 w_k1_INTC_s101
                                 + 0.874831 w_k1_KO_s101
                                 + 0.460383 w_k1_MAT_s101
                                 + 0.955128 w_k1_MSFT_s101
                                 + 0.946285 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.966644 w_k1_ADM_s101
                                 + 0.884677 w_k1_AXP_s101
                                 + 0.984833 w_k1_DIS_s101
                                 + 0.910848 w_k1_FDX_s101
                                 + 1.34244 w_k1_IBM_s101
                                 + 1.02032 w_k1_INTC_s101
                                 + 1.19602 w_k1_KO_s101 + 1.2367 w_k1_MAT_s101
                                 + 0.950787 w_k1_MSFT_s101
                                 + 1.04735 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.3562 w_k1_ADM_s101
                                 + 1.29169 w_k1_AXP_s101
                                 + 1.28561 w_k1_DIS_s101
                                 + 1.24461 w_k1_FDX_s101
                                 + 1.00085 w_k1_IBM_s101
                                 + 1.19941 w_k1_INTC_s101
                                 + 1.12722 w_k1_KO_s101 + 1.28416 w_k1_MAT_s101
                                 + 1.09593 w_k1_MSFT_s101 + 1.1435 w_k1_PG_s101
                                 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.07971 w_k1_ADM_s101
                                 + 0.935329 w_k1_AXP_s101
                                 + 0.896014 w_k1_DIS_s101
                                 + 0.985254 w_k1_FDX_s101
                                 + 0.96261 w_k1_IBM_s101
                                 + 0.896086 w_k1_INTC_s101
                                 + 0.861809 w_k1_KO_s101
                                 + 0.773768 w_k1_MAT_s101
                                 + 0.902689 w_k1_MSFT_s101
                                 + 0.918021 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.15853 w_k1_ADM_s101
                                 + 1.00693 w_k1_AXP_s101
                                 + 1.18504 w_k1_DIS_s101
                                 + 1.08349 w_k1_FDX_s101
                                 + 0.819206 w_k1_IBM_s101
                                 + 1.10346 w_k1_INTC_s101
                                 + 1.07725 w_k1_KO_s101
                                 + 0.730853 w_k1_MAT_s101
                                 + 1.11039 w_k1_MSFT_s101
                                 + 1.04442 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.992632 w_k1_ADM_s102
                                 + 1.05052 w_k1_AXP_s102 + 1.2071 w_k1_DIS_s102
                                 + 1.11203 w_k1_FDX_s102
                                 + 1.78643 w_k1_IBM_s102
                                 + 1.10049 w_k1_INTC_s102
                                 + 1.45147 w_k1_KO_s102 + 1.78903 w_k1_MAT_s102
                                 + 1.26464 w_k1_MSFT_s102 + 1.1731 w_k1_PG_s102
                                 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.796279 w_k1_ADM_s102
                                 + 0.755434 w_k1_AXP_s102
                                 + 0.760654 w_k1_DIS_s102
                                 + 0.855045 w_k1_FDX_s102
                                 + 0.609718 w_k1_IBM_s102
                                 + 1.01477 w_k1_INTC_s102
                                 + 0.809991 w_k1_KO_s102
                                 + 0.542876 w_k1_MAT_s102
                                 + 1.00189 w_k1_MSFT_s102
                                 + 0.810751 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.971319 w_k1_ADM_s102
                                 + 1.12746 w_k1_AXP_s102
                                 + 1.12941 w_k1_DIS_s102
                                 + 1.06318 w_k1_FDX_s102
                                 + 0.788446 w_k1_IBM_s102
                                 + 1.0874 w_k1_INTC_s102 + 1.08306 w_k1_KO_s102
                                 + 0.894305 w_k1_MAT_s102
                                 + 1.24812 w_k1_MSFT_s102
                                 + 1.00007 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.698358 w_k1_ADM_s102
                                 + 0.838496 w_k1_AXP_s102
                                 + 0.913439 w_k1_DIS_s102
                                 + 0.880964 w_k1_FDX_s102
                                 + 1.37703 w_k1_IBM_s102
                                 + 0.893342 w_k1_INTC_s102
                                 + 1.11075 w_k1_KO_s102 + 1.53913 w_k1_MAT_s102
                                 + 0.968547 w_k1_MSFT_s102
                                 + 0.966677 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.791885 w_k1_ADM_s102
                                 + 0.921226 w_k1_AXP_s102
                                 + 0.705864 w_k1_DIS_s102
                                 + 0.866298 w_k1_FDX_s102
                                 + 1.02539 w_k1_IBM_s102
                                 + 0.977924 w_k1_INTC_s102
                                 + 0.971863 w_k1_KO_s102
                                 + 1.18251 w_k1_MAT_s102
                                 + 0.959533 w_k1_MSFT_s102
                                 + 0.934514 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.914208 w_k1_ADM_s102
                                 + 0.736028 w_k1_AXP_s102
                                 + 0.993142 w_k1_DIS_s102
                                 + 0.845439 w_k1_FDX_s102
                                 + 0.977179 w_k1_IBM_s102
                                 + 0.94126 w_k1_INTC_s102
                                 + 1.10762 w_k1_KO_s102 + 1.00671 w_k1_MAT_s102
                                 + 1.16526 w_k1_MSFT_s102
                                 + 0.901484 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.877622 w_k1_ADM_s102
                                 + 1.14884 w_k1_AXP_s102
                                 + 1.46278 w_k1_DIS_s102 + 1.2372 w_k1_FDX_s102
                                 + 0.89539 w_k1_IBM_s102
                                 + 1.07284 w_k1_INTC_s102
                                 + 1.14286 w_k1_KO_s102
                                 + 0.979049 w_k1_MAT_s102
                                 + 1.48825 w_k1_MSFT_s102
                                 + 1.04083 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.02514 w_k1_ADM_s102
                                 + 1.2832 w_k1_AXP_s102 + 1.63864 w_k1_DIS_s102
                                 + 1.24925 w_k1_FDX_s102
                                 + 1.23706 w_k1_IBM_s102
                                 + 1.08326 w_k1_INTC_s102
                                 + 1.52362 w_k1_KO_s102 + 1.70037 w_k1_MAT_s102
                                 + 1.33394 w_k1_MSFT_s102
                                 + 1.29139 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.22116 w_k1_ADM_s103
                                 + 1.15802 w_k1_AXP_s103
                                 + 1.25659 w_k1_DIS_s103
                                 + 1.11792 w_k1_FDX_s103
                                 + 0.978778 w_k1_IBM_s103
                                 + 1.16141 w_k1_INTC_s103
                                 + 1.13127 w_k1_KO_s103
                                 + 0.700323 w_k1_MAT_s103
                                 + 1.03547 w_k1_MSFT_s103
                                 + 1.00379 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.772484 w_k1_ADM_s103
                                 + 0.703134 w_k1_AXP_s103
                                 + 0.67556 w_k1_DIS_s103
                                 + 0.687473 w_k1_FDX_s103
                                 + 0.755624 w_k1_IBM_s103
                                 + 0.771282 w_k1_INTC_s103
                                 + 0.804201 w_k1_KO_s103
                                 + 0.600325 w_k1_MAT_s103
                                 + 0.747902 w_k1_MSFT_s103
                                 + 0.813058 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.45298 w_k1_ADM_s103
                                 + 1.34597 w_k1_AXP_s103 + 1.387 w_k1_DIS_s103
                                 + 1.32984 w_k1_FDX_s103
                                 + 0.817762 w_k1_IBM_s103
                                 + 1.2805 w_k1_INTC_s103 + 1.05678 w_k1_KO_s103
                                 + 0.776012 w_k1_MAT_s103
                                 + 1.19552 w_k1_MSFT_s103
                                 + 1.10054 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.32391 w_k1_ADM_s103
                                 + 1.44091 w_k1_AXP_s103
                                 + 1.44892 w_k1_DIS_s103
                                 + 1.24643 w_k1_FDX_s103 + 1.3526 w_k1_IBM_s103
                                 + 1.55377 w_k1_INTC_s103
                                 + 1.37549 w_k1_KO_s103
                                 + 0.822994 w_k1_MAT_s103
                                 + 1.18264 w_k1_MSFT_s103
                                 + 1.17731 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.02711 w_k1_ADM_s103
                                 + 0.808582 w_k1_AXP_s103
                                 + 0.785691 w_k1_DIS_s103
                                 + 0.843712 w_k1_FDX_s103
                                 + 0.864551 w_k1_IBM_s103
                                 + 0.894611 w_k1_INTC_s103
                                 + 0.866187 w_k1_KO_s103
                                 + 0.592166 w_k1_MAT_s103
                                 + 0.83577 w_k1_MSFT_s103
                                 + 0.808547 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.12499 w_k1_ADM_s103
                                 + 0.992935 w_k1_AXP_s103
                                 + 1.02513 w_k1_DIS_s103
                                 + 1.04393 w_k1_FDX_s103
                                 + 0.885761 w_k1_IBM_s103
                                 + 1.08286 w_k1_INTC_s103 + 1.0108 w_k1_KO_s103
                                 + 0.555196 w_k1_MAT_s103
                                 + 0.953169 w_k1_MSFT_s103
                                 + 0.962557 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.09617 w_k1_ADM_s103
                                 + 1.27536 w_k1_AXP_s103
                                 + 1.24348 w_k1_DIS_s103
                                 + 1.05394 w_k1_FDX_s103
                                 + 1.62902 w_k1_IBM_s103
                                 + 1.17342 w_k1_INTC_s103 + 1.2847 w_k1_KO_s103
                                 + 1.28268 w_k1_MAT_s103
                                 + 0.948727 w_k1_MSFT_s103
                                 + 1.10539 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.893578 w_k1_ADM_s103
                                 + 1.01883 w_k1_AXP_s103
                                 + 0.997694 w_k1_DIS_s103
                                 + 0.878996 w_k1_FDX_s103
                                 + 1.21973 w_k1_IBM_s103
                                 + 1.04321 w_k1_INTC_s103 + 1.0838 w_k1_KO_s103
                                 + 1.06282 w_k1_MAT_s103
                                 + 0.847632 w_k1_MSFT_s103
                                 + 0.924774 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.986807 w_k1_ADM_s104
                                 + 1.38534 w_k1_AXP_s104
                                 + 1.24059 w_k1_DIS_s104
                                 + 1.12321 w_k1_FDX_s104 + 1.1785 w_k1_IBM_s104
                                 + 0.99001 w_k1_INTC_s104
                                 + 1.15045 w_k1_KO_s104 + 1.4732 w_k1_MAT_s104
                                 + 1.02885 w_k1_MSFT_s104
                                 + 1.08704 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.14813 w_k1_ADM_s104
                                 + 1.16039 w_k1_AXP_s104
                                 + 0.931899 w_k1_DIS_s104
                                 + 1.04614 w_k1_FDX_s104
                                 + 0.758192 w_k1_IBM_s104
                                 + 0.84131 w_k1_INTC_s104
                                 + 0.937892 w_k1_KO_s104
                                 + 0.967803 w_k1_MAT_s104
                                 + 0.99341 w_k1_MSFT_s104
                                 + 0.987491 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.95224 w_k1_ADM_s104
                                 + 0.83933 w_k1_AXP_s104 + 1.0081 w_k1_DIS_s104
                                 + 0.964718 w_k1_FDX_s104
                                 + 1.36593 w_k1_IBM_s104
                                 + 0.78451 w_k1_INTC_s104
                                 + 1.38981 w_k1_KO_s104 + 1.27964 w_k1_MAT_s104
                                 + 0.956773 w_k1_MSFT_s104
                                 + 0.98487 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.03838 w_k1_ADM_s104
                                 + 0.952543 w_k1_AXP_s104
                                 + 1.32198 w_k1_DIS_s104
                                 + 1.21592 w_k1_FDX_s104
                                 + 0.912012 w_k1_IBM_s104
                                 + 0.990749 w_k1_INTC_s104
                                 + 1.15711 w_k1_KO_s104
                                 + 0.904461 w_k1_MAT_s104
                                 + 1.16716 w_k1_MSFT_s104
                                 + 1.03896 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.904952 w_k1_ADM_s104
                                 + 0.786226 w_k1_AXP_s104
                                 + 0.679323 w_k1_DIS_s104
                                 + 0.825021 w_k1_FDX_s104
                                 + 0.864048 w_k1_IBM_s104
                                 + 0.792879 w_k1_INTC_s104
                                 + 0.734835 w_k1_KO_s104
                                 + 0.683748 w_k1_MAT_s104
                                 + 0.6837 w_k1_MSFT_s104
                                 + 0.812922 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.876164 w_k1_ADM_s104
                                 + 0.773338 w_k1_AXP_s104
                                 + 0.904005 w_k1_DIS_s104
                                 + 0.864428 w_k1_FDX_s104
                                 + 0.871812 w_k1_IBM_s104
                                 + 0.844142 w_k1_INTC_s104
                                 + 0.988263 w_k1_KO_s104
                                 + 1.05105 w_k1_MAT_s104
                                 + 0.898648 w_k1_MSFT_s104
                                 + 0.942764 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.962796 w_k1_ADM_s104
                                 + 0.77715 w_k1_AXP_s104
                                 + 0.882782 w_k1_DIS_s104
                                 + 1.07727 w_k1_FDX_s104
                                 + 0.58153 w_k1_IBM_s104
                                 + 0.767718 w_k1_INTC_s104
                                 + 0.761589 w_k1_KO_s104
                                 + 0.590772 w_k1_MAT_s104
                                 + 0.995831 w_k1_MSFT_s104
                                 + 0.949121 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.16244 w_k1_ADM_s104
                                 + 1.37872 w_k1_AXP_s104
                                 + 1.43685 w_k1_DIS_s104
                                 + 1.30221 w_k1_FDX_s104
                                 + 1.41971 w_k1_IBM_s104
                                 + 1.10741 w_k1_INTC_s104
                                 + 1.41877 w_k1_KO_s104 + 1.7295 w_k1_MAT_s104
                                 + 1.2375 w_k1_MSFT_s104 + 1.10265 w_k1_PG_s104
                                 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.863133 w_k1_ADM_s105
                                 + 0.917375 w_k1_AXP_s105
                                 + 0.925961 w_k1_DIS_s105
                                 + 0.890448 w_k1_FDX_s105
                                 + 1.7478 w_k1_IBM_s105
                                 + 0.82011 w_k1_INTC_s105
                                 + 1.26799 w_k1_KO_s105 + 1.92868 w_k1_MAT_s105
                                 + 0.916573 w_k1_MSFT_s105
                                 + 1.16921 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.884353 w_k1_ADM_s105
                                 + 1.00897 w_k1_AXP_s105
                                 + 1.30893 w_k1_DIS_s105
                                 + 1.14605 w_k1_FDX_s105
                                 + 0.820794 w_k1_IBM_s105
                                 + 0.927942 w_k1_INTC_s105
                                 + 1.06672 w_k1_KO_s105
                                 + 0.792528 w_k1_MAT_s105
                                 + 1.11893 w_k1_MSFT_s105
                                 + 1.20188 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.843238 w_k1_ADM_s105
                                 + 0.833204 w_k1_AXP_s105
                                 + 0.712944 w_k1_DIS_s105
                                 + 0.817366 w_k1_FDX_s105
                                 + 0.92923 w_k1_IBM_s105
                                 + 0.696654 w_k1_INTC_s105
                                 + 0.853821 w_k1_KO_s105
                                 + 0.969096 w_k1_MAT_s105
                                 + 0.780408 w_k1_MSFT_s105
                                 + 0.882607 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.04424 w_k1_ADM_s105
                                 + 1.07404 w_k1_AXP_s105
                                 + 0.932774 w_k1_DIS_s105
                                 + 1.15219 w_k1_FDX_s105
                                 + 0.78777 w_k1_IBM_s105
                                 + 1.04853 w_k1_INTC_s105
                                 + 1.04951 w_k1_KO_s105
                                 + 0.881452 w_k1_MAT_s105
                                 + 1.12464 w_k1_MSFT_s105
                                 + 1.11548 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.875995 w_k1_ADM_s105
                                 + 1.17639 w_k1_AXP_s105
                                 + 1.32117 w_k1_DIS_s105
                                 + 1.08687 w_k1_FDX_s105
                                 + 1.19401 w_k1_IBM_s105
                                 + 1.03388 w_k1_INTC_s105
                                 + 1.38476 w_k1_KO_s105 + 1.28701 w_k1_MAT_s105
                                 + 1.12065 w_k1_MSFT_s105
                                 + 1.13655 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.824419 w_k1_ADM_s105
                                 + 0.764951 w_k1_AXP_s105
                                 + 0.765218 w_k1_DIS_s105
                                 + 0.970117 w_k1_FDX_s105
                                 + 0.510528 w_k1_IBM_s105
                                 + 0.903447 w_k1_INTC_s105
                                 + 0.811904 w_k1_KO_s105
                                 + 0.451858 w_k1_MAT_s105
                                 + 0.982871 w_k1_MSFT_s105
                                 + 0.848906 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.16436 w_k1_ADM_s105
                                 + 1.65553 w_k1_AXP_s105
                                 + 1.39809 w_k1_DIS_s105 + 1.275 w_k1_FDX_s105
                                 + 1.41177 w_k1_IBM_s105
                                 + 1.22852 w_k1_INTC_s105
                                 + 1.41018 w_k1_KO_s105 + 1.57232 w_k1_MAT_s105
                                 + 1.22546 w_k1_MSFT_s105
                                 + 1.23154 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.91864 w_k1_ADM_s105
                                 + 0.904282 w_k1_AXP_s105
                                 + 0.967274 w_k1_DIS_s105
                                 + 0.844274 w_k1_FDX_s105
                                 + 1.32944 w_k1_IBM_s105
                                 + 0.864161 w_k1_INTC_s105
                                 + 1.34285 w_k1_KO_s105 + 1.4001 w_k1_MAT_s105
                                 + 1.02586 w_k1_MSFT_s105
                                 + 1.02111 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.884266 w_k1_ADM_s106
                                 + 0.954201 w_k1_AXP_s106
                                 + 0.896981 w_k1_DIS_s106
                                 + 1.01364 w_k1_FDX_s106
                                 + 0.607689 w_k1_IBM_s106
                                 + 1.28776 w_k1_INTC_s106
                                 + 0.859783 w_k1_KO_s106
                                 + 0.622255 w_k1_MAT_s106
                                 + 1.12212 w_k1_MSFT_s106
                                 + 1.00718 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.05904 w_k1_ADM_s106
                                 + 1.20669 w_k1_AXP_s106
                                 + 0.831059 w_k1_DIS_s106
                                 + 1.10843 w_k1_FDX_s106
                                 + 0.726042 w_k1_IBM_s106
                                 + 1.19206 w_k1_INTC_s106
                                 + 0.862179 w_k1_KO_s106
                                 + 1.07642 w_k1_MAT_s106
                                 + 1.06729 w_k1_MSFT_s106
                                 + 1.05484 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.859474 w_k1_ADM_s106
                                 + 1.04961 w_k1_AXP_s106
                                 + 0.715787 w_k1_DIS_s106
                                 + 0.933215 w_k1_FDX_s106
                                 + 0.898292 w_k1_IBM_s106
                                 + 1.1892 w_k1_INTC_s106
                                 + 0.851607 w_k1_KO_s106
                                 + 0.86683 w_k1_MAT_s106
                                 + 0.83881 w_k1_MSFT_s106
                                 + 0.983391 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.996371 w_k1_ADM_s106
                                 + 1.53539 w_k1_AXP_s106
                                 + 1.47336 w_k1_DIS_s106
                                 + 1.26113 w_k1_FDX_s106
                                 + 1.30506 w_k1_IBM_s106
                                 + 1.23779 w_k1_INTC_s106
                                 + 1.25523 w_k1_KO_s106 + 1.32196 w_k1_MAT_s106
                                 + 1.25515 w_k1_MSFT_s106 + 1.1973 w_k1_PG_s106
                                 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.912207 w_k1_ADM_s106
                                 + 1.32604 w_k1_AXP_s106
                                 + 0.877738 w_k1_DIS_s106
                                 + 0.823935 w_k1_FDX_s106
                                 + 1.76173 w_k1_IBM_s106
                                 + 0.950966 w_k1_INTC_s106
                                 + 1.18727 w_k1_KO_s106 + 2.08337 w_k1_MAT_s106
                                 + 0.838982 w_k1_MSFT_s106
                                 + 1.05467 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.71121 w_k1_ADM_s106
                                 + 0.832274 w_k1_AXP_s106
                                 + 0.700259 w_k1_DIS_s106
                                 + 0.75555 w_k1_FDX_s106
                                 + 0.958027 w_k1_IBM_s106
                                 + 1.06092 w_k1_INTC_s106
                                 + 0.915291 w_k1_KO_s106
                                 + 0.874332 w_k1_MAT_s106
                                 + 0.879166 w_k1_MSFT_s106
                                 + 0.867185 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.13941 w_k1_ADM_s106
                                 + 1.35851 w_k1_AXP_s106 + 1.1337 w_k1_DIS_s106
                                 + 1.13491 w_k1_FDX_s106
                                 + 0.969282 w_k1_IBM_s106
                                 + 1.33519 w_k1_INTC_s106
                                 + 1.05861 w_k1_KO_s106
                                 + 0.843175 w_k1_MAT_s106
                                 + 1.12502 w_k1_MSFT_s106
                                 + 1.11004 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.942596 w_k1_ADM_s106
                                 + 1.14369 w_k1_AXP_s106
                                 + 0.792106 w_k1_DIS_s106
                                 + 0.875307 w_k1_FDX_s106
                                 + 1.31045 w_k1_IBM_s106
                                 + 1.0148 w_k1_INTC_s106 + 1.03995 w_k1_KO_s106
                                 + 1.30684 w_k1_MAT_s106
                                 + 0.983333 w_k1_MSFT_s106
                                 + 1.09959 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.998824 w_k1_ADM_s107
                                 + 0.768965 w_k1_AXP_s107
                                 + 0.753848 w_k1_DIS_s107
                                 + 0.865976 w_k1_FDX_s107
                                 + 1.04214 w_k1_IBM_s107
                                 + 0.832525 w_k1_INTC_s107
                                 + 0.972544 w_k1_KO_s107
                                 + 0.815545 w_k1_MAT_s107
                                 + 0.934726 w_k1_MSFT_s107
                                 + 0.911685 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.02103 w_k1_ADM_s107
                                 + 0.758032 w_k1_AXP_s107
                                 + 1.03347 w_k1_DIS_s107
                                 + 1.07315 w_k1_FDX_s107
                                 + 0.951736 w_k1_IBM_s107
                                 + 1.02048 w_k1_INTC_s107
                                 + 1.16839 w_k1_KO_s107
                                 + 0.879486 w_k1_MAT_s107
                                 + 1.11454 w_k1_MSFT_s107
                                 + 0.986213 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.24601 w_k1_ADM_s107
                                 + 1.31715 w_k1_AXP_s107
                                 + 1.67308 w_k1_DIS_s107
                                 + 1.44488 w_k1_FDX_s107
                                 + 1.15528 w_k1_IBM_s107
                                 + 1.19162 w_k1_INTC_s107
                                 + 1.43089 w_k1_KO_s107 + 1.26382 w_k1_MAT_s107
                                 + 1.26062 w_k1_MSFT_s107
                                 + 1.11515 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.938786 w_k1_ADM_s107
                                 + 0.685144 w_k1_AXP_s107
                                 + 0.876637 w_k1_DIS_s107
                                 + 0.981678 w_k1_FDX_s107
                                 + 0.670934 w_k1_IBM_s107
                                 + 0.902829 w_k1_INTC_s107
                                 + 0.896847 w_k1_KO_s107
                                 + 0.516698 w_k1_MAT_s107
                                 + 0.976793 w_k1_MSFT_s107
                                 + 0.92221 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.685629 w_k1_ADM_s107
                                 + 0.45405 w_k1_AXP_s107
                                 + 0.450312 w_k1_DIS_s107
                                 + 0.612691 w_k1_FDX_s107
                                 + 0.688241 w_k1_IBM_s107
                                 + 0.543391 w_k1_INTC_s107 + 0.659 w_k1_KO_s107
                                 + 0.775788 w_k1_MAT_s107
                                 + 0.610745 w_k1_MSFT_s107
                                 + 0.710284 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.04778 w_k1_ADM_s107
                                 + 0.998932 w_k1_AXP_s107
                                 + 1.02742 w_k1_DIS_s107
                                 + 0.96544 w_k1_FDX_s107
                                 + 1.46725 w_k1_IBM_s107
                                 + 1.17164 w_k1_INTC_s107
                                 + 1.53122 w_k1_KO_s107 + 1.35996 w_k1_MAT_s107
                                 + 0.962712 w_k1_MSFT_s107
                                 + 1.01965 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.31386 w_k1_ADM_s107
                                 + 1.32975 w_k1_AXP_s107
                                 + 1.10092 w_k1_DIS_s107
                                 + 1.21866 w_k1_FDX_s107
                                 + 0.916068 w_k1_IBM_s107
                                 + 1.16375 w_k1_INTC_s107 + 1.1366 w_k1_KO_s107
                                 + 0.879154 w_k1_MAT_s107
                                 + 1.09911 w_k1_MSFT_s107
                                 + 0.953237 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.92088 w_k1_ADM_s107
                                 + 0.633181 w_k1_AXP_s107
                                 + 0.665375 w_k1_DIS_s107
                                 + 0.796768 w_k1_FDX_s107
                                 + 0.713277 w_k1_IBM_s107
                                 + 0.753288 w_k1_INTC_s107
                                 + 0.841304 w_k1_KO_s107
                                 + 0.632946 w_k1_MAT_s107
                                 + 0.780176 w_k1_MSFT_s107
                                 + 0.74723 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.10531 w_k1_ADM_s110
                                 + 0.950378 w_k1_AXP_s110
                                 + 1.25135 w_k1_DIS_s110
                                 + 0.984091 w_k1_FDX_s110
                                 + 1.38044 w_k1_IBM_s110
                                 + 0.931354 w_k1_INTC_s110
                                 + 1.21358 w_k1_KO_s110 + 1.24729 w_k1_MAT_s110
                                 + 0.938161 w_k1_MSFT_s110
                                 + 1.18424 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.880258 w_k1_ADM_s110
                                 + 0.624972 w_k1_AXP_s110
                                 + 0.717034 w_k1_DIS_s110
                                 + 0.80875 w_k1_FDX_s110
                                 + 0.75063 w_k1_IBM_s110
                                 + 0.764183 w_k1_INTC_s110
                                 + 0.802873 w_k1_KO_s110
                                 + 0.514507 w_k1_MAT_s110
                                 + 0.816601 w_k1_MSFT_s110
                                 + 0.899209 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.931343 w_k1_ADM_s110
                                 + 1.54828 w_k1_AXP_s110
                                 + 0.948046 w_k1_DIS_s110
                                 + 0.983982 w_k1_FDX_s110
                                 + 1.44119 w_k1_IBM_s110
                                 + 0.884476 w_k1_INTC_s110
                                 + 1.18131 w_k1_KO_s110 + 1.57036 w_k1_MAT_s110
                                 + 0.823168 w_k1_MSFT_s110
                                 + 1.10743 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.10567 w_k1_ADM_s110
                                 + 1.08953 w_k1_AXP_s110
                                 + 1.37257 w_k1_DIS_s110
                                 + 1.30291 w_k1_FDX_s110
                                 + 0.854465 w_k1_IBM_s110
                                 + 1.11466 w_k1_INTC_s110
                                 + 1.07177 w_k1_KO_s110
                                 + 0.824924 w_k1_MAT_s110
                                 + 1.33193 w_k1_MSFT_s110
                                 + 1.08895 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.979677 w_k1_ADM_s110
                                 + 0.943047 w_k1_AXP_s110
                                 + 0.88255 w_k1_DIS_s110
                                 + 0.876437 w_k1_FDX_s110
                                 + 1.25571 w_k1_IBM_s110
                                 + 0.805953 w_k1_INTC_s110
                                 + 1.09136 w_k1_KO_s110 + 1.1267 w_k1_MAT_s110
                                 + 0.848776 w_k1_MSFT_s110
                                 + 1.11231 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.946077 w_k1_ADM_s110
                                 + 0.929373 w_k1_AXP_s110
                                 + 0.837686 w_k1_DIS_s110
                                 + 0.839639 w_k1_FDX_s110
                                 + 1.15769 w_k1_IBM_s110
                                 + 0.904045 w_k1_INTC_s110
                                 + 1.04113 w_k1_KO_s110 + 0.82545 w_k1_MAT_s110
                                 + 0.883921 w_k1_MSFT_s110
                                 + 1.01965 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.20821 w_k1_ADM_s110
                                 + 1.22627 w_k1_AXP_s110
                                 + 1.11738 w_k1_DIS_s110 + 1.1549 w_k1_FDX_s110
                                 + 0.832003 w_k1_IBM_s110
                                 + 1.10294 w_k1_INTC_s110
                                 + 1.01107 w_k1_KO_s110
                                 + 0.692858 w_k1_MAT_s110
                                 + 1.1057 w_k1_MSFT_s110 + 1.07645 w_k1_PG_s110
                                 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.832104 w_k1_ADM_s110
                                 + 1.2751 w_k1_AXP_s110 + 1.12742 w_k1_DIS_s110
                                 + 1.06508 w_k1_FDX_s110
                                 + 0.97788 w_k1_IBM_s110
                                 + 1.15745 w_k1_INTC_s110
                                 + 1.00336 w_k1_KO_s110
                                 + 0.993303 w_k1_MAT_s110
                                 + 1.06858 w_k1_MSFT_s110
                                 + 1.06856 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.0087 w_k1_ADM_s111
                                 + 0.822187 w_k1_AXP_s111
                                 + 1.18335 w_k1_DIS_s111
                                 + 1.02506 w_k1_FDX_s111
                                 + 1.26954 w_k1_IBM_s111
                                 + 0.805949 w_k1_INTC_s111
                                 + 1.06993 w_k1_KO_s111 + 1.32847 w_k1_MAT_s111
                                 + 0.946392 w_k1_MSFT_s111
                                 + 1.21595 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.45398 w_k1_ADM_s111
                                 + 1.30594 w_k1_AXP_s111 + 1.6411 w_k1_DIS_s111
                                 + 1.36371 w_k1_FDX_s111
                                 + 1.11392 w_k1_IBM_s111
                                 + 1.19333 w_k1_INTC_s111
                                 + 1.28897 w_k1_KO_s111 + 1.41329 w_k1_MAT_s111
                                 + 1.15726 w_k1_MSFT_s111
                                 + 1.35411 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.13253 w_k1_ADM_s111
                                 + 1.26226 w_k1_AXP_s111
                                 + 1.27975 w_k1_DIS_s111
                                 + 1.28071 w_k1_FDX_s111
                                 + 0.708048 w_k1_IBM_s111
                                 + 1.36235 w_k1_INTC_s111
                                 + 0.940227 w_k1_KO_s111
                                 + 0.766272 w_k1_MAT_s111
                                 + 1.29809 w_k1_MSFT_s111
                                 + 1.10595 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.739959 w_k1_ADM_s111
                                 + 0.848837 w_k1_AXP_s111
                                 + 0.788891 w_k1_DIS_s111
                                 + 0.797442 w_k1_FDX_s111
                                 + 1.39054 w_k1_IBM_s111
                                 + 0.935048 w_k1_INTC_s111
                                 + 0.935982 w_k1_KO_s111
                                 + 1.41446 w_k1_MAT_s111
                                 + 0.914728 w_k1_MSFT_s111
                                 + 1.08842 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.773872 w_k1_ADM_s111
                                 + 1.09474 w_k1_AXP_s111
                                 + 0.949029 w_k1_DIS_s111
                                 + 0.984749 w_k1_FDX_s111
                                 + 0.893304 w_k1_IBM_s111
                                 + 1.17555 w_k1_INTC_s111
                                 + 0.890729 w_k1_KO_s111
                                 + 1.26258 w_k1_MAT_s111
                                 + 1.00103 w_k1_MSFT_s111
                                 + 1.04959 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.02553 w_k1_ADM_s111
                                 + 0.860328 w_k1_AXP_s111
                                 + 1.03506 w_k1_DIS_s111
                                 + 1.07455 w_k1_FDX_s111
                                 + 0.854453 w_k1_IBM_s111
                                 + 0.95805 w_k1_INTC_s111
                                 + 0.932989 w_k1_KO_s111
                                 + 1.05125 w_k1_MAT_s111
                                 + 1.04206 w_k1_MSFT_s111
                                 + 1.03473 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.00716 w_k1_ADM_s111
                                 + 0.667638 w_k1_AXP_s111
                                 + 0.731806 w_k1_DIS_s111
                                 + 0.840239 w_k1_FDX_s111
                                 + 0.696381 w_k1_IBM_s111
                                 + 0.780605 w_k1_INTC_s111
                                 + 0.720752 w_k1_KO_s111
                                 + 0.643789 w_k1_MAT_s111
                                 + 0.734743 w_k1_MSFT_s111
                                 + 0.863882 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.19462 w_k1_ADM_s111
                                 + 1.46537 w_k1_AXP_s111
                                 + 1.21236 w_k1_DIS_s111
                                 + 1.23369 w_k1_FDX_s111
                                 + 1.50225 w_k1_IBM_s111
                                 + 1.09401 w_k1_INTC_s111
                                 + 1.19574 w_k1_KO_s111 + 1.57658 w_k1_MAT_s111
                                 + 1.1202 w_k1_MSFT_s111 + 1.29623 w_k1_PG_s111
                                 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.735814 w_k1_ADM_s112
                                 + 1.54171 w_k1_AXP_s112
                                 + 1.53856 w_k1_DIS_s112 + 1.4629 w_k1_FDX_s112
                                 + 1.02529 w_k1_IBM_s112
                                 + 1.25225 w_k1_INTC_s112
                                 + 1.05711 w_k1_KO_s112 + 0.88467 w_k1_MAT_s112
                                 + 1.46579 w_k1_MSFT_s112 + 1.0748 w_k1_PG_s112
                                 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.647406 w_k1_ADM_s112
                                 + 0.594754 w_k1_AXP_s112
                                 + 0.644098 w_k1_DIS_s112
                                 + 0.68237 w_k1_FDX_s112
                                 + 0.805753 w_k1_IBM_s112
                                 + 0.739915 w_k1_INTC_s112
                                 + 0.658674 w_k1_KO_s112
                                 + 0.683695 w_k1_MAT_s112
                                 + 0.762722 w_k1_MSFT_s112
                                 + 0.91004 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.719427 w_k1_ADM_s112
                                 + 1.06124 w_k1_AXP_s112
                                 + 0.858442 w_k1_DIS_s112
                                 + 0.985814 w_k1_FDX_s112
                                 + 0.776279 w_k1_IBM_s112
                                 + 0.975208 w_k1_INTC_s112
                                 + 0.803649 w_k1_KO_s112
                                 + 0.623114 w_k1_MAT_s112
                                 + 0.92259 w_k1_MSFT_s112 + 1.0376 w_k1_PG_s112
                                 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.13293 w_k1_ADM_s112
                                 + 1.35678 w_k1_AXP_s112
                                 + 1.20927 w_k1_DIS_s112
                                 + 1.21052 w_k1_FDX_s112
                                 + 1.06892 w_k1_IBM_s112
                                 + 0.968537 w_k1_INTC_s112
                                 + 0.974675 w_k1_KO_s112
                                 + 1.03168 w_k1_MAT_s112
                                 + 1.13379 w_k1_MSFT_s112 + 1.1318 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.775459 w_k1_ADM_s112
                                 + 1.05804 w_k1_AXP_s112
                                 + 1.27921 w_k1_DIS_s112
                                 + 1.03309 w_k1_FDX_s112
                                 + 1.06868 w_k1_IBM_s112
                                 + 1.13737 w_k1_INTC_s112
                                 + 0.914807 w_k1_KO_s112
                                 + 0.788657 w_k1_MAT_s112
                                 + 1.23971 w_k1_MSFT_s112
                                 + 1.04891 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.919319 w_k1_ADM_s112
                                 + 1.67011 w_k1_AXP_s112
                                 + 1.54104 w_k1_DIS_s112
                                 + 1.07076 w_k1_FDX_s112
                                 + 1.63183 w_k1_IBM_s112
                                 + 1.0992 w_k1_INTC_s112 + 1.07933 w_k1_KO_s112
                                 + 1.25418 w_k1_MAT_s112
                                 + 1.13019 w_k1_MSFT_s112
                                 + 1.04865 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.80751 w_k1_ADM_s112
                                 + 1.10544 w_k1_AXP_s112
                                 + 1.00062 w_k1_DIS_s112
                                 + 1.06734 w_k1_FDX_s112
                                 + 0.887761 w_k1_IBM_s112
                                 + 1.0122 w_k1_INTC_s112 + 1.03314 w_k1_KO_s112
                                 + 1.17092 w_k1_MAT_s112
                                 + 0.990073 w_k1_MSFT_s112
                                 + 1.17746 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.600039 w_k1_ADM_s112
                                 + 0.79431 w_k1_AXP_s112
                                 + 0.816797 w_k1_DIS_s112
                                 + 0.799402 w_k1_FDX_s112
                                 + 0.882016 w_k1_IBM_s112
                                 + 0.816822 w_k1_INTC_s112
                                 + 0.8326 w_k1_KO_s112 + 0.983054 w_k1_MAT_s112
                                 + 0.984299 w_k1_MSFT_s112
                                 + 0.90149 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.93543 w_k1_ADM_s113
                                 + 1.21585 w_k1_AXP_s113
                                 + 1.38008 w_k1_DIS_s113
                                 + 1.24009 w_k1_FDX_s113
                                 + 1.17559 w_k1_IBM_s113
                                 + 1.30553 w_k1_INTC_s113
                                 + 1.36146 w_k1_KO_s113 + 1.12172 w_k1_MAT_s113
                                 + 1.1321 w_k1_MSFT_s113 + 1.23402 w_k1_PG_s113
                                 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.09487 w_k1_ADM_s113
                                 + 1.34966 w_k1_AXP_s113
                                 + 1.13813 w_k1_DIS_s113
                                 + 1.22785 w_k1_FDX_s113
                                 + 1.08099 w_k1_IBM_s113
                                 + 1.3908 w_k1_INTC_s113 + 1.08569 w_k1_KO_s113
                                 + 1.05758 w_k1_MAT_s113
                                 + 1.12013 w_k1_MSFT_s113
                                 + 1.05842 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.0528 w_k1_ADM_s113
                                 + 0.822602 w_k1_AXP_s113
                                 + 1.13532 w_k1_DIS_s113
                                 + 1.00064 w_k1_FDX_s113
                                 + 0.899539 w_k1_IBM_s113
                                 + 1.37897 w_k1_INTC_s113
                                 + 1.11136 w_k1_KO_s113
                                 + 0.659094 w_k1_MAT_s113
                                 + 1.09554 w_k1_MSFT_s113
                                 + 1.11409 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.07314 w_k1_ADM_s113
                                 + 0.981062 w_k1_AXP_s113
                                 + 1.04132 w_k1_DIS_s113
                                 + 1.14901 w_k1_FDX_s113
                                 + 1.03734 w_k1_IBM_s113
                                 + 1.49134 w_k1_INTC_s113
                                 + 1.10124 w_k1_KO_s113 + 0.60011 w_k1_MAT_s113
                                 + 1.12572 w_k1_MSFT_s113
                                 + 1.01137 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.26787 w_k1_ADM_s113
                                 + 1.34743 w_k1_AXP_s113
                                 + 1.33925 w_k1_DIS_s113
                                 + 1.11909 w_k1_FDX_s113
                                 + 1.79544 w_k1_IBM_s113
                                 + 1.52902 w_k1_INTC_s113 + 1.6553 w_k1_KO_s113
                                 + 1.53844 w_k1_MAT_s113
                                 + 1.21267 w_k1_MSFT_s113
                                 + 1.28436 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.948853 w_k1_ADM_s113
                                 + 0.996097 w_k1_AXP_s113
                                 + 0.675869 w_k1_DIS_s113
                                 + 0.966835 w_k1_FDX_s113
                                 + 0.767294 w_k1_IBM_s113
                                 + 1.03971 w_k1_INTC_s113
                                 + 0.737401 w_k1_KO_s113
                                 + 0.669494 w_k1_MAT_s113
                                 + 0.84184 w_k1_MSFT_s113
                                 + 0.924061 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.881856 w_k1_ADM_s113
                                 + 0.714487 w_k1_AXP_s113
                                 + 0.754763 w_k1_DIS_s113
                                 + 0.829695 w_k1_FDX_s113
                                 + 0.970316 w_k1_IBM_s113
                                 + 0.992014 w_k1_INTC_s113
                                 + 0.949916 w_k1_KO_s113
                                 + 0.842381 w_k1_MAT_s113
                                 + 1.00882 w_k1_MSFT_s113
                                 + 0.890749 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.902017 w_k1_ADM_s113
                                 + 0.924413 w_k1_AXP_s113
                                 + 1.11163 w_k1_DIS_s113
                                 + 1.01294 w_k1_FDX_s113
                                 + 1.31703 w_k1_IBM_s113
                                 + 1.03935 w_k1_INTC_s113
                                 + 1.10707 w_k1_KO_s113
                                 + 0.913647 w_k1_MAT_s113
                                 + 0.917022 w_k1_MSFT_s113
                                 + 0.975352 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.15859 w_k1_ADM_s114
                                 + 1.06152 w_k1_AXP_s114
                                 + 1.13122 w_k1_DIS_s114
                                 + 0.939823 w_k1_FDX_s114
                                 + 1.35534 w_k1_IBM_s114
                                 + 0.845335 w_k1_INTC_s114
                                 + 1.12045 w_k1_KO_s114 + 1.32608 w_k1_MAT_s114
                                 + 1.0852 w_k1_MSFT_s114 + 1.14672 w_k1_PG_s114
                                 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.752445 w_k1_ADM_s114
                                 + 1.21214 w_k1_AXP_s114
                                 + 1.04624 w_k1_DIS_s114
                                 + 1.04793 w_k1_FDX_s114
                                 + 1.16908 w_k1_IBM_s114
                                 + 0.987764 w_k1_INTC_s114
                                 + 0.898579 w_k1_KO_s114
                                 + 1.25593 w_k1_MAT_s114
                                 + 1.04405 w_k1_MSFT_s114
                                 + 0.915612 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.10684 w_k1_ADM_s114
                                 + 1.03331 w_k1_AXP_s114
                                 + 1.05514 w_k1_DIS_s114
                                 + 1.09302 w_k1_FDX_s114
                                 + 0.889944 w_k1_IBM_s114
                                 + 1.11079 w_k1_INTC_s114
                                 + 1.00705 w_k1_KO_s114 + 1.0464 w_k1_MAT_s114
                                 + 1.12652 w_k1_MSFT_s114
                                 + 1.04469 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.02677 w_k1_ADM_s114
                                 + 0.941795 w_k1_AXP_s114
                                 + 0.744723 w_k1_DIS_s114
                                 + 0.781277 w_k1_FDX_s114
                                 + 1.10851 w_k1_IBM_s114 + 1.059 w_k1_INTC_s114
                                 + 1.05091 w_k1_KO_s114
                                 + 0.994862 w_k1_MAT_s114
                                 + 0.896114 w_k1_MSFT_s114
                                 + 0.951481 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.5843 w_k1_ADM_s114
                                 + 1.44651 w_k1_AXP_s114
                                 + 1.72256 w_k1_DIS_s114
                                 + 1.37139 w_k1_FDX_s114
                                 + 1.05486 w_k1_IBM_s114
                                 + 1.11608 w_k1_INTC_s114
                                 + 1.44169 w_k1_KO_s114 + 1.29617 w_k1_MAT_s114
                                 + 1.49312 w_k1_MSFT_s114
                                 + 1.31551 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.987661 w_k1_ADM_s114
                                 + 0.822795 w_k1_AXP_s114
                                 + 0.774235 w_k1_DIS_s114
                                 + 0.798705 w_k1_FDX_s114
                                 + 0.68243 w_k1_IBM_s114
                                 + 0.811072 w_k1_INTC_s114
                                 + 0.751555 w_k1_KO_s114
                                 + 0.734408 w_k1_MAT_s114
                                 + 0.87796 w_k1_MSFT_s114
                                 + 0.847367 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.27735 w_k1_ADM_s114
                                 + 1.31194 w_k1_AXP_s114
                                 + 1.35594 w_k1_DIS_s114
                                 + 1.19629 w_k1_FDX_s114
                                 + 0.952411 w_k1_IBM_s114
                                 + 1.01851 w_k1_INTC_s114
                                 + 1.16708 w_k1_KO_s114 + 1.27291 w_k1_MAT_s114
                                 + 1.17851 w_k1_MSFT_s114
                                 + 1.32881 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.28833 w_k1_ADM_s114
                                 + 1.45883 w_k1_AXP_s114
                                 + 1.06894 w_k1_DIS_s114 + 1.1003 w_k1_FDX_s114
                                 + 0.740601 w_k1_IBM_s114
                                 + 1.05007 w_k1_INTC_s114
                                 + 0.910969 w_k1_KO_s114
                                 + 1.10614 w_k1_MAT_s114
                                 + 1.14136 w_k1_MSFT_s114
                                 + 0.94417 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.772474 w_k1_ADM_s115
                                 + 0.967808 w_k1_AXP_s115
                                 + 1.11016 w_k1_DIS_s115
                                 + 1.01393 w_k1_FDX_s115
                                 + 0.90839 w_k1_IBM_s115
                                 + 1.12654 w_k1_INTC_s115
                                 + 0.847892 w_k1_KO_s115
                                 + 0.729685 w_k1_MAT_s115
                                 + 1.16619 w_k1_MSFT_s115
                                 + 0.943788 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.611249 w_k1_ADM_s115
                                 + 0.718268 w_k1_AXP_s115
                                 + 0.522893 w_k1_DIS_s115
                                 + 0.644354 w_k1_FDX_s115
                                 + 0.87787 w_k1_IBM_s115
                                 + 0.765261 w_k1_INTC_s115
                                 + 0.677658 w_k1_KO_s115
                                 + 0.835012 w_k1_MAT_s115
                                 + 0.811734 w_k1_MSFT_s115
                                 + 0.78079 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.92637 w_k1_ADM_s115
                                 + 1.02497 w_k1_AXP_s115
                                 + 1.23504 w_k1_DIS_s115
                                 + 1.10483 w_k1_FDX_s115
                                 + 1.14528 w_k1_IBM_s115
                                 + 1.06416 w_k1_INTC_s115
                                 + 0.994468 w_k1_KO_s115
                                 + 1.28785 w_k1_MAT_s115
                                 + 1.13614 w_k1_MSFT_s115
                                 + 0.951323 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.10738 w_k1_ADM_s115
                                 + 1.47408 w_k1_AXP_s115
                                 + 1.44199 w_k1_DIS_s115
                                 + 1.23873 w_k1_FDX_s115
                                 + 1.61786 w_k1_IBM_s115
                                 + 1.23024 w_k1_INTC_s115
                                 + 1.52578 w_k1_KO_s115 + 2.2515 w_k1_MAT_s115
                                 + 1.27925 w_k1_MSFT_s115
                                 + 1.26622 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.08462 w_k1_ADM_s115
                                 + 1.28185 w_k1_AXP_s115
                                 + 1.38792 w_k1_DIS_s115
                                 + 1.20836 w_k1_FDX_s115
                                 + 0.99426 w_k1_IBM_s115
                                 + 1.4433 w_k1_INTC_s115 + 1.05921 w_k1_KO_s115
                                 + 1.08943 w_k1_MAT_s115
                                 + 1.37409 w_k1_MSFT_s115 + 1.021 w_k1_PG_s115
                                 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.655712 w_k1_ADM_s115
                                 + 0.796845 w_k1_AXP_s115
                                 + 0.764771 w_k1_DIS_s115
                                 + 0.930411 w_k1_FDX_s115
                                 + 0.655063 w_k1_IBM_s115
                                 + 1.0728 w_k1_INTC_s115
                                 + 0.674852 w_k1_KO_s115
                                 + 0.50994 w_k1_MAT_s115
                                 + 0.890602 w_k1_MSFT_s115
                                 + 0.777601 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.685933 w_k1_ADM_s115
                                 + 0.975479 w_k1_AXP_s115
                                 + 0.753072 w_k1_DIS_s115
                                 + 0.881068 w_k1_FDX_s115
                                 + 0.974897 w_k1_IBM_s115
                                 + 0.780823 w_k1_INTC_s115
                                 + 0.830433 w_k1_KO_s115
                                 + 1.21002 w_k1_MAT_s115
                                 + 0.847552 w_k1_MSFT_s115
                                 + 0.951854 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.01619 w_k1_ADM_s115
                                 + 1.73257 w_k1_AXP_s115
                                 + 1.89478 w_k1_DIS_s115
                                 + 1.44869 w_k1_FDX_s115
                                 + 1.27999 w_k1_IBM_s115
                                 + 1.39262 w_k1_INTC_s115
                                 + 1.28743 w_k1_KO_s115 + 1.77493 w_k1_MAT_s115
                                 + 1.48582 w_k1_MSFT_s115
                                 + 1.18926 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.91909 w_k1_ADM_s116
                                 + 0.654904 w_k1_AXP_s116
                                 + 0.647519 w_k1_DIS_s116
                                 + 0.865266 w_k1_FDX_s116
                                 + 0.608828 w_k1_IBM_s116
                                 + 0.929381 w_k1_INTC_s116
                                 + 0.69898 w_k1_KO_s116
                                 + 0.435593 w_k1_MAT_s116
                                 + 0.894227 w_k1_MSFT_s116
                                 + 0.854443 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.0814 w_k1_ADM_s116
                                 + 1.06824 w_k1_AXP_s116
                                 + 1.12432 w_k1_DIS_s116 + 1.1481 w_k1_FDX_s116
                                 + 1.12675 w_k1_IBM_s116
                                 + 1.16641 w_k1_INTC_s116
                                 + 1.07922 w_k1_KO_s116 + 1.24696 w_k1_MAT_s116
                                 + 1.16776 w_k1_MSFT_s116
                                 + 1.19021 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.1844 w_k1_ADM_s116
                                 + 1.38648 w_k1_AXP_s116
                                 + 1.32763 w_k1_DIS_s116 + 1.2238 w_k1_FDX_s116
                                 + 1.01597 w_k1_IBM_s116
                                 + 1.40817 w_k1_INTC_s116
                                 + 1.32749 w_k1_KO_s116 + 1.43785 w_k1_MAT_s116
                                 + 1.3372 w_k1_MSFT_s116 + 1.27446 w_k1_PG_s116
                                 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.09414 w_k1_ADM_s116
                                 + 1.11557 w_k1_AXP_s116
                                 + 0.709627 w_k1_DIS_s116
                                 + 0.896209 w_k1_FDX_s116
                                 + 0.905342 w_k1_IBM_s116
                                 + 0.964545 w_k1_INTC_s116
                                 + 0.947943 w_k1_KO_s116 + 1.2311 w_k1_MAT_s116
                                 + 0.904347 w_k1_MSFT_s116
                                 + 1.02964 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.07934 w_k1_ADM_s116
                                 + 1.62238 w_k1_AXP_s116 + 1.3716 w_k1_DIS_s116
                                 + 1.14942 w_k1_FDX_s116
                                 + 2.15948 w_k1_IBM_s116
                                 + 1.2483 w_k1_INTC_s116 + 1.73336 w_k1_KO_s116
                                 + 2.62913 w_k1_MAT_s116
                                 + 1.16267 w_k1_MSFT_s116
                                 + 1.31089 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.888424 w_k1_ADM_s116
                                 + 0.894133 w_k1_AXP_s116
                                 + 0.942999 w_k1_DIS_s116
                                 + 0.987863 w_k1_FDX_s116
                                 + 0.783435 w_k1_IBM_s116
                                 + 1.06511 w_k1_INTC_s116
                                 + 1.00721 w_k1_KO_s116
                                 + 0.907099 w_k1_MAT_s116
                                 + 1.16595 w_k1_MSFT_s116
                                 + 0.985789 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.03285 w_k1_ADM_s116
                                 + 1.06645 w_k1_AXP_s116
                                 + 1.17389 w_k1_DIS_s116
                                 + 0.935391 w_k1_FDX_s116
                                 + 1.48911 w_k1_IBM_s116
                                 + 1.1525 w_k1_INTC_s116 + 1.3801 w_k1_KO_s116
                                 + 1.23175 w_k1_MAT_s116
                                 + 1.04968 w_k1_MSFT_s116
                                 + 1.03427 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.692964 w_k1_ADM_s116
                                 + 0.779186 w_k1_AXP_s116
                                 + 0.589808 w_k1_DIS_s116
                                 + 0.707602 w_k1_FDX_s116
                                 + 0.889145 w_k1_IBM_s116
                                 + 0.809785 w_k1_INTC_s116
                                 + 0.841258 w_k1_KO_s116
                                 + 0.755464 w_k1_MAT_s116
                                 + 0.725626 w_k1_MSFT_s116
                                 + 0.854428 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.791718 w_k1_ADM_s117
                                 + 0.652279 w_k1_AXP_s117
                                 + 0.664225 w_k1_DIS_s117
                                 + 0.729346 w_k1_FDX_s117
                                 + 0.828964 w_k1_IBM_s117
                                 + 1.01417 w_k1_INTC_s117
                                 + 0.913197 w_k1_KO_s117
                                 + 0.522077 w_k1_MAT_s117
                                 + 0.926344 w_k1_MSFT_s117
                                 + 0.811365 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.760946 w_k1_ADM_s117
                                 + 0.838805 w_k1_AXP_s117
                                 + 1.22939 w_k1_DIS_s117
                                 + 1.01263 w_k1_FDX_s117 + 1.0142 w_k1_IBM_s117
                                 + 1.12382 w_k1_INTC_s117
                                 + 1.25329 w_k1_KO_s117
                                 + 0.879677 w_k1_MAT_s117
                                 + 1.06881 w_k1_MSFT_s117
                                 + 0.956144 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.16639 w_k1_ADM_s117
                                 + 1.28351 w_k1_AXP_s117
                                 + 1.34192 w_k1_DIS_s117
                                 + 1.30162 w_k1_FDX_s117
                                 + 0.870595 w_k1_IBM_s117
                                 + 1.2574 w_k1_INTC_s117 + 1.11322 w_k1_KO_s117
                                 + 0.650566 w_k1_MAT_s117
                                 + 1.22841 w_k1_MSFT_s117
                                 + 1.06472 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.922217 w_k1_ADM_s117
                                 + 0.851114 w_k1_AXP_s117
                                 + 0.862814 w_k1_DIS_s117
                                 + 0.956386 w_k1_FDX_s117
                                 + 0.60575 w_k1_IBM_s117
                                 + 0.994919 w_k1_INTC_s117
                                 + 0.793758 w_k1_KO_s117
                                 + 0.391639 w_k1_MAT_s117
                                 + 0.949181 w_k1_MSFT_s117
                                 + 0.892091 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.939925 w_k1_ADM_s117
                                 + 0.978063 w_k1_AXP_s117
                                 + 0.96502 w_k1_DIS_s117
                                 + 0.974753 w_k1_FDX_s117
                                 + 1.04355 w_k1_IBM_s117
                                 + 0.986817 w_k1_INTC_s117
                                 + 1.16251 w_k1_KO_s117 + 0.9745 w_k1_MAT_s117
                                 + 0.952731 w_k1_MSFT_s117
                                 + 0.919555 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.962072 w_k1_ADM_s117
                                 + 0.899751 w_k1_AXP_s117
                                 + 0.756901 w_k1_DIS_s117
                                 + 0.753138 w_k1_FDX_s117
                                 + 1.17594 w_k1_IBM_s117
                                 + 0.841948 w_k1_INTC_s117
                                 + 1.05066 w_k1_KO_s117 + 1.2796 w_k1_MAT_s117
                                 + 0.817421 w_k1_MSFT_s117
                                 + 1.0143 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.967912 w_k1_ADM_s117
                                 + 1.1669 w_k1_AXP_s117 + 1.43063 w_k1_DIS_s117
                                 + 1.16505 w_k1_FDX_s117 + 1.5525 w_k1_IBM_s117
                                 + 1.3017 w_k1_INTC_s117 + 1.42776 w_k1_KO_s117
                                 + 1.22964 w_k1_MAT_s117
                                 + 1.10661 w_k1_MSFT_s117
                                 + 1.03987 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.719114 w_k1_ADM_s117
                                 + 0.876158 w_k1_AXP_s117
                                 + 0.874536 w_k1_DIS_s117
                                 + 0.837571 w_k1_FDX_s117
                                 + 1.36699 w_k1_IBM_s117
                                 + 1.04981 w_k1_INTC_s117
                                 + 1.14309 w_k1_KO_s117 + 1.15533 w_k1_MAT_s117
                                 + 0.876841 w_k1_MSFT_s117
                                 + 0.865901 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.040796112026506 w_k1_ADM_s000
                                 + 0.0101691985891508 w_k1_ADM_s001
                                 + 0.0104500136475334 w_k1_ADM_s002
                                 + 0.0293383519986727 w_k1_ADM_s003
                                 + 0.0131239992396458 w_k1_ADM_s004
                                 + 0.0278229368519197 w_k1_ADM_s005
                                 + 0.043796022017944 w_k1_ADM_s006
                                 + 0.0347214055072584 w_k1_ADM_s007
                                 + 0.0542089412964079 w_k1_ADM_s010
                                 + 0.0507385033104818 w_k1_ADM_s011
                                 + 0.0259579946967535 w_k1_ADM_s012
                                 + 0.0217423725233689 w_k1_ADM_s013
                                 + 0.0502189573136864 w_k1_ADM_s014
                                 + 0.0131433259429274 w_k1_ADM_s015
                                 + 0.0319233622810502 w_k1_ADM_s016
                                 + 0.032710672935815 w_k1_ADM_s017
                                 + 0.096598348545285 w_k1_ADM_s100
                                 + 0.0411755225896118 w_k1_ADM_s101
                                 + 0.0665763177167606 w_k1_ADM_s102
                                 + 0.0230605933567453 w_k1_ADM_s103
                                 + 0.0533958378702383 w_k1_ADM_s104
                                 + 0.0444226353627693 w_k1_ADM_s105
                                 + 0.0392313374817624 w_k1_ADM_s106
                                 + 0.0396126266442558 w_k1_ADM_s107
                                 + 0.0283158922954452 w_k1_ADM_s110
                                 + 0.0140449956873837 w_k1_ADM_s111
                                 + 0.00507654323987986 w_k1_ADM_s112
                                 + 0.0096454835307218 w_k1_ADM_s113
                                 + 0.00665085875797897 w_k1_ADM_s114
                                 + 0.00539842564717363 w_k1_ADM_s115
                                 + 0.0183336871881788 w_k1_ADM_s116
                                 + 0.00735054975613461 w_k1_ADM_s117
                                 + 0.0405536039128203 w_k1_AXP_s000
                                 + 0.00878330052883983 w_k1_AXP_s001
                                 + 0.00993057187806157 w_k1_AXP_s002
                                 + 0.0332583941152421 w_k1_AXP_s003
                                 + 0.0144741633829551 w_k1_AXP_s004
                                 + 0.0285777661077387 w_k1_AXP_s005
                                 + 0.0440330696149555 w_k1_AXP_s006
                                 + 0.0380783449320013 w_k1_AXP_s007
                                 + 0.0505189626236286 w_k1_AXP_s010
                                 + 0.0496589583500803 w_k1_AXP_s011
                                 + 0.0269507456632967 w_k1_AXP_s012
                                 + 0.0203291210769539 w_k1_AXP_s013
                                 + 0.0476144028581564 w_k1_AXP_s014
                                 + 0.01244110953963 w_k1_AXP_s015
                                 + 0.0369290600894499 w_k1_AXP_s016
                                 + 0.0364324768095087 w_k1_AXP_s017
                                 + 0.0978517740533426 w_k1_AXP_s100
                                 + 0.0362720096956898 w_k1_AXP_s101
                                 + 0.0724190561725071 w_k1_AXP_s102
                                 + 0.0224555048543259 w_k1_AXP_s103
                                 + 0.0514242134247394 w_k1_AXP_s104
                                 + 0.0502481773174782 w_k1_AXP_s105
                                 + 0.0491613970412396 w_k1_AXP_s106
                                 + 0.0343707850481646 w_k1_AXP_s107
                                 + 0.0290530328551711 w_k1_AXP_s110
                                 + 0.0133411087872077 w_k1_AXP_s111
                                 + 0.00757537561783605 w_k1_AXP_s112
                                 + 0.0100267973173412 w_k1_AXP_s113
                                 + 0.00656793332750375 w_k1_AXP_s114
                                 + 0.00692895463305263 w_k1_AXP_s115
                                 + 0.0192306151329609 w_k1_AXP_s116
                                 + 0.00777595661729833 w_k1_AXP_s117
                                 + 0.0365952470523708 w_k1_DIS_s000
                                 + 0.00891969399513469 w_k1_DIS_s001
                                 + 0.0102034012627571 w_k1_DIS_s002
                                 + 0.0349532796462082 w_k1_DIS_s003
                                 + 0.0143862770212036 w_k1_DIS_s004
                                 + 0.0305767857226587 w_k1_DIS_s005
                                 + 0.0442940268552308 w_k1_DIS_s006
                                 + 0.0391265681823406 w_k1_DIS_s007
                                 + 0.0561480513294348 w_k1_DIS_s010
                                 + 0.0505988574970901 w_k1_DIS_s011
                                 + 0.0295025879724482 w_k1_DIS_s012
                                 + 0.020840744189998 w_k1_DIS_s013
                                 + 0.0487220664257539 w_k1_DIS_s014
                                 + 0.0134038549070747 w_k1_DIS_s015
                                 + 0.0383309007178927 w_k1_DIS_s016
                                 + 0.036631740879553 w_k1_DIS_s017
                                 + 0.110324924752658 w_k1_DIS_s100
                                 + 0.0404244621269202 w_k1_DIS_s101
                                 + 0.080491324710352 w_k1_DIS_s102
                                 + 0.0225395420179264 w_k1_DIS_s103
                                 + 0.0563104611291996 w_k1_DIS_s104
                                 + 0.0493739335305808 w_k1_DIS_s105
                                 + 0.039338834639488 w_k1_DIS_s106
                                 + 0.0378651598783915 w_k1_DIS_s107
                                 + 0.0293291002087971 w_k1_DIS_s110
                                 + 0.0145838294651699 w_k1_DIS_s111
                                 + 0.0071851859130049 w_k1_DIS_s112
                                 + 0.00950245606571535 w_k1_DIS_s113
                                 + 0.00627889256521741 w_k1_DIS_s114
                                 + 0.00710922049863677 w_k1_DIS_s115
                                 + 0.0178244156380192 w_k1_DIS_s116
                                 + 0.00839896004259043 w_k1_DIS_s117
                                 + 0.034966251366982 w_k1_FDX_s000
                                 + 0.00880031717115128 w_k1_FDX_s001
                                 + 0.0100590383457779 w_k1_FDX_s002
                                 + 0.0316263848951036 w_k1_FDX_s003
                                 + 0.01369547657532 w_k1_FDX_s004
                                 + 0.0313912673987865 w_k1_FDX_s005
                                 + 0.0437728554073727 w_k1_FDX_s006
                                 + 0.0371447969597337 w_k1_FDX_s007
                                 + 0.0551699370364646 w_k1_FDX_s010
                                 + 0.050238632839415 w_k1_FDX_s011
                                 + 0.0278652864925333 w_k1_FDX_s012
                                 + 0.0214712032703981 w_k1_FDX_s013
                                 + 0.0463273471836245 w_k1_FDX_s014
                                 + 0.0133664903409988 w_k1_FDX_s015
                                 + 0.0359526791089959 w_k1_FDX_s016
                                 + 0.0349845232562269 w_k1_FDX_s017
                                 + 0.0954688234616664 w_k1_FDX_s100
                                 + 0.038185683247154 w_k1_FDX_s101
                                 + 0.0743659279123126 w_k1_FDX_s102
                                 + 0.0211215786439913 w_k1_FDX_s103
                                 + 0.0557707668370313 w_k1_FDX_s104
                                 + 0.0495630659216731 w_k1_FDX_s105
                                 + 0.0420585027249973 w_k1_FDX_s106
                                 + 0.0388423010844629 w_k1_FDX_s107
                                 + 0.0281968108946041 w_k1_FDX_s110
                                 + 0.0142644064033653 w_k1_FDX_s111
                                 + 0.00676058956008372 w_k1_FDX_s112
                                 + 0.0100956849322885 w_k1_FDX_s113
                                 + 0.00589617318146057 w_k1_FDX_s114
                                 + 0.00676359726430191 w_k1_FDX_s115
                                 + 0.0179601339339708 w_k1_FDX_s116
                                 + 0.00801393192393589 w_k1_FDX_s117
                                 + 0.0358813482208282 w_k1_IBM_s000
                                 + 0.00966421027845388 w_k1_IBM_s001
                                 + 0.0111478766058391 w_k1_IBM_s002
                                 + 0.0313336894599722 w_k1_IBM_s003
                                 + 0.0140826020024063 w_k1_IBM_s004
                                 + 0.0311617364258003 w_k1_IBM_s005
                                 + 0.043322622440779 w_k1_IBM_s006
                                 + 0.0382452267212532 w_k1_IBM_s007
                                 + 0.0543370711416751 w_k1_IBM_s010
                                 + 0.0499627648054919 w_k1_IBM_s011
                                 + 0.0278981037842037 w_k1_IBM_s012
                                 + 0.0209807123391209 w_k1_IBM_s013
                                 + 0.0470061855007326 w_k1_IBM_s014
                                 + 0.0144411085884162 w_k1_IBM_s015
                                 + 0.0371345557788505 w_k1_IBM_s016
                                 + 0.0376831221704967 w_k1_IBM_s017
                                 + 0.09854718637402 w_k1_IBM_s100
                                 + 0.0410189421512598 w_k1_IBM_s101
                                 + 0.0754789490758942 w_k1_IBM_s102
                                 + 0.0225080705222424 w_k1_IBM_s103
                                 + 0.0553445188262581 w_k1_IBM_s104
                                 + 0.0485798933055672 w_k1_IBM_s105
                                 + 0.041894291147081 w_k1_IBM_s106
                                 + 0.0382268289095886 w_k1_IBM_s107
                                 + 0.0298998798678371 w_k1_IBM_s110
                                 + 0.0144306795395934 w_k1_IBM_s111
                                 + 0.00666565774958664 w_k1_IBM_s112
                                 + 0.0101406184268347 w_k1_IBM_s113
                                 + 0.00598018734851444 w_k1_IBM_s114
                                 + 0.00644193857733924 w_k1_IBM_s115
                                 + 0.019019263935809 w_k1_IBM_s116
                                 + 0.00790505279140341 w_k1_IBM_s117
                                 + 0.0354733033982783 w_k1_INTC_s000
                                 + 0.00932089706741589 w_k1_INTC_s001
                                 + 0.00926260458054113 w_k1_INTC_s002
                                 + 0.0280422479472324 w_k1_INTC_s003
                                 + 0.0152636221245603 w_k1_INTC_s004
                                 + 0.0314173144091142 w_k1_INTC_s005
                                 + 0.0398526197510408 w_k1_INTC_s006
                                 + 0.0334170221548797 w_k1_INTC_s007
                                 + 0.0551022923415529 w_k1_INTC_s010
                                 + 0.0505886310780939 w_k1_INTC_s011
                                 + 0.0300697612251365 w_k1_INTC_s012
                                 + 0.0229038122235087 w_k1_INTC_s013
                                 + 0.0473697310621507 w_k1_INTC_s014
                                 + 0.0158096361567968 w_k1_INTC_s015
                                 + 0.0400159079732801 w_k1_INTC_s016
                                 + 0.0391591138379197 w_k1_INTC_s017
                                 + 0.0973442212037383 w_k1_INTC_s100
                                 + 0.0399805032680475 w_k1_INTC_s101
                                 + 0.076069551784385 w_k1_INTC_s102
                                 + 0.0233640743478977 w_k1_INTC_s103
                                 + 0.0481198656148552 w_k1_INTC_s104
                                 + 0.0450536774357382 w_k1_INTC_s105
                                 + 0.0488094034884759 w_k1_INTC_s106
                                 + 0.0373735952564536 w_k1_INTC_s107
                                 + 0.0272246919951196 w_k1_INTC_s110
                                 + 0.0136460135556589 w_k1_INTC_s111
                                 + 0.00656900101821828 w_k1_INTC_s112
                                 + 0.0117629204676312 w_k1_INTC_s113
                                 + 0.00576837582028937 w_k1_INTC_s114
                                 + 0.0071918409239867 w_k1_INTC_s115
                                 + 0.0199122132497172 w_k1_INTC_s116
                                 + 0.00850897002478892 w_k1_INTC_s117
                                 + 0.0366886475778461 w_k1_KO_s000
                                 + 0.00818339757362138 w_k1_KO_s001
                                 + 0.0102354547789211 w_k1_KO_s002
                                 + 0.029370434806653 w_k1_KO_s003
                                 + 0.01483684178919 w_k1_KO_s004
                                 + 0.0296999068994091 w_k1_KO_s005
                                 + 0.0423148417061938 w_k1_KO_s006
                                 + 0.0349026586562605 w_k1_KO_s007
                                 + 0.0543592219424218 w_k1_KO_s010
                                 + 0.0439546416584017 w_k1_KO_s011
                                 + 0.0323223606266402 w_k1_KO_s012
                                 + 0.0202159637088268 w_k1_KO_s013
                                 + 0.0490526818952928 w_k1_KO_s014
                                 + 0.0148590013012856 w_k1_KO_s015
                                 + 0.035366066760898 w_k1_KO_s016
                                 + 0.0367946131376292 w_k1_KO_s017
                                 + 0.0978665156403327 w_k1_KO_s100
                                 + 0.0404074909524946 w_k1_KO_s101
                                 + 0.0833547268648567 w_k1_KO_s102
                                 + 0.022597571633064 w_k1_KO_s103
                                 + 0.0590603874759226 w_k1_KO_s104
                                 + 0.0527648498911426 w_k1_KO_s105
                                 + 0.0409600272890099 w_k1_KO_s106
                                 + 0.0431185866799647 w_k1_KO_s107
                                 + 0.0294318828068354 w_k1_KO_s110
                                 + 0.0134447271488168 w_k1_KO_s111
                                 + 0.00599541821025207 w_k1_KO_s112
                                 + 0.0102008462838619 w_k1_KO_s113
                                 + 0.00604712085442262 w_k1_KO_s114
                                 + 0.00605433237941528 w_k1_KO_s115
                                 + 0.0198672526350894 w_k1_KO_s116
                                 + 0.00866697588223673 w_k1_KO_s117
                                 + 0.0339458590819421 w_k1_MAT_s000
                                 + 0.00842520145789114 w_k1_MAT_s001
                                 + 0.00962478262947942 w_k1_MAT_s002
                                 + 0.0294267501968388 w_k1_MAT_s003
                                 + 0.0139213578884615 w_k1_MAT_s004
                                 + 0.0311908882120753 w_k1_MAT_s005
                                 + 0.0345248018791655 w_k1_MAT_s006
                                 + 0.0404592319389239 w_k1_MAT_s007
                                 + 0.0567212783181492 w_k1_MAT_s010
                                 + 0.0505414419194645 w_k1_MAT_s011
                                 + 0.0328956567870768 w_k1_MAT_s012
                                 + 0.0194868732785116 w_k1_MAT_s013
                                 + 0.0596259297873095 w_k1_MAT_s014
                                 + 0.0157439225177118 w_k1_MAT_s015
                                 + 0.0330828834031936 w_k1_MAT_s016
                                 + 0.0405566910536853 w_k1_MAT_s017
                                 + 0.117860432612808 w_k1_MAT_s100
                                 + 0.0370372280301408 w_k1_MAT_s101
                                 + 0.0823621259171231 w_k1_MAT_s102
                                 + 0.0164308794013137 w_k1_MAT_s103
                                 + 0.0582486578255397 w_k1_MAT_s104
                                 + 0.0514568473899447 w_k1_MAT_s105
                                 + 0.0434279944354464 w_k1_MAT_s106
                                 + 0.0354402316130562 w_k1_MAT_s107
                                 + 0.0263504160276947 w_k1_MAT_s110
                                 + 0.0160601382187633 w_k1_MAT_s111
                                 + 0.00596802864687111 w_k1_MAT_s112
                                 + 0.0087061556739945 w_k1_MAT_s113
                                 + 0.00654171182043657 w_k1_MAT_s114
                                 + 0.00695804099371203 w_k1_MAT_s115
                                 + 0.0210980750346759 w_k1_MAT_s116
                                 + 0.00653685809986339 w_k1_MAT_s117
                                 + 0.0351954502569229 w_k1_MSFT_s000
                                 + 0.00908552757237751 w_k1_MSFT_s001
                                 + 0.00946845134000822 w_k1_MSFT_s002
                                 + 0.0319630056985719 w_k1_MSFT_s003
                                 + 0.0138724627739086 w_k1_MSFT_s004
                                 + 0.0314690633043544 w_k1_MSFT_s005
                                 + 0.0414262796287102 w_k1_MSFT_s006
                                 + 0.0380213889635204 w_k1_MSFT_s007
                                 + 0.0582469769051222 w_k1_MSFT_s010
                                 + 0.0457265757134886 w_k1_MSFT_s011
                                 + 0.0297071230606713 w_k1_MSFT_s012
                                 + 0.0213100848476689 w_k1_MSFT_s013
                                 + 0.0458853236730672 w_k1_MSFT_s014
                                 + 0.0135786803162578 w_k1_MSFT_s015
                                 + 0.0350186580917007 w_k1_MSFT_s016
                                 + 0.0380967661233033 w_k1_MSFT_s017
                                 + 0.0976538027361863 w_k1_MSFT_s100
                                 + 0.0394152521657525 w_k1_MSFT_s101
                                 + 0.087489938206834 w_k1_MSFT_s102
                                 + 0.0201185222988644 w_k1_MSFT_s103
                                 + 0.0531968319808622 w_k1_MSFT_s104
                                 + 0.0492837151232899 w_k1_MSFT_s105
                                 + 0.0424031150498193 w_k1_MSFT_s106
                                 + 0.0380191747951028 w_k1_MSFT_s107
                                 + 0.0276997560533108 w_k1_MSFT_s110
                                 + 0.0136137267205844 w_k1_MSFT_s111
                                 + 0.00704052976544764 w_k1_MSFT_s112
                                 + 0.00994713333160928 w_k1_MSFT_s113
                                 + 0.00636475409492882 w_k1_MSFT_s114
                                 + 0.0071853117136437 w_k1_MSFT_s115
                                 + 0.0191124348234533 w_k1_MSFT_s116
                                 + 0.00796928953937442 w_k1_MSFT_s117
                                 + 0.0360176342701435 w_k1_PG_s000
                                 + 0.00949410580488677 w_k1_PG_s001
                                 + 0.0101900660256887 w_k1_PG_s002
                                 + 0.029728022702571 w_k1_PG_s003
                                 + 0.0150896192460996 w_k1_PG_s004
                                 + 0.0324390665218429 w_k1_PG_s005
                                 + 0.0435060935550319 w_k1_PG_s006
                                 + 0.036015680203307 w_k1_PG_s007
                                 + 0.0548869726034173 w_k1_PG_s010
                                 + 0.0491355729088482 w_k1_PG_s011
                                 + 0.0316156563153207 w_k1_PG_s012
                                 + 0.019814167508361 w_k1_PG_s013
                                 + 0.0461177051137023 w_k1_PG_s014
                                 + 0.0138235188907475 w_k1_PG_s015
                                 + 0.0364022911393983 w_k1_PG_s016
                                 + 0.0337879357311123 w_k1_PG_s017
                                 + 0.10159724932563 w_k1_PG_s100
                                 + 0.0393484146997616 w_k1_PG_s101
                                 + 0.0737604354446026 w_k1_PG_s102
                                 + 0.0206305591144578 w_k1_PG_s103
                                 + 0.0535084535081215 w_k1_PG_s104
                                 + 0.0508079304119725 w_k1_PG_s105
                                 + 0.043434395394893 w_k1_PG_s106
                                 + 0.0356943517244458 w_k1_PG_s107
                                 + 0.0302331016613805 w_k1_PG_s110
                                 + 0.0152560116234067 w_k1_PG_s111
                                 + 0.0067857551448056 w_k1_PG_s112
                                 + 0.00988731273280596 w_k1_PG_s113
                                 + 0.00610782418744114 w_k1_PG_s114
                                 + 0.00622568512828682 w_k1_PG_s115
                                 + 0.0192663403973975 w_k1_PG_s116
                                 + 0.00744772769038722 w_k1_PG_s117 >= 1.229
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
