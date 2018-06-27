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
 w_k1_ADM_s0_return_constraint:  1.21393 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.94936 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.941129 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.06634 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.820497 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.03488 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.28092 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.22074 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.13254 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.924878 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.1467 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.81676 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.805089 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.888947 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.0064 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.723977 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.656941 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.836745 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.24033 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.690087 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.792185 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.25212 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.98257 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.855775 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.07336 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.04397 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.929864 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.12991 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.827673 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.30479 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.16626 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.887067 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.00635 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.19203 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.48129 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.06769 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.14698 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.965305 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.23539 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.922537 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.930661 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.24676 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.852141 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.20855 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.74005 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.45641 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.969353 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.13378 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.39253 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.891598 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.776335 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.862927 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.02429 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.814095 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.609489 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.758248 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.24184 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.724356 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.11758 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.47787 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.998095 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.834984 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.830382 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.891099 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.957815 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.10913 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.64169 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.39624 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.0746 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.711078 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.923491 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.16051 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.47574 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.624113 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.04383 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.806572 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.33782 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.850086 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.948998 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.27971 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.827811 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.29374 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.90102 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.28937 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.03301 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.755419 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.19925 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.11076 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.596825 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.864124 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.965828 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.733622 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.531878 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.726873 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.63842 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.571099 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.06068 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.29414 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.751779 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.731161 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.882849 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.19525 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.08377 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.32873 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.658702 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.36447 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.984558 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.603087 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.778721 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.06986 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.6653 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.873233 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.26066 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.817202 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.21089 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.922092 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.00463 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.07882 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.826749 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.16169 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.53423 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.01189 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.09289 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.822574 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.21101 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.949908 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.758631 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.836061 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.04152 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.797998 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.607712 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.842834 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.29991 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.700614 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.9414 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.38143 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.961254 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.836908 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.936367 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.18233 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.930885 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.25835 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.886017 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.24832 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.06186 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.754898 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.00614 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.987159 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.31182 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.911803 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.21064 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.802799 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.13781 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.961986 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.889628 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.55869 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.00499 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.08363 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.32318 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.56028 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.610441 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.24083 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.951665 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.36082 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.657032 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.00598 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.761539 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.08268 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.02324 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.780712 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.38679 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.727237 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.23786 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.923397 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.7182 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.919217 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.975091 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.706598 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.66105 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.93478 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.510839 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.44145 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.963065 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.567832 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.62515 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.52086 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.25295 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.724258 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.675204 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.973039 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.11146 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.916905 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.933167 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.08741 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.917662 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.06501 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.37316 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.26541 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.04959 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.00322 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.22085 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.16634 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.961243 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.946384 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.888031 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.8957 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.750055 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.03509 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.6064 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.818026 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.13726 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.31146 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.989885 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.822866 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.908279 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.0755 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.941542 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.14171 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.83325 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.19587 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.20223 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.800874 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.00674 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.97275 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.22774 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.935149 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.21159 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.857297 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.16078 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.958757 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.923512 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.30864 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.85729 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.05237 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.43409 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.33959 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.813963 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.00562 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.07364 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.28096 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.689025 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.04588 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.819622 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.8668 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.798324 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.839691 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.43916 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.649101 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.10646 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.18901 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.767037 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.800635 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.974564 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.903548 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.23458 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.14697 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.664726 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.38149 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.09277 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.685637 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.792514 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.44063 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.42975 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.894281 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.00196 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.948042 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.17548 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.02816 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.941268 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.64856 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.991802 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.05158 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.37402 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.59592 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.519357 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.17333 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.08952 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.3007 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.585858 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.03412 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.85373 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.04061 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.771827 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.68826 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.16154 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.662005 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.37309 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.12352 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.678907 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.785526 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.717718 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.632284 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.17689 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.821951 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.506079 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.43943 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.00131 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.623865 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.683391 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.71919 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.44051 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.466854 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.794493 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.899372 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.22593 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.899016 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.935267 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.974061 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.974937 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.21924 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.29591 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.10525 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.06884 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.813356 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.19218 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.06885 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.776542 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.955904 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.06907 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.845039 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.679864 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.968913 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.46436 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.714624 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.05512 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.36867 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.841035 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.723031 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.866526 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.06039 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.87194 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.12804 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.870968 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.17473 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.02652 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.74722 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.964687 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.905141 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.24461 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.967843 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.18826 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.883253 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.09632 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.941015 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.988889 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.16015 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.931166 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.05868 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.27519 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.23823 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.915855 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.9236 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.14641 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.04019 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.827912 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.99923 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.982763 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.83433 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.740737 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.86785 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.17738 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.769963 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.99856 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.23613 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.906444 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.878149 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.886775 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.949374 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.0743 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.993271 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.779816 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.19979 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.02175 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.82158 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.923672 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.16837 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.29921 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.964208 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.04428 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.887496 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.179000102315609 z_k1_s0000
                                 - 0.418001338647329 z_k1_s0001
                                 - 0.304999943750121 z_k1_s0002
                                 - 0.256999989036977 z_k1_s0003
                                 - 0.562000250287013 z_k1_s0004
                                 - 0.528999309078236 z_k1_s0005
                                 - 0.368998401701373 z_k1_s0006
                                 - 0.555999501885571 z_k1_s0007
                                 - 0.151999943300637 z_k1_s0010
                                 - 0.174999941490863 z_k1_s0011
                                 - 0.448999955722254 z_k1_s0012
                                 - 0.270999450754503 z_k1_s0013
                                 - 0.152999925326302 z_k1_s0014
                                 - 0.1340000610387 z_k1_s0015
                                 - 0.411999180172894 z_k1_s0016
                                 - 0.311998919606738 z_k1_s0017
                                 - 0.506998467095645 z_k1_s0020
                                 - 0.255999862662944 z_k1_s0021
                                 - 0.141999776980396 z_k1_s0022
                                 - 0.379999604609279 z_k1_s0023
                                 - 0.134999854155934 z_k1_s0024
                                 - 0.12199985155353 z_k1_s0025
                                 - 0.174999705009667 z_k1_s0026
                                 - 0.325999090907565 z_k1_s0027
                                 - 0.571000432479614 z_k1_s0030
                                 - 0.159999931284034 z_k1_s0031
                                 - 0.196999993163435 z_k1_s0032
                                 - 0.211999820071383 z_k1_s0033
                                 - 0.329000291992614 z_k1_s0034
                                 - 0.229999123521126 z_k1_s0035
                                 - 0.251999136292671 z_k1_s0036
                                 - 0.270999450322515 z_k1_s0037
                                 - 0.455998889375327 z_k1_s0040
                                 - 0.403998977095785 z_k1_s0041
                                 - 0.661997005640542 z_k1_s0042
                                 - 0.411998895323582 z_k1_s0043
                                 - 0.279999356967206 z_k1_s0044
                                 - 0.804997873906431 z_k1_s0045
                                 - 0.955995997606965 z_k1_s0046
                                 - 0.462997597377789 z_k1_s0047
                                 - 0.202999699085677 z_k1_s0050
                                 - 0.402000458977336 z_k1_s0051
                                 - 0.128999902000239 z_k1_s0052
                                 - 0.0990000560466688 z_k1_s0053
                                 - 0.158999913953776 z_k1_s0054
                                 - 0.192999861101132 z_k1_s0055
                                 - 0.339000151674965 z_k1_s0056
                                 - 0.135999954589388 z_k1_s0057
                                 - 0.807998894037908 z_k1_s0060
                                 - 0.410999542797836 z_k1_s0061
                                 - 0.739999069212812 z_k1_s0062
                                 - 0.664000240524458 z_k1_s0063
                                 - 0.733000631831414 z_k1_s0064
                                 - 0.477999353940235 z_k1_s0065
                                 - 1.34599962173915 z_k1_s0066
                                 - 0.347999233587064 z_k1_s0067
                                 - 0.784999971672766 z_k1_s0070
                                 - 0.551998586209056 z_k1_s0071
                                 - 0.598998366720583 z_k1_s0072
                                 - 1.34999951569948 z_k1_s0073
                                 - 0.455999021298311 z_k1_s0074
                                 - 0.81299505251291 z_k1_s0075
                                 - 1.17399668678887 z_k1_s0076
                                 - 0.487999083501373 z_k1_s0077
                                 - 0.214001081203933 z_k1_s0100
                                 - 0.142000632073306 z_k1_s0101
                                 - 0.121000446709994 z_k1_s0102
                                 - 0.222001167351922 z_k1_s0103
                                 - 0.125000489783989 z_k1_s0104
                                 - 0.281001191065957 z_k1_s0105
                                 - 0.204001381270535 z_k1_s0106
                                 - 0.322001428698605 z_k1_s0107
                                 - 0.215000233000627 z_k1_s0110
                                 - 0.121000467745517 z_k1_s0111
                                 - 0.190000524327061 z_k1_s0112
                                 - 0.0570001572981183 z_k1_s0113
                                 - 0.0860002888008194 z_k1_s0114
                                 - 0.17500089472349 z_k1_s0115
                                 - 0.071000170195827 z_k1_s0116
                                 - 0.0630002047401153 z_k1_s0117
                                 - 0.0480000826247383 z_k1_s0120
                                 - 0.0590001909971666 z_k1_s0121
                                 - 0.0290000081815696 z_k1_s0122
                                 - 0.142999977840089 z_k1_s0123
                                 - 0.0610002540649257 z_k1_s0124
                                 - 0.042000036521609 z_k1_s0125
                                 - 0.0570001279294076 z_k1_s0126
                                 - 0.0380000534862388 z_k1_s0127
                                 - 0.118999966395283 z_k1_s0130
                                 - 0.103000209334106 z_k1_s0131
                                 - 0.104000150900362 z_k1_s0132
                                 - 0.0780002861755424 z_k1_s0133
                                 - 0.100000384635339 z_k1_s0134
                                 - 0.0410000954203913 z_k1_s0135
                                 - 0.0839999355730774 z_k1_s0136
                                 - 0.0630001246800791 z_k1_s0137
                                 - 0.155999626549366 z_k1_s0140
                                 - 0.181999564307594 z_k1_s0141
                                 - 0.273000883464095 z_k1_s0142
                                 - 0.229001107027303 z_k1_s0143
                                 - 0.0910000895543377 z_k1_s0144
                                 - 0.176000642749875 z_k1_s0145
                                 - 0.157000215310338 z_k1_s0146
                                 - 0.273000883464095 z_k1_s0147
                                 - 0.0970003850685311 z_k1_s0150
                                 - 0.0480001315946027 z_k1_s0151
                                 - 0.0950002825102015 z_k1_s0152
                                 - 0.0959999807885618 z_k1_s0153
                                 - 0.0580001501851248 z_k1_s0154
                                 - 0.168000142880385 z_k1_s0155
                                 - 0.0830003731618324 z_k1_s0156
                                 - 0.0610001628222976 z_k1_s0157
                                 - 0.223000366698674 z_k1_s0160
                                 - 0.119000150329724 z_k1_s0161
                                 - 0.196999951405946 z_k1_s0162
                                 - 0.0540000752992122 z_k1_s0163
                                 - 0.116000139457537 z_k1_s0164
                                 - 0.0830001402636372 z_k1_s0165
                                 - 0.150999864965846 z_k1_s0166
                                 - 0.260999862278845 z_k1_s0167
                                 - 0.0850000945086385 z_k1_s0170
                                 - 0.130000023307182 z_k1_s0171
                                 - 0.104000133145885 z_k1_s0172
                                 - 0.247000845784621 z_k1_s0173
                                 - 0.106000107081348 z_k1_s0174
                                 - 0.25100033565499 z_k1_s0175
                                 - 0.163999694710191 z_k1_s0176
                                 - 0.0580000456294009 z_k1_s0177
                                 - 0.245999727248285 z_k1_s1000
                                 - 0.259999830295644 z_k1_s1001
                                 - 0.513999084485745 z_k1_s1002
                                 - 0.216999735435703 z_k1_s1003
                                 - 0.360999598481021 z_k1_s1004
                                 - 0.568999710957317 z_k1_s1005
                                 - 0.587999363193448 z_k1_s1006
                                 - 0.348001142949883 z_k1_s1007
                                 - 0.267999383209425 z_k1_s1010
                                 - 0.489999915558844 z_k1_s1011
                                 - 0.296999121037157 z_k1_s1012
                                 - 0.290999202438234 z_k1_s1013
                                 - 0.349998927059389 z_k1_s1014
                                 - 1.11999530643387 z_k1_s1015
                                 - 0.445000526066918 z_k1_s1016
                                 - 0.676999216286385 z_k1_s1017
                                 - 0.181000517854672 z_k1_s1020
                                 - 0.142999801226622 z_k1_s1021
                                 - 0.25200044507416 z_k1_s1022
                                 - 0.13999979976383 z_k1_s1023
                                 - 0.218999140184633 z_k1_s1024
                                 - 0.192999234907012 z_k1_s1025
                                 - 0.301999395454992 z_k1_s1026
                                 - 0.181000517854672 z_k1_s1027
                                 - 0.223999365693206 z_k1_s1030
                                 - 0.16799988466983 z_k1_s1031
                                 - 0.2609998240442 z_k1_s1032
                                 - 0.271999229770322 z_k1_s1033
                                 - 0.153999969363995 z_k1_s1034
                                 - 0.326999862400187 z_k1_s1035
                                 - 0.289999404049193 z_k1_s1036
                                 - 0.105999925086917 z_k1_s1037
                                 - 0.881999231687828 z_k1_s1040
                                 - 0.3469993007213 z_k1_s1041
                                 - 0.265999532068102 z_k1_s1042
                                 - 0.514997070093445 z_k1_s1043
                                 - 0.50599797119737 z_k1_s1044
                                 - 0.73399994619514 z_k1_s1045
                                 - 0.237999466140048 z_k1_s1046
                                 - 0.888998044496594 z_k1_s1047
                                 - 0.507998458478599 z_k1_s1050
                                 - 0.843997764557903 z_k1_s1051
                                 - 0.510000276151885 z_k1_s1052
                                 - 0.42800019931188 z_k1_s1053
                                 - 0.277999120588008 z_k1_s1054
                                 - 0.875997895421976 z_k1_s1055
                                 - 0.233999250848927 z_k1_s1056
                                 - 0.457998512502668 z_k1_s1057
                                 - 0.539999164732851 z_k1_s1060
                                 - 0.300000062462407 z_k1_s1061
                                 - 0.601999770831424 z_k1_s1062
                                 - 0.651001099037897 z_k1_s1063
                                 - 0.981000588596862 z_k1_s1064
                                 - 0.55699984043702 z_k1_s1065
                                 - 0.863001473117818 z_k1_s1066
                                 - 0.771000913422975 z_k1_s1067
                                 - 1.33799791092263 z_k1_s1070
                                 - 0.535999851118142 z_k1_s1071
                                 - 0.619999621376968 z_k1_s1072
                                 - 0.472000228787594 z_k1_s1073
                                 - 1.08499791900978 z_k1_s1074
                                 - 0.561999830664918 z_k1_s1075
                                 - 1.49900048468519 z_k1_s1076
                                 - 0.980000865686082 z_k1_s1077
                                 - 0.32200095174977 z_k1_s1100
                                 - 0.35100101664922 z_k1_s1101
                                 - 0.449002609220518 z_k1_s1102
                                 - 0.431002759290406 z_k1_s1103
                                 - 0.282000671880983 z_k1_s1104
                                 - 0.66900296559524 z_k1_s1105
                                 - 0.604999994277479 z_k1_s1106
                                 - 0.834002247123274 z_k1_s1107
                                 - 0.852000797605243 z_k1_s1110
                                 - 0.74300246776219 z_k1_s1111
                                 - 0.485000357782283 z_k1_s1112
                                 - 0.328000311588875 z_k1_s1113
                                 - 0.680000998952707 z_k1_s1114
                                 - 0.397999960398505 z_k1_s1115
                                 - 1.00600224402744 z_k1_s1116
                                 - 0.332000374218034 z_k1_s1117
                                 - 0.328000073904484 z_k1_s1120
                                 - 0.173000073621578 z_k1_s1121
                                 - 0.296000710035852 z_k1_s1122
                                 - 0.275000556840725 z_k1_s1123
                                 - 0.158999971491493 z_k1_s1124
                                 - 0.472000254813792 z_k1_s1125
                                 - 0.179000030434458 z_k1_s1126
                                 - 0.147000057865733 z_k1_s1127
                                 - 1.20699889900827 z_k1_s1130
                                 - 1.43499578152697 z_k1_s1131
                                 - 0.960995787663173 z_k1_s1132
                                 - 0.473999993863797 z_k1_s1133
                                 - 1.80499955192464 z_k1_s1134
                                 - 0.498999946350313 z_k1_s1135
                                 - 0.48599967281751 z_k1_s1136
                                 - 0.588999775301773 z_k1_s1137
                                 - 0.862002691764373 z_k1_s1140
                                 - 0.347000593620085 z_k1_s1141
                                 - 0.418000870326551 z_k1_s1142
                                 - 0.590001692404521 z_k1_s1143
                                 - 1.55300546824158 z_k1_s1144
                                 - 0.403000735994033 z_k1_s1145
                                 - 0.431000807181007 z_k1_s1146
                                 - 0.783002163846837 z_k1_s1147
                                 - 0.275000328164356 z_k1_s1150
                                 - 0.499999123479673 z_k1_s1151
                                 - 0.14800003872218 z_k1_s1152
                                 - 0.362998721927818 z_k1_s1153
                                 - 0.391999209095995 z_k1_s1154
                                 - 0.711001326943796 z_k1_s1155
                                 - 0.299000273127371 z_k1_s1156
                                 - 0.552999231510651 z_k1_s1157
                                 - 0.399000756377388 z_k1_s1160
                                 - 0.161000048341645 z_k1_s1161
                                 - 0.180000150759644 z_k1_s1162
                                 - 0.362999657185008 z_k1_s1163
                                 - 0.307999849975218 z_k1_s1164
                                 - 0.450000957899354 z_k1_s1165
                                 - 0.232999690325326 z_k1_s1166
                                 - 0.230000102259506 z_k1_s1167
                                 - 0.671002178513628 z_k1_s1170
                                 - 0.601002220291019 z_k1_s1171
                                 - 0.53000183283527 z_k1_s1172
                                 - 0.202000577916071 z_k1_s1173
                                 - 0.15200055402832 z_k1_s1174
                                 - 0.334001197609534 z_k1_s1175
                                 - 0.62300038038365 z_k1_s1176
                                 - 0.648003589187857 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.26536 w_k1_ADM_s000
                                 + 1.06553 w_k1_AXP_s000
                                 + 1.72488 w_k1_DIS_s000
                                 + 1.19312 w_k1_FDX_s000 + 1.6619 w_k1_IBM_s000
                                 + 1.21255 w_k1_INTC_s000
                                 + 1.57301 w_k1_KO_s000 + 1.22448 w_k1_MAT_s000
                                 + 1.41964 w_k1_MSFT_s000
                                 + 1.23899 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.02783 w_k1_ADM_s000
                                 + 1.03947 w_k1_AXP_s000
                                 + 0.922047 w_k1_DIS_s000
                                 + 0.857237 w_k1_FDX_s000
                                 + 1.41289 w_k1_IBM_s000
                                 + 1.11698 w_k1_INTC_s000
                                 + 1.15388 w_k1_KO_s000 + 1.52725 w_k1_MAT_s000
                                 + 1.08807 w_k1_MSFT_s000
                                 + 1.07214 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.08033 w_k1_ADM_s000
                                 + 0.934603 w_k1_AXP_s000
                                 + 1.40614 w_k1_DIS_s000 + 1.1018 w_k1_FDX_s000
                                 + 0.984991 w_k1_IBM_s000
                                 + 1.22968 w_k1_INTC_s000
                                 + 1.07307 w_k1_KO_s000
                                 + 0.655578 w_k1_MAT_s000
                                 + 1.29075 w_k1_MSFT_s000
                                 + 0.94691 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.35593 w_k1_ADM_s000
                                 + 1.55628 w_k1_AXP_s000
                                 + 1.50508 w_k1_DIS_s000
                                 + 1.45104 w_k1_FDX_s000
                                 + 0.911053 w_k1_IBM_s000
                                 + 1.39095 w_k1_INTC_s000
                                 + 1.14956 w_k1_KO_s000 + 1.28008 w_k1_MAT_s000
                                 + 1.4318 w_k1_MSFT_s000 + 1.19381 w_k1_PG_s000
                                 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.830899 w_k1_ADM_s000
                                 + 0.65749 w_k1_AXP_s000
                                 + 0.92177 w_k1_DIS_s000
                                 + 0.83111 w_k1_FDX_s000
                                 + 0.795754 w_k1_IBM_s000
                                 + 1.15086 w_k1_INTC_s000
                                 + 1.05372 w_k1_KO_s000
                                 + 0.901914 w_k1_MAT_s000
                                 + 1.12537 w_k1_MSFT_s000
                                 + 0.943847 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.03502 w_k1_ADM_s000
                                 + 0.917386 w_k1_AXP_s000
                                 + 0.925382 w_k1_DIS_s000
                                 + 1.09397 w_k1_FDX_s000
                                 + 0.62848 w_k1_IBM_s000
                                 + 1.07055 w_k1_INTC_s000
                                 + 0.825912 w_k1_KO_s000
                                 + 0.759249 w_k1_MAT_s000
                                 + 1.11217 w_k1_MSFT_s000 + 1.0684 w_k1_PG_s000
                                 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.16171 w_k1_ADM_s000
                                 + 1.4316 w_k1_AXP_s000 + 1.18949 w_k1_DIS_s000
                                 + 1.0658 w_k1_FDX_s000 + 1.27462 w_k1_IBM_s000
                                 + 1.20016 w_k1_INTC_s000 + 1.3399 w_k1_KO_s000
                                 + 1.67319 w_k1_MAT_s000
                                 + 1.1174 w_k1_MSFT_s000 + 1.24442 w_k1_PG_s000
                                 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.84272 w_k1_ADM_s000
                                 + 0.792122 w_k1_AXP_s000
                                 + 0.657227 w_k1_DIS_s000
                                 + 0.824169 w_k1_FDX_s000
                                 + 0.98056 w_k1_IBM_s000
                                 + 0.962282 w_k1_INTC_s000
                                 + 0.774096 w_k1_KO_s000 + 1.0371 w_k1_MAT_s000
                                 + 0.874568 w_k1_MSFT_s000
                                 + 0.912915 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.888185 w_k1_ADM_s001
                                 + 1.25471 w_k1_AXP_s001
                                 + 1.57575 w_k1_DIS_s001
                                 + 1.51849 w_k1_FDX_s001
                                 + 1.45507 w_k1_IBM_s001
                                 + 1.28541 w_k1_INTC_s001
                                 + 1.34312 w_k1_KO_s001 + 1.39857 w_k1_MAT_s001
                                 + 1.2938 w_k1_MSFT_s001 + 1.15519 w_k1_PG_s001
                                 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.0563 w_k1_ADM_s001
                                 + 1.13499 w_k1_AXP_s001
                                 + 1.24533 w_k1_DIS_s001
                                 + 1.37801 w_k1_FDX_s001
                                 + 0.804581 w_k1_IBM_s001
                                 + 1.05523 w_k1_INTC_s001
                                 + 1.00989 w_k1_KO_s001
                                 + 0.731224 w_k1_MAT_s001
                                 + 1.17145 w_k1_MSFT_s001
                                 + 0.90544 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.938018 w_k1_ADM_s001
                                 + 0.869202 w_k1_AXP_s001
                                 + 1.08119 w_k1_DIS_s001
                                 + 0.958351 w_k1_FDX_s001
                                 + 1.14834 w_k1_IBM_s001
                                 + 0.853915 w_k1_INTC_s001
                                 + 1.18327 w_k1_KO_s001 + 1.1901 w_k1_MAT_s001
                                 + 0.898626 w_k1_MSFT_s001
                                 + 1.02329 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.82422 w_k1_ADM_s001
                                 + 0.819544 w_k1_AXP_s001
                                 + 1.01147 w_k1_DIS_s001
                                 + 1.09771 w_k1_FDX_s001
                                 + 0.677662 w_k1_IBM_s001
                                 + 0.970673 w_k1_INTC_s001
                                 + 0.903474 w_k1_KO_s001
                                 + 0.622775 w_k1_MAT_s001
                                 + 1.05747 w_k1_MSFT_s001
                                 + 0.969539 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.666925 w_k1_ADM_s001
                                 + 0.629426 w_k1_AXP_s001
                                 + 0.701216 w_k1_DIS_s001
                                 + 0.73257 w_k1_FDX_s001
                                 + 1.05495 w_k1_IBM_s001
                                 + 0.795802 w_k1_INTC_s001
                                 + 0.875086 w_k1_KO_s001
                                 + 0.935982 w_k1_MAT_s001
                                 + 0.739875 w_k1_MSFT_s001
                                 + 0.717934 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.732809 w_k1_ADM_s001
                                 + 0.791432 w_k1_AXP_s001
                                 + 1.0652 w_k1_DIS_s001
                                 + 0.978679 w_k1_FDX_s001
                                 + 1.67384 w_k1_IBM_s001
                                 + 0.871814 w_k1_INTC_s001
                                 + 1.23545 w_k1_KO_s001 + 1.8818 w_k1_MAT_s001
                                 + 1.15148 w_k1_MSFT_s001
                                 + 0.961137 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.10117 w_k1_ADM_s001
                                 + 1.10768 w_k1_AXP_s001
                                 + 1.14299 w_k1_DIS_s001
                                 + 1.21776 w_k1_FDX_s001
                                 + 1.09284 w_k1_IBM_s001
                                 + 1.17428 w_k1_INTC_s001
                                 + 1.26408 w_k1_KO_s001
                                 + 0.980001 w_k1_MAT_s001
                                 + 1.13971 w_k1_MSFT_s001
                                 + 1.07036 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.663649 w_k1_ADM_s001
                                 + 0.707585 w_k1_AXP_s001
                                 + 0.708742 w_k1_DIS_s001
                                 + 0.842457 w_k1_FDX_s001
                                 + 0.749233 w_k1_IBM_s001
                                 + 0.955201 w_k1_INTC_s001
                                 + 0.877799 w_k1_KO_s001
                                 + 0.648456 w_k1_MAT_s001
                                 + 0.979566 w_k1_MSFT_s001
                                 + 0.767787 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.66926 w_k1_ADM_s002
                                 + 0.787976 w_k1_AXP_s002
                                 + 0.794731 w_k1_DIS_s002
                                 + 0.817542 w_k1_FDX_s002
                                 + 1.0222 w_k1_IBM_s002
                                 + 0.962268 w_k1_INTC_s002
                                 + 0.912836 w_k1_KO_s002
                                 + 0.943378 w_k1_MAT_s002
                                 + 0.867948 w_k1_MSFT_s002
                                 + 0.913007 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.04684 w_k1_ADM_s002
                                 + 1.11879 w_k1_AXP_s002
                                 + 1.26338 w_k1_DIS_s002
                                 + 1.17003 w_k1_FDX_s002
                                 + 1.13063 w_k1_IBM_s002
                                 + 1.31156 w_k1_INTC_s002
                                 + 1.39587 w_k1_KO_s002 + 1.17685 w_k1_MAT_s002
                                 + 1.22174 w_k1_MSFT_s002
                                 + 1.21348 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.847386 w_k1_ADM_s002
                                 + 0.96018 w_k1_AXP_s002
                                 + 0.878679 w_k1_DIS_s002
                                 + 1.04822 w_k1_FDX_s002
                                 + 0.699117 w_k1_IBM_s002
                                 + 1.15948 w_k1_INTC_s002
                                 + 0.861735 w_k1_KO_s002
                                 + 1.04502 w_k1_MAT_s002
                                 + 1.15794 w_k1_MSFT_s002
                                 + 1.01856 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.0288 w_k1_ADM_s002
                                 + 1.08543 w_k1_AXP_s002
                                 + 0.777995 w_k1_DIS_s002
                                 + 0.908185 w_k1_FDX_s002
                                 + 0.940865 w_k1_IBM_s002
                                 + 0.934348 w_k1_INTC_s002
                                 + 0.844808 w_k1_KO_s002
                                 + 1.23695 w_k1_MAT_s002
                                 + 0.912402 w_k1_MSFT_s002
                                 + 0.955148 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.06502 w_k1_ADM_s002
                                 + 1.66276 w_k1_AXP_s002
                                 + 0.930751 w_k1_DIS_s002
                                 + 1.25908 w_k1_FDX_s002
                                 + 1.13064 w_k1_IBM_s002
                                 + 1.15088 w_k1_INTC_s002
                                 + 1.04703 w_k1_KO_s002 + 1.91351 w_k1_MAT_s002
                                 + 1.02673 w_k1_MSFT_s002
                                 + 1.29015 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.917794 w_k1_ADM_s002
                                 + 0.94056 w_k1_AXP_s002
                                 + 0.994457 w_k1_DIS_s002
                                 + 1.05012 w_k1_FDX_s002
                                 + 0.935873 w_k1_IBM_s002
                                 + 1.20406 w_k1_INTC_s002
                                 + 1.08149 w_k1_KO_s002
                                 + 0.882682 w_k1_MAT_s002
                                 + 1.06025 w_k1_MSFT_s002
                                 + 1.09306 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.22668 w_k1_ADM_s002
                                 + 1.52752 w_k1_AXP_s002
                                 + 1.68891 w_k1_DIS_s002
                                 + 1.37452 w_k1_FDX_s002
                                 + 1.42676 w_k1_IBM_s002
                                 + 1.68949 w_k1_INTC_s002
                                 + 1.52307 w_k1_KO_s002 + 2.09963 w_k1_MAT_s002
                                 + 1.36605 w_k1_MSFT_s002
                                 + 1.35776 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.05562 w_k1_ADM_s002
                                 + 1.11744 w_k1_AXP_s002
                                 + 1.09574 w_k1_DIS_s002
                                 + 1.01934 w_k1_FDX_s002
                                 + 1.43972 w_k1_IBM_s002
                                 + 1.32024 w_k1_INTC_s002
                                 + 1.36922 w_k1_KO_s002 + 1.23973 w_k1_MAT_s002
                                 + 1.07573 w_k1_MSFT_s002
                                 + 1.04625 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.722209 w_k1_ADM_s003
                                 + 0.718527 w_k1_AXP_s003
                                 + 0.686854 w_k1_DIS_s003
                                 + 0.824275 w_k1_FDX_s003
                                 + 0.784595 w_k1_IBM_s003
                                 + 1.03649 w_k1_INTC_s003
                                 + 0.801439 w_k1_KO_s003
                                 + 0.695885 w_k1_MAT_s003
                                 + 0.920704 w_k1_MSFT_s003
                                 + 0.906634 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.39114 w_k1_ADM_s003
                                 + 1.31554 w_k1_AXP_s003
                                 + 1.15248 w_k1_DIS_s003 + 1.1254 w_k1_FDX_s003
                                 + 1.40799 w_k1_IBM_s003
                                 + 0.834813 w_k1_INTC_s003
                                 + 1.23467 w_k1_KO_s003 + 2.03185 w_k1_MAT_s003
                                 + 1.02578 w_k1_MSFT_s003
                                 + 1.16313 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.104 w_k1_ADM_s003
                                 + 1.11505 w_k1_AXP_s003
                                 + 1.19301 w_k1_DIS_s003
                                 + 1.21169 w_k1_FDX_s003
                                 + 0.886132 w_k1_IBM_s003
                                 + 1.05786 w_k1_INTC_s003
                                 + 0.934314 w_k1_KO_s003
                                 + 1.07577 w_k1_MAT_s003
                                 + 1.17399 w_k1_MSFT_s003
                                 + 1.03113 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.05655 w_k1_ADM_s003
                                 + 1.09194 w_k1_AXP_s003
                                 + 1.19733 w_k1_DIS_s003
                                 + 0.940521 w_k1_FDX_s003
                                 + 0.976615 w_k1_IBM_s003
                                 + 1.00334 w_k1_INTC_s003 + 1.0592 w_k1_KO_s003
                                 + 1.10831 w_k1_MAT_s003
                                 + 1.15148 w_k1_MSFT_s003
                                 + 1.03628 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.89878 w_k1_ADM_s003
                                 + 0.841849 w_k1_AXP_s003
                                 + 0.84028 w_k1_DIS_s003
                                 + 0.859493 w_k1_FDX_s003
                                 + 1.32192 w_k1_IBM_s003
                                 + 0.912839 w_k1_INTC_s003
                                 + 1.06416 w_k1_KO_s003 + 1.32528 w_k1_MAT_s003
                                 + 0.876889 w_k1_MSFT_s003
                                 + 1.06524 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.809413 w_k1_ADM_s003
                                 + 0.758293 w_k1_AXP_s003
                                 + 0.893534 w_k1_DIS_s003
                                 + 0.931583 w_k1_FDX_s003
                                 + 1.12564 w_k1_IBM_s003
                                 + 0.999947 w_k1_INTC_s003
                                 + 0.960483 w_k1_KO_s003 + 1.0343 w_k1_MAT_s003
                                 + 0.899138 w_k1_MSFT_s003
                                 + 1.00753 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.12165 w_k1_ADM_s003
                                 + 1.32514 w_k1_AXP_s003
                                 + 1.68332 w_k1_DIS_s003
                                 + 1.30189 w_k1_FDX_s003
                                 + 1.49597 w_k1_IBM_s003
                                 + 1.17727 w_k1_INTC_s003
                                 + 1.40621 w_k1_KO_s003 + 1.77188 w_k1_MAT_s003
                                 + 1.40563 w_k1_MSFT_s003
                                 + 1.26188 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.5495 w_k1_ADM_s003
                                 + 1.59305 w_k1_AXP_s003
                                 + 1.51881 w_k1_DIS_s003
                                 + 1.51173 w_k1_FDX_s003
                                 + 0.917832 w_k1_IBM_s003
                                 + 1.15629 w_k1_INTC_s003
                                 + 1.20893 w_k1_KO_s003 + 1.36202 w_k1_MAT_s003
                                 + 1.30478 w_k1_MSFT_s003
                                 + 1.37042 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.17959 w_k1_ADM_s004
                                 + 0.993326 w_k1_AXP_s004
                                 + 1.13524 w_k1_DIS_s004
                                 + 0.972366 w_k1_FDX_s004
                                 + 1.52715 w_k1_IBM_s004
                                 + 0.922935 w_k1_INTC_s004
                                 + 1.32313 w_k1_KO_s004 + 1.13735 w_k1_MAT_s004
                                 + 0.92446 w_k1_MSFT_s004
                                 + 0.999092 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.18707 w_k1_ADM_s004
                                 + 1.52893 w_k1_AXP_s004
                                 + 1.33412 w_k1_DIS_s004
                                 + 1.22757 w_k1_FDX_s004
                                 + 1.20973 w_k1_IBM_s004
                                 + 1.14854 w_k1_INTC_s004
                                 + 1.15576 w_k1_KO_s004 + 1.09598 w_k1_MAT_s004
                                 + 1.04226 w_k1_MSFT_s004
                                 + 1.01737 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.990506 w_k1_ADM_s004
                                 + 0.895008 w_k1_AXP_s004
                                 + 0.981344 w_k1_DIS_s004
                                 + 0.968893 w_k1_FDX_s004
                                 + 1.12894 w_k1_IBM_s004
                                 + 0.939942 w_k1_INTC_s004
                                 + 1.10167 w_k1_KO_s004 + 1.00361 w_k1_MAT_s004
                                 + 0.917207 w_k1_MSFT_s004
                                 + 0.956769 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.21603 w_k1_ADM_s004
                                 + 0.926566 w_k1_AXP_s004
                                 + 1.52522 w_k1_DIS_s004
                                 + 1.24216 w_k1_FDX_s004
                                 + 0.820298 w_k1_IBM_s004
                                 + 1.21911 w_k1_INTC_s004
                                 + 1.05995 w_k1_KO_s004
                                 + 0.774399 w_k1_MAT_s004
                                 + 1.23382 w_k1_MSFT_s004
                                 + 1.10883 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.53692 w_k1_ADM_s004
                                 + 1.21636 w_k1_AXP_s004
                                 + 1.65337 w_k1_DIS_s004 + 1.297 w_k1_FDX_s004
                                 + 1.58267 w_k1_IBM_s004
                                 + 1.27922 w_k1_INTC_s004
                                 + 1.69514 w_k1_KO_s004 + 1.23106 w_k1_MAT_s004
                                 + 1.32873 w_k1_MSFT_s004 + 1.2492 w_k1_PG_s004
                                 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.10171 w_k1_ADM_s004
                                 + 0.826051 w_k1_AXP_s004
                                 + 1.08443 w_k1_DIS_s004
                                 + 1.04768 w_k1_FDX_s004
                                 + 0.796976 w_k1_IBM_s004
                                 + 1.06941 w_k1_INTC_s004
                                 + 0.890799 w_k1_KO_s004
                                 + 0.806828 w_k1_MAT_s004
                                 + 0.986098 w_k1_MSFT_s004
                                 + 0.941607 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.883936 w_k1_ADM_s004
                                 + 0.638958 w_k1_AXP_s004
                                 + 0.785559 w_k1_DIS_s004
                                 + 0.800569 w_k1_FDX_s004
                                 + 0.947804 w_k1_IBM_s004
                                 + 0.864234 w_k1_INTC_s004
                                 + 1.00711 w_k1_KO_s004
                                 + 0.610093 w_k1_MAT_s004
                                 + 0.787362 w_k1_MSFT_s004
                                 + 0.843907 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.83663 w_k1_ADM_s004
                                 + 0.973624 w_k1_AXP_s004
                                 + 0.72963 w_k1_DIS_s004
                                 + 0.870638 w_k1_FDX_s004
                                 + 0.739093 w_k1_IBM_s004
                                 + 1.03936 w_k1_INTC_s004
                                 + 0.795931 w_k1_KO_s004
                                 + 0.842763 w_k1_MAT_s004
                                 + 0.829971 w_k1_MSFT_s004
                                 + 0.888426 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.35862 w_k1_ADM_s005
                                 + 1.27066 w_k1_AXP_s005
                                 + 1.29897 w_k1_DIS_s005
                                 + 1.15102 w_k1_FDX_s005
                                 + 1.56173 w_k1_IBM_s005
                                 + 1.03041 w_k1_INTC_s005
                                 + 1.22989 w_k1_KO_s005 + 1.60277 w_k1_MAT_s005
                                 + 1.08315 w_k1_MSFT_s005 + 1.1205 w_k1_PG_s005
                                 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.11988 w_k1_ADM_s005
                                 + 0.930886 w_k1_AXP_s005
                                 + 1.14657 w_k1_DIS_s005
                                 + 1.02751 w_k1_FDX_s005
                                 + 0.874511 w_k1_IBM_s005
                                 + 1.05166 w_k1_INTC_s005
                                 + 1.02077 w_k1_KO_s005
                                 + 0.868924 w_k1_MAT_s005
                                 + 1.05044 w_k1_MSFT_s005
                                 + 1.07495 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.99462 w_k1_ADM_s005
                                 + 0.692301 w_k1_AXP_s005
                                 + 0.556356 w_k1_DIS_s005
                                 + 0.736959 w_k1_FDX_s005
                                 + 0.797827 w_k1_IBM_s005
                                 + 0.908716 w_k1_INTC_s005
                                 + 0.690684 w_k1_KO_s005 + 0.5356 w_k1_MAT_s005
                                 + 0.781794 w_k1_MSFT_s005
                                 + 0.757206 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.30222 w_k1_ADM_s005
                                 + 1.52706 w_k1_AXP_s005
                                 + 1.04326 w_k1_DIS_s005
                                 + 1.16286 w_k1_FDX_s005
                                 + 1.02013 w_k1_IBM_s005 + 1.014 w_k1_INTC_s005
                                 + 1.1319 w_k1_KO_s005 + 1.42264 w_k1_MAT_s005
                                 + 0.966298 w_k1_MSFT_s005
                                 + 1.17015 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.19177 w_k1_ADM_s005
                                 + 1.47999 w_k1_AXP_s005
                                 + 1.71251 w_k1_DIS_s005 + 1.414 w_k1_FDX_s005
                                 + 1.06783 w_k1_IBM_s005
                                 + 1.3343 w_k1_INTC_s005 + 1.27791 w_k1_KO_s005
                                 + 1.10205 w_k1_MAT_s005
                                 + 1.53303 w_k1_MSFT_s005
                                 + 1.19434 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.0493 w_k1_ADM_s005
                                 + 1.08434 w_k1_AXP_s005
                                 + 0.846518 w_k1_DIS_s005
                                 + 0.837739 w_k1_FDX_s005 + 1.312 w_k1_IBM_s005
                                 + 0.894463 w_k1_INTC_s005
                                 + 1.10783 w_k1_KO_s005 + 1.48943 w_k1_MAT_s005
                                 + 0.949319 w_k1_MSFT_s005
                                 + 0.998048 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.914592 w_k1_ADM_s005
                                 + 0.920781 w_k1_AXP_s005
                                 + 0.780202 w_k1_DIS_s005
                                 + 0.857036 w_k1_FDX_s005
                                 + 0.838162 w_k1_IBM_s005
                                 + 0.774368 w_k1_INTC_s005
                                 + 0.883919 w_k1_KO_s005 + 1.0254 w_k1_MAT_s005
                                 + 0.845411 w_k1_MSFT_s005
                                 + 0.973769 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.918476 w_k1_ADM_s005
                                 + 0.933507 w_k1_AXP_s005
                                 + 1.12889 w_k1_DIS_s005
                                 + 0.923855 w_k1_FDX_s005
                                 + 1.15747 w_k1_IBM_s005
                                 + 0.981584 w_k1_INTC_s005
                                 + 1.09881 w_k1_KO_s005
                                 + 0.992699 w_k1_MAT_s005
                                 + 1.07616 w_k1_MSFT_s005
                                 + 1.06434 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.34333 w_k1_ADM_s006
                                 + 1.09154 w_k1_AXP_s006
                                 + 1.46324 w_k1_DIS_s006
                                 + 1.09041 w_k1_FDX_s006 + 1.2741 w_k1_IBM_s006
                                 + 0.965088 w_k1_INTC_s006
                                 + 1.44313 w_k1_KO_s006 + 1.38447 w_k1_MAT_s006
                                 + 1.14008 w_k1_MSFT_s006
                                 + 1.11959 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.54857 w_k1_ADM_s006
                                 + 1.34728 w_k1_AXP_s006 + 1.3535 w_k1_DIS_s006
                                 + 1.51275 w_k1_FDX_s006
                                 + 0.822421 w_k1_IBM_s006
                                 + 1.13708 w_k1_INTC_s006
                                 + 1.09993 w_k1_KO_s006
                                 + 0.964103 w_k1_MAT_s006
                                 + 1.34192 w_k1_MSFT_s006
                                 + 1.14782 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.864363 w_k1_ADM_s006
                                 + 0.569339 w_k1_AXP_s006
                                 + 0.582541 w_k1_DIS_s006
                                 + 0.670584 w_k1_FDX_s006
                                 + 0.859551 w_k1_IBM_s006
                                 + 0.775598 w_k1_INTC_s006
                                 + 0.760867 w_k1_KO_s006
                                 + 0.768713 w_k1_MAT_s006
                                 + 0.800276 w_k1_MSFT_s006
                                 + 0.774502 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.89255 w_k1_ADM_s006
                                 + 0.844856 w_k1_AXP_s006
                                 + 0.813381 w_k1_DIS_s006
                                 + 0.757888 w_k1_FDX_s006
                                 + 1.23262 w_k1_IBM_s006
                                 + 0.812498 w_k1_INTC_s006
                                 + 1.12871 w_k1_KO_s006 + 1.20699 w_k1_MAT_s006
                                 + 0.814105 w_k1_MSFT_s006
                                 + 0.954641 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.0786 w_k1_ADM_s006
                                 + 1.23191 w_k1_AXP_s006
                                 + 1.00828 w_k1_DIS_s006
                                 + 0.957326 w_k1_FDX_s006
                                 + 1.29265 w_k1_IBM_s006
                                 + 0.843888 w_k1_INTC_s006
                                 + 1.02058 w_k1_KO_s006 + 1.33364 w_k1_MAT_s006
                                 + 0.911638 w_k1_MSFT_s006
                                 + 1.04137 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.10127 w_k1_ADM_s006
                                 + 1.12457 w_k1_AXP_s006
                                 + 0.845266 w_k1_DIS_s006
                                 + 0.998031 w_k1_FDX_s006
                                 + 0.726612 w_k1_IBM_s006
                                 + 0.852531 w_k1_INTC_s006
                                 + 0.778898 w_k1_KO_s006
                                 + 1.10146 w_k1_MAT_s006
                                 + 0.921017 w_k1_MSFT_s006
                                 + 0.819854 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.21339 w_k1_ADM_s006
                                 + 1.10294 w_k1_AXP_s006
                                 + 1.28597 w_k1_DIS_s006
                                 + 1.24326 w_k1_FDX_s006
                                 + 0.952661 w_k1_IBM_s006
                                 + 1.24247 w_k1_INTC_s006
                                 + 1.18436 w_k1_KO_s006
                                 + 0.757861 w_k1_MAT_s006
                                 + 1.20378 w_k1_MSFT_s006
                                 + 1.05887 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.798153 w_k1_ADM_s006
                                 + 0.514303 w_k1_AXP_s006
                                 + 0.930275 w_k1_DIS_s006
                                 + 0.846027 w_k1_FDX_s006
                                 + 0.669482 w_k1_IBM_s006
                                 + 0.763872 w_k1_INTC_s006
                                 + 0.817531 w_k1_KO_s006
                                 + 0.63112 w_k1_MAT_s006
                                 + 0.99634 w_k1_MSFT_s006
                                 + 0.806165 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.756386 w_k1_ADM_s007
                                 + 0.714665 w_k1_AXP_s007
                                 + 0.627061 w_k1_DIS_s007
                                 + 0.739934 w_k1_FDX_s007
                                 + 0.955751 w_k1_IBM_s007
                                 + 0.827903 w_k1_INTC_s007
                                 + 0.704698 w_k1_KO_s007
                                 + 0.717739 w_k1_MAT_s007
                                 + 0.712825 w_k1_MSFT_s007
                                 + 0.802378 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.25366 w_k1_ADM_s007
                                 + 1.6233 w_k1_AXP_s007 + 1.80883 w_k1_DIS_s007
                                 + 1.42596 w_k1_FDX_s007
                                 + 1.37238 w_k1_IBM_s007
                                 + 1.59903 w_k1_INTC_s007
                                 + 1.30646 w_k1_KO_s007 + 1.28307 w_k1_MAT_s007
                                 + 1.27658 w_k1_MSFT_s007 + 1.1902 w_k1_PG_s007
                                 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.991718 w_k1_ADM_s007
                                 + 0.945014 w_k1_AXP_s007
                                 + 1.30212 w_k1_DIS_s007
                                 + 1.06734 w_k1_FDX_s007
                                 + 1.23361 w_k1_IBM_s007
                                 + 0.982662 w_k1_INTC_s007
                                 + 1.17103 w_k1_KO_s007 + 1.18572 w_k1_MAT_s007
                                 + 1.16658 w_k1_MSFT_s007
                                 + 0.978839 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.985807 w_k1_ADM_s007
                                 + 0.987232 w_k1_AXP_s007
                                 + 0.903094 w_k1_DIS_s007
                                 + 0.884888 w_k1_FDX_s007
                                 + 1.11248 w_k1_IBM_s007
                                 + 1.01835 w_k1_INTC_s007
                                 + 0.874026 w_k1_KO_s007
                                 + 0.910525 w_k1_MAT_s007
                                 + 0.764524 w_k1_MSFT_s007
                                 + 0.881292 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.685135 w_k1_ADM_s007
                                 + 0.5977 w_k1_AXP_s007
                                 + 0.722694 w_k1_DIS_s007
                                 + 0.78767 w_k1_FDX_s007
                                 + 0.650347 w_k1_IBM_s007
                                 + 0.863302 w_k1_INTC_s007
                                 + 0.642754 w_k1_KO_s007
                                 + 0.496362 w_k1_MAT_s007
                                 + 0.749316 w_k1_MSFT_s007
                                 + 0.763675 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.21253 w_k1_ADM_s007
                                 + 1.30534 w_k1_AXP_s007
                                 + 1.16146 w_k1_DIS_s007
                                 + 1.24896 w_k1_FDX_s007
                                 + 1.23289 w_k1_IBM_s007
                                 + 1.13062 w_k1_INTC_s007
                                 + 1.07921 w_k1_KO_s007 + 1.08131 w_k1_MAT_s007
                                 + 0.98629 w_k1_MSFT_s007
                                 + 1.05153 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.958615 w_k1_ADM_s007
                                 + 1.06249 w_k1_AXP_s007
                                 + 1.15352 w_k1_DIS_s007
                                 + 1.18758 w_k1_FDX_s007
                                 + 0.848343 w_k1_IBM_s007
                                 + 1.27155 w_k1_INTC_s007
                                 + 0.958663 w_k1_KO_s007
                                 + 0.900063 w_k1_MAT_s007
                                 + 1.01484 w_k1_MSFT_s007
                                 + 1.01224 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.973304 w_k1_ADM_s007
                                 + 0.936415 w_k1_AXP_s007
                                 + 0.718754 w_k1_DIS_s007
                                 + 0.955491 w_k1_FDX_s007
                                 + 0.694972 w_k1_IBM_s007
                                 + 1.05988 w_k1_INTC_s007
                                 + 0.608132 w_k1_KO_s007
                                 + 0.581207 w_k1_MAT_s007
                                 + 0.818087 w_k1_MSFT_s007
                                 + 0.800671 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.879812 w_k1_ADM_s010
                                 + 0.714584 w_k1_AXP_s010
                                 + 0.873887 w_k1_DIS_s010
                                 + 0.880007 w_k1_FDX_s010
                                 + 1.06315 w_k1_IBM_s010
                                 + 1.14399 w_k1_INTC_s010
                                 + 1.12681 w_k1_KO_s010
                                 + 0.888712 w_k1_MAT_s010
                                 + 1.03942 w_k1_MSFT_s010
                                 + 0.922048 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.05857 w_k1_ADM_s010
                                 + 1.13485 w_k1_AXP_s010
                                 + 1.18098 w_k1_DIS_s010
                                 + 1.15992 w_k1_FDX_s010
                                 + 0.877759 w_k1_IBM_s010
                                 + 1.42177 w_k1_INTC_s010
                                 + 1.14044 w_k1_KO_s010
                                 + 0.867726 w_k1_MAT_s010
                                 + 1.06779 w_k1_MSFT_s010
                                 + 1.09892 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.14233 w_k1_ADM_s010
                                 + 1.34088 w_k1_AXP_s010 + 1.0052 w_k1_DIS_s010
                                 + 1.18702 w_k1_FDX_s010
                                 + 1.07688 w_k1_IBM_s010
                                 + 1.13468 w_k1_INTC_s010
                                 + 1.05777 w_k1_KO_s010 + 1.16269 w_k1_MAT_s010
                                 + 0.99655 w_k1_MSFT_s010 + 1.1236 w_k1_PG_s010
                                 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.06314 w_k1_ADM_s010
                                 + 0.974936 w_k1_AXP_s010
                                 + 1.0412 w_k1_DIS_s010 + 1.0706 w_k1_FDX_s010
                                 + 0.749702 w_k1_IBM_s010
                                 + 0.923113 w_k1_INTC_s010
                                 + 1.03684 w_k1_KO_s010
                                 + 0.995698 w_k1_MAT_s010
                                 + 0.995597 w_k1_MSFT_s010
                                 + 1.00017 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.01276 w_k1_ADM_s010
                                 + 0.884814 w_k1_AXP_s010
                                 + 1.42915 w_k1_DIS_s010
                                 + 0.946259 w_k1_FDX_s010
                                 + 1.14742 w_k1_IBM_s010
                                 + 1.1506 w_k1_INTC_s010 + 1.28077 w_k1_KO_s010
                                 + 0.604494 w_k1_MAT_s010
                                 + 1.0639 w_k1_MSFT_s010 + 1.01177 w_k1_PG_s010
                                 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.03025 w_k1_ADM_s010
                                 + 0.896347 w_k1_AXP_s010
                                 + 0.895111 w_k1_DIS_s010
                                 + 0.890453 w_k1_FDX_s010
                                 + 1.24956 w_k1_IBM_s010
                                 + 1.14199 w_k1_INTC_s010
                                 + 1.05312 w_k1_KO_s010 + 0.84131 w_k1_MAT_s010
                                 + 0.88057 w_k1_MSFT_s010
                                 + 0.951091 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.22769 w_k1_ADM_s010
                                 + 1.38585 w_k1_AXP_s010
                                 + 1.53064 w_k1_DIS_s010
                                 + 1.23407 w_k1_FDX_s010
                                 + 1.54373 w_k1_IBM_s010
                                 + 1.1187 w_k1_INTC_s010 + 1.52869 w_k1_KO_s010
                                 + 1.58339 w_k1_MAT_s010
                                 + 1.2338 w_k1_MSFT_s010 + 1.20607 w_k1_PG_s010
                                 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.929037 w_k1_ADM_s010
                                 + 0.705909 w_k1_AXP_s010
                                 + 0.635142 w_k1_DIS_s010
                                 + 0.778074 w_k1_FDX_s010
                                 + 0.739686 w_k1_IBM_s010
                                 + 0.815328 w_k1_INTC_s010
                                 + 0.723966 w_k1_KO_s010
                                 + 0.56867 w_k1_MAT_s010
                                 + 0.775558 w_k1_MSFT_s010
                                 + 0.805196 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.12126 w_k1_ADM_s011
                                 + 0.81951 w_k1_AXP_s011
                                 + 0.703355 w_k1_DIS_s011
                                 + 0.859327 w_k1_FDX_s011
                                 + 0.774064 w_k1_IBM_s011
                                 + 0.881219 w_k1_INTC_s011
                                 + 0.834084 w_k1_KO_s011
                                 + 0.873862 w_k1_MAT_s011
                                 + 0.928979 w_k1_MSFT_s011
                                 + 0.933719 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.22341 w_k1_ADM_s011
                                 + 1.22489 w_k1_AXP_s011
                                 + 0.990842 w_k1_DIS_s011
                                 + 1.11522 w_k1_FDX_s011
                                 + 0.799952 w_k1_IBM_s011
                                 + 1.01139 w_k1_INTC_s011
                                 + 1.02041 w_k1_KO_s011
                                 + 0.743579 w_k1_MAT_s011
                                 + 1.02208 w_k1_MSFT_s011
                                 + 1.09128 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.40876 w_k1_ADM_s011
                                 + 1.48034 w_k1_AXP_s011 + 1.4464 w_k1_DIS_s011
                                 + 1.36631 w_k1_FDX_s011
                                 + 1.39787 w_k1_IBM_s011
                                 + 1.26214 w_k1_INTC_s011
                                 + 1.22578 w_k1_KO_s011 + 1.54017 w_k1_MAT_s011
                                 + 1.13663 w_k1_MSFT_s011
                                 + 1.28814 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.15208 w_k1_ADM_s011
                                 + 1.23785 w_k1_AXP_s011
                                 + 1.00696 w_k1_DIS_s011
                                 + 1.14444 w_k1_FDX_s011 + 1.1495 w_k1_IBM_s011
                                 + 1.00637 w_k1_INTC_s011
                                 + 1.28749 w_k1_KO_s011 + 1.04928 w_k1_MAT_s011
                                 + 0.976454 w_k1_MSFT_s011
                                 + 1.22995 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.19204 w_k1_ADM_s011
                                 + 0.875273 w_k1_AXP_s011
                                 + 0.90644 w_k1_DIS_s011
                                 + 0.985566 w_k1_FDX_s011
                                 + 1.04619 w_k1_IBM_s011
                                 + 0.823672 w_k1_INTC_s011
                                 + 1.10771 w_k1_KO_s011
                                 + 0.861619 w_k1_MAT_s011
                                 + 1.07315 w_k1_MSFT_s011
                                 + 1.05879 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.30949 w_k1_ADM_s011
                                 + 1.14895 w_k1_AXP_s011
                                 + 1.38074 w_k1_DIS_s011
                                 + 1.15695 w_k1_FDX_s011
                                 + 0.958551 w_k1_IBM_s011
                                 + 1.10387 w_k1_INTC_s011
                                 + 1.20866 w_k1_KO_s011
                                 + 0.887718 w_k1_MAT_s011
                                 + 1.21756 w_k1_MSFT_s011
                                 + 1.19633 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.01958 w_k1_ADM_s011
                                 + 1.0733 w_k1_AXP_s011
                                 + 0.927522 w_k1_DIS_s011
                                 + 0.909668 w_k1_FDX_s011
                                 + 1.20509 w_k1_IBM_s011
                                 + 0.906696 w_k1_INTC_s011
                                 + 1.07882 w_k1_KO_s011
                                 + 0.841794 w_k1_MAT_s011
                                 + 0.903074 w_k1_MSFT_s011
                                 + 0.954364 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.987081 w_k1_ADM_s011
                                 + 0.766892 w_k1_AXP_s011
                                 + 0.702216 w_k1_DIS_s011
                                 + 0.736346 w_k1_FDX_s011
                                 + 1.36455 w_k1_IBM_s011
                                 + 0.777953 w_k1_INTC_s011
                                 + 1.23114 w_k1_KO_s011 + 1.20288 w_k1_MAT_s011
                                 + 0.918899 w_k1_MSFT_s011
                                 + 1.17981 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.08319 w_k1_ADM_s012
                                 + 1.30599 w_k1_AXP_s012
                                 + 1.58636 w_k1_DIS_s012
                                 + 1.27382 w_k1_FDX_s012
                                 + 1.25684 w_k1_IBM_s012
                                 + 1.22203 w_k1_INTC_s012
                                 + 1.29596 w_k1_KO_s012 + 1.36379 w_k1_MAT_s012
                                 + 1.37017 w_k1_MSFT_s012
                                 + 1.26589 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.14608 w_k1_ADM_s012
                                 + 1.134 w_k1_AXP_s012 + 0.994922 w_k1_DIS_s012
                                 + 0.992869 w_k1_FDX_s012
                                 + 0.756368 w_k1_IBM_s012
                                 + 1.23346 w_k1_INTC_s012
                                 + 0.855578 w_k1_KO_s012
                                 + 0.826537 w_k1_MAT_s012
                                 + 1.14082 w_k1_MSFT_s012
                                 + 1.06688 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.03079 w_k1_ADM_s012
                                 + 0.826856 w_k1_AXP_s012
                                 + 0.839358 w_k1_DIS_s012
                                 + 0.832388 w_k1_FDX_s012
                                 + 1.04003 w_k1_IBM_s012
                                 + 0.755889 w_k1_INTC_s012
                                 + 0.979464 w_k1_KO_s012
                                 + 0.860481 w_k1_MAT_s012
                                 + 0.824043 w_k1_MSFT_s012
                                 + 0.987168 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.99095 w_k1_ADM_s012
                                 + 1.13858 w_k1_AXP_s012
                                 + 1.00815 w_k1_DIS_s012
                                 + 1.00243 w_k1_FDX_s012
                                 + 0.929544 w_k1_IBM_s012
                                 + 1.15645 w_k1_INTC_s012
                                 + 1.07116 w_k1_KO_s012 + 1.14105 w_k1_MAT_s012
                                 + 1.19431 w_k1_MSFT_s012 + 1.0943 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.01597 w_k1_ADM_s012
                                 + 1.17313 w_k1_AXP_s012
                                 + 1.13826 w_k1_DIS_s012
                                 + 0.993313 w_k1_FDX_s012
                                 + 1.32795 w_k1_IBM_s012
                                 + 1.09311 w_k1_INTC_s012
                                 + 1.19551 w_k1_KO_s012 + 1.66468 w_k1_MAT_s012
                                 + 1.18994 w_k1_MSFT_s012
                                 + 1.11732 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.809014 w_k1_ADM_s012
                                 + 0.667099 w_k1_AXP_s012
                                 + 0.54024 w_k1_DIS_s012
                                 + 0.666771 w_k1_FDX_s012
                                 + 0.746661 w_k1_IBM_s012
                                 + 0.736178 w_k1_INTC_s012
                                 + 0.689423 w_k1_KO_s012
                                 + 0.69686 w_k1_MAT_s012
                                 + 0.765804 w_k1_MSFT_s012
                                 + 0.796763 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.06822 w_k1_ADM_s012
                                 + 0.874083 w_k1_AXP_s012
                                 + 0.895746 w_k1_DIS_s012
                                 + 1.06109 w_k1_FDX_s012
                                 + 0.595433 w_k1_IBM_s012
                                 + 1.07944 w_k1_INTC_s012
                                 + 0.763077 w_k1_KO_s012
                                 + 0.383609 w_k1_MAT_s012
                                 + 1.05518 w_k1_MSFT_s012
                                 + 0.870173 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.14017 w_k1_ADM_s012
                                 + 1.32832 w_k1_AXP_s012
                                 + 1.66421 w_k1_DIS_s012
                                 + 1.16636 w_k1_FDX_s012
                                 + 1.71525 w_k1_IBM_s012
                                 + 1.4287 w_k1_INTC_s012 + 1.62949 w_k1_KO_s012
                                 + 1.47837 w_k1_MAT_s012
                                 + 1.58996 w_k1_MSFT_s012
                                 + 1.55478 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.866216 w_k1_ADM_s013
                                 + 0.812015 w_k1_AXP_s013
                                 + 0.792058 w_k1_DIS_s013
                                 + 0.8824 w_k1_FDX_s013
                                 + 0.651553 w_k1_IBM_s013
                                 + 0.705934 w_k1_INTC_s013
                                 + 0.734099 w_k1_KO_s013
                                 + 0.802194 w_k1_MAT_s013
                                 + 0.879904 w_k1_MSFT_s013
                                 + 0.926049 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.27356 w_k1_ADM_s013
                                 + 1.34946 w_k1_AXP_s013
                                 + 1.47793 w_k1_DIS_s013
                                 + 1.22144 w_k1_FDX_s013
                                 + 1.46328 w_k1_IBM_s013
                                 + 1.22836 w_k1_INTC_s013
                                 + 1.53893 w_k1_KO_s013 + 1.65624 w_k1_MAT_s013
                                 + 1.2711 w_k1_MSFT_s013 + 1.27985 w_k1_PG_s013
                                 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.970132 w_k1_ADM_s013
                                 + 1.05547 w_k1_AXP_s013
                                 + 0.95308 w_k1_DIS_s013
                                 + 0.837656 w_k1_FDX_s013
                                 + 0.997896 w_k1_IBM_s013
                                 + 0.732164 w_k1_INTC_s013
                                 + 1.0013 w_k1_KO_s013 + 1.40737 w_k1_MAT_s013
                                 + 0.855615 w_k1_MSFT_s013
                                 + 1.09604 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.767177 w_k1_ADM_s013
                                 + 1.1189 w_k1_AXP_s013 + 1.08971 w_k1_DIS_s013
                                 + 0.895117 w_k1_FDX_s013
                                 + 1.32646 w_k1_IBM_s013
                                 + 0.817238 w_k1_INTC_s013
                                 + 1.02755 w_k1_KO_s013 + 1.22281 w_k1_MAT_s013
                                 + 0.744367 w_k1_MSFT_s013
                                 + 0.888956 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.19821 w_k1_ADM_s013
                                 + 1.07574 w_k1_AXP_s013 + 1.0765 w_k1_DIS_s013
                                 + 1.14736 w_k1_FDX_s013
                                 + 0.890983 w_k1_IBM_s013
                                 + 1.11823 w_k1_INTC_s013
                                 + 1.12976 w_k1_KO_s013 + 1.08295 w_k1_MAT_s013
                                 + 1.08133 w_k1_MSFT_s013
                                 + 1.20658 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.1923 w_k1_ADM_s013
                                 + 1.69836 w_k1_AXP_s013
                                 + 1.15618 w_k1_DIS_s013
                                 + 1.12216 w_k1_FDX_s013 + 1.2655 w_k1_IBM_s013
                                 + 0.979828 w_k1_INTC_s013
                                 + 1.2344 w_k1_KO_s013 + 1.29284 w_k1_MAT_s013
                                 + 0.926647 w_k1_MSFT_s013
                                 + 1.06692 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.933304 w_k1_ADM_s013
                                 + 0.974594 w_k1_AXP_s013
                                 + 1.4613 w_k1_DIS_s013 + 1.13482 w_k1_FDX_s013
                                 + 1.00408 w_k1_IBM_s013
                                 + 1.1379 w_k1_INTC_s013 + 1.22008 w_k1_KO_s013
                                 + 1.22092 w_k1_MAT_s013
                                 + 1.23056 w_k1_MSFT_s013
                                 + 1.12177 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.59853 w_k1_ADM_s013
                                 + 0.945377 w_k1_AXP_s013
                                 + 0.655832 w_k1_DIS_s013
                                 + 0.789058 w_k1_FDX_s013
                                 + 0.861253 w_k1_IBM_s013
                                 + 0.723856 w_k1_INTC_s013
                                 + 0.85114 w_k1_KO_s013
                                 + 0.969712 w_k1_MAT_s013
                                 + 0.722621 w_k1_MSFT_s013
                                 + 0.797617 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.12949 w_k1_ADM_s014
                                 + 1.2167 w_k1_AXP_s014
                                 + 0.928348 w_k1_DIS_s014
                                 + 0.919408 w_k1_FDX_s014
                                 + 0.98463 w_k1_IBM_s014
                                 + 0.695138 w_k1_INTC_s014
                                 + 0.932082 w_k1_KO_s014
                                 + 1.42212 w_k1_MAT_s014
                                 + 0.825418 w_k1_MSFT_s014
                                 + 0.977717 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.717358 w_k1_ADM_s014
                                 + 0.513545 w_k1_AXP_s014
                                 + 0.567585 w_k1_DIS_s014
                                 + 0.719898 w_k1_FDX_s014
                                 + 0.706387 w_k1_IBM_s014
                                 + 0.572639 w_k1_INTC_s014
                                 + 0.735242 w_k1_KO_s014
                                 + 0.810611 w_k1_MAT_s014
                                 + 0.740444 w_k1_MSFT_s014
                                 + 0.853687 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.28852 w_k1_ADM_s014
                                 + 1.42271 w_k1_AXP_s014
                                 + 1.35055 w_k1_DIS_s014
                                 + 1.18798 w_k1_FDX_s014
                                 + 0.861873 w_k1_IBM_s014
                                 + 0.858608 w_k1_INTC_s014
                                 + 1.12809 w_k1_KO_s014 + 1.3007 w_k1_MAT_s014
                                 + 1.06455 w_k1_MSFT_s014
                                 + 1.13485 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.911274 w_k1_ADM_s014
                                 + 0.857688 w_k1_AXP_s014
                                 + 0.890654 w_k1_DIS_s014
                                 + 0.889424 w_k1_FDX_s014
                                 + 0.656238 w_k1_IBM_s014
                                 + 0.725876 w_k1_INTC_s014
                                 + 0.812817 w_k1_KO_s014
                                 + 0.552667 w_k1_MAT_s014
                                 + 0.910056 w_k1_MSFT_s014
                                 + 0.954198 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.844313 w_k1_ADM_s014
                                 + 0.781518 w_k1_AXP_s014
                                 + 0.763109 w_k1_DIS_s014
                                 + 0.861087 w_k1_FDX_s014
                                 + 1.06242 w_k1_IBM_s014
                                 + 0.776101 w_k1_INTC_s014
                                 + 0.838894 w_k1_KO_s014
                                 + 0.825795 w_k1_MAT_s014
                                 + 0.78958 w_k1_MSFT_s014
                                 + 0.849239 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.00856 w_k1_ADM_s014
                                 + 0.861298 w_k1_AXP_s014
                                 + 1.11641 w_k1_DIS_s014
                                 + 1.02516 w_k1_FDX_s014
                                 + 0.907857 w_k1_IBM_s014
                                 + 0.90569 w_k1_INTC_s014
                                 + 1.12415 w_k1_KO_s014 + 1.1006 w_k1_MAT_s014
                                 + 1.05247 w_k1_MSFT_s014
                                 + 1.10923 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.24846 w_k1_ADM_s014
                                 + 1.97322 w_k1_AXP_s014
                                 + 1.87223 w_k1_DIS_s014
                                 + 1.55148 w_k1_FDX_s014
                                 + 1.36553 w_k1_IBM_s014
                                 + 1.18688 w_k1_INTC_s014
                                 + 1.26096 w_k1_KO_s014 + 1.57178 w_k1_MAT_s014
                                 + 1.24572 w_k1_MSFT_s014
                                 + 1.24505 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.659753 w_k1_ADM_s014
                                 + 1.14574 w_k1_AXP_s014
                                 + 0.843735 w_k1_DIS_s014
                                 + 0.842657 w_k1_FDX_s014
                                 + 1.42269 w_k1_IBM_s014
                                 + 0.822938 w_k1_INTC_s014
                                 + 1.05447 w_k1_KO_s014 + 1.58734 w_k1_MAT_s014
                                 + 0.891686 w_k1_MSFT_s014
                                 + 1.04346 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.49762 w_k1_ADM_s015
                                 + 1.02096 w_k1_AXP_s015
                                 + 1.15338 w_k1_DIS_s015
                                 + 1.21341 w_k1_FDX_s015
                                 + 1.17691 w_k1_IBM_s015
                                 + 1.17189 w_k1_INTC_s015
                                 + 1.23813 w_k1_KO_s015
                                 + 0.792052 w_k1_MAT_s015
                                 + 1.18683 w_k1_MSFT_s015
                                 + 1.18372 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.904817 w_k1_ADM_s015
                                 + 0.651302 w_k1_AXP_s015
                                 + 0.615714 w_k1_DIS_s015
                                 + 0.77497 w_k1_FDX_s015
                                 + 0.63544 w_k1_IBM_s015
                                 + 0.869806 w_k1_INTC_s015
                                 + 0.657362 w_k1_KO_s015
                                 + 0.47148 w_k1_MAT_s015
                                 + 0.894957 w_k1_MSFT_s015
                                 + 0.791527 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.05373 w_k1_ADM_s015
                                 + 1.09275 w_k1_AXP_s015
                                 + 1.43669 w_k1_DIS_s015
                                 + 1.30022 w_k1_FDX_s015
                                 + 1.27303 w_k1_IBM_s015
                                 + 1.60449 w_k1_INTC_s015
                                 + 1.45242 w_k1_KO_s015
                                 + 0.955557 w_k1_MAT_s015
                                 + 1.50564 w_k1_MSFT_s015
                                 + 1.26052 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.41691 w_k1_ADM_s015
                                 + 0.942205 w_k1_AXP_s015
                                 + 1.01077 w_k1_DIS_s015
                                 + 1.15601 w_k1_FDX_s015
                                 + 0.737706 w_k1_IBM_s015
                                 + 0.903475 w_k1_INTC_s015
                                 + 1.01133 w_k1_KO_s015
                                 + 0.645253 w_k1_MAT_s015
                                 + 1.11662 w_k1_MSFT_s015
                                 + 1.07511 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.10294 w_k1_ADM_s015
                                 + 1.19314 w_k1_AXP_s015 + 1.3567 w_k1_DIS_s015
                                 + 1.11315 w_k1_FDX_s015
                                 + 1.72548 w_k1_IBM_s015
                                 + 1.05138 w_k1_INTC_s015
                                 + 1.56452 w_k1_KO_s015 + 1.68038 w_k1_MAT_s015
                                 + 1.28838 w_k1_MSFT_s015
                                 + 1.22474 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.922074 w_k1_ADM_s015
                                 + 0.733301 w_k1_AXP_s015
                                 + 0.78386 w_k1_DIS_s015
                                 + 0.853141 w_k1_FDX_s015
                                 + 0.838669 w_k1_IBM_s015
                                 + 1.00201 w_k1_INTC_s015
                                 + 0.862288 w_k1_KO_s015
                                 + 0.536395 w_k1_MAT_s015
                                 + 0.989783 w_k1_MSFT_s015
                                 + 0.85009 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.23424 w_k1_ADM_s015
                                 + 0.886888 w_k1_AXP_s015
                                 + 0.959223 w_k1_DIS_s015
                                 + 1.0457 w_k1_FDX_s015
                                 + 0.935642 w_k1_IBM_s015
                                 + 0.901991 w_k1_INTC_s015
                                 + 1.11218 w_k1_KO_s015
                                 + 0.786186 w_k1_MAT_s015
                                 + 0.993428 w_k1_MSFT_s015
                                 + 1.07588 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.06098 w_k1_ADM_s015
                                 + 0.882363 w_k1_AXP_s015
                                 + 0.783318 w_k1_DIS_s015
                                 + 0.833737 w_k1_FDX_s015
                                 + 1.19816 w_k1_IBM_s015
                                 + 0.757287 w_k1_INTC_s015
                                 + 1.09641 w_k1_KO_s015 + 1.39043 w_k1_MAT_s015
                                 + 0.969464 w_k1_MSFT_s015
                                 + 0.988597 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.16599 w_k1_ADM_s016
                                 + 0.806119 w_k1_AXP_s016
                                 + 0.780865 w_k1_DIS_s016
                                 + 0.912615 w_k1_FDX_s016
                                 + 0.981113 w_k1_IBM_s016
                                 + 0.701074 w_k1_INTC_s016
                                 + 0.880994 w_k1_KO_s016
                                 + 0.921218 w_k1_MAT_s016
                                 + 0.853325 w_k1_MSFT_s016
                                 + 0.970042 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.1521 w_k1_ADM_s016
                                 + 1.10984 w_k1_AXP_s016
                                 + 0.857529 w_k1_DIS_s016
                                 + 0.988177 w_k1_FDX_s016
                                 + 1.2412 w_k1_IBM_s016
                                 + 0.935359 w_k1_INTC_s016
                                 + 1.06613 w_k1_KO_s016 + 1.26947 w_k1_MAT_s016
                                 + 0.885975 w_k1_MSFT_s016
                                 + 1.03177 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.34521 w_k1_ADM_s016
                                 + 0.954376 w_k1_AXP_s016
                                 + 1.20117 w_k1_DIS_s016
                                 + 1.09741 w_k1_FDX_s016
                                 + 0.971691 w_k1_IBM_s016
                                 + 0.939096 w_k1_INTC_s016
                                 + 1.24165 w_k1_KO_s016
                                 + 0.860542 w_k1_MAT_s016
                                 + 1.18409 w_k1_MSFT_s016
                                 + 1.11207 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.98067 w_k1_ADM_s016
                                 + 1.18682 w_k1_AXP_s016 + 1.7102 w_k1_DIS_s016
                                 + 1.49569 w_k1_FDX_s016
                                 + 1.03207 w_k1_IBM_s016
                                 + 1.15128 w_k1_INTC_s016
                                 + 1.37618 w_k1_KO_s016
                                 + 0.765689 w_k1_MAT_s016
                                 + 1.43725 w_k1_MSFT_s016
                                 + 1.21621 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.19103 w_k1_ADM_s016
                                 + 1.41395 w_k1_AXP_s016
                                 + 1.28671 w_k1_DIS_s016
                                 + 1.32905 w_k1_FDX_s016
                                 + 1.00357 w_k1_IBM_s016
                                 + 1.08536 w_k1_INTC_s016
                                 + 1.17782 w_k1_KO_s016 + 1.2036 w_k1_MAT_s016
                                 + 1.22624 w_k1_MSFT_s016
                                 + 1.19301 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.860686 w_k1_ADM_s016
                                 + 0.980049 w_k1_AXP_s016
                                 + 1.35774 w_k1_DIS_s016
                                 + 1.12845 w_k1_FDX_s016
                                 + 1.44576 w_k1_IBM_s016
                                 + 1.02089 w_k1_INTC_s016 + 1.3941 w_k1_KO_s016
                                 + 1.73504 w_k1_MAT_s016
                                 + 1.07686 w_k1_MSFT_s016
                                 + 1.31427 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.33243 w_k1_ADM_s016
                                 + 1.22154 w_k1_AXP_s016
                                 + 1.02223 w_k1_DIS_s016
                                 + 1.12615 w_k1_FDX_s016
                                 + 1.57012 w_k1_IBM_s016
                                 + 0.98061 w_k1_INTC_s016
                                 + 1.15817 w_k1_KO_s016 + 1.29547 w_k1_MAT_s016
                                 + 0.975018 w_k1_MSFT_s016
                                 + 1.08975 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.29728 w_k1_ADM_s016
                                 + 0.754231 w_k1_AXP_s016
                                 + 0.752987 w_k1_DIS_s016
                                 + 0.943168 w_k1_FDX_s016
                                 + 0.662234 w_k1_IBM_s016
                                 + 0.865652 w_k1_INTC_s016
                                 + 0.820017 w_k1_KO_s016
                                 + 0.628534 w_k1_MAT_s016
                                 + 0.854325 w_k1_MSFT_s016
                                 + 0.989925 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.699389 w_k1_ADM_s017
                                 + 0.690662 w_k1_AXP_s017
                                 + 0.673129 w_k1_DIS_s017
                                 + 0.787924 w_k1_FDX_s017
                                 + 0.741332 w_k1_IBM_s017
                                 + 0.821098 w_k1_INTC_s017
                                 + 0.620561 w_k1_KO_s017
                                 + 0.531253 w_k1_MAT_s017
                                 + 0.801606 w_k1_MSFT_s017
                                 + 0.695279 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.706127 w_k1_ADM_s017
                                 + 0.744437 w_k1_AXP_s017
                                 + 0.595948 w_k1_DIS_s017
                                 + 0.730864 w_k1_FDX_s017
                                 + 0.915589 w_k1_IBM_s017
                                 + 0.78935 w_k1_INTC_s017
                                 + 0.666632 w_k1_KO_s017
                                 + 0.895539 w_k1_MAT_s017
                                 + 0.74822 w_k1_MSFT_s017
                                 + 0.710169 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.39239 w_k1_ADM_s017
                                 + 1.36101 w_k1_AXP_s017
                                 + 1.54917 w_k1_DIS_s017
                                 + 1.25563 w_k1_FDX_s017
                                 + 1.32636 w_k1_IBM_s017
                                 + 1.23465 w_k1_INTC_s017
                                 + 1.32151 w_k1_KO_s017 + 1.23943 w_k1_MAT_s017
                                 + 1.23694 w_k1_MSFT_s017 + 1.2348 w_k1_PG_s017
                                 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.90822 w_k1_ADM_s017
                                 + 1.09433 w_k1_AXP_s017
                                 + 0.990447 w_k1_DIS_s017
                                 + 0.981774 w_k1_FDX_s017
                                 + 1.29608 w_k1_IBM_s017
                                 + 1.00886 w_k1_INTC_s017
                                 + 0.934101 w_k1_KO_s017
                                 + 1.32134 w_k1_MAT_s017
                                 + 0.934075 w_k1_MSFT_s017
                                 + 1.07329 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.13254 w_k1_ADM_s017
                                 + 1.48617 w_k1_AXP_s017
                                 + 1.25733 w_k1_DIS_s017
                                 + 1.04761 w_k1_FDX_s017
                                 + 1.62193 w_k1_IBM_s017
                                 + 0.89478 w_k1_INTC_s017
                                 + 1.09584 w_k1_KO_s017 + 1.50903 w_k1_MAT_s017
                                 + 0.8562 w_k1_MSFT_s017
                                 + 0.980076 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.937744 w_k1_ADM_s017
                                 + 0.91516 w_k1_AXP_s017
                                 + 0.92888 w_k1_DIS_s017
                                 + 0.942004 w_k1_FDX_s017
                                 + 0.892987 w_k1_IBM_s017
                                 + 0.809755 w_k1_INTC_s017
                                 + 0.855552 w_k1_KO_s017
                                 + 0.83493 w_k1_MAT_s017
                                 + 0.885312 w_k1_MSFT_s017
                                 + 0.964179 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.95592 w_k1_ADM_s017
                                 + 1.00216 w_k1_AXP_s017 + 1.199 w_k1_DIS_s017
                                 + 1.12719 w_k1_FDX_s017
                                 + 0.798148 w_k1_IBM_s017
                                 + 0.990984 w_k1_INTC_s017
                                 + 0.80245 w_k1_KO_s017 + 0.82998 w_k1_MAT_s017
                                 + 0.988779 w_k1_MSFT_s017
                                 + 0.78245 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.19237 w_k1_ADM_s017
                                 + 1.66206 w_k1_AXP_s017
                                 + 1.82094 w_k1_DIS_s017
                                 + 1.35639 w_k1_FDX_s017
                                 + 0.970578 w_k1_IBM_s017
                                 + 1.14373 w_k1_INTC_s017 + 1.1795 w_k1_KO_s017
                                 + 1.41525 w_k1_MAT_s017
                                 + 1.20646 w_k1_MSFT_s017
                                 + 1.11726 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.15238 w_k1_ADM_s100
                                 + 1.19149 w_k1_AXP_s100 + 1.1805 w_k1_DIS_s100
                                 + 1.07344 w_k1_FDX_s100
                                 + 1.16855 w_k1_IBM_s100
                                 + 0.889061 w_k1_INTC_s100
                                 + 1.07036 w_k1_KO_s100 + 1.02727 w_k1_MAT_s100
                                 + 0.842816 w_k1_MSFT_s100
                                 + 0.896298 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.02039 w_k1_ADM_s100
                                 + 0.979114 w_k1_AXP_s100
                                 + 1.09268 w_k1_DIS_s100 + 1.1912 w_k1_FDX_s100
                                 + 0.719506 w_k1_IBM_s100
                                 + 0.896877 w_k1_INTC_s100
                                 + 0.937016 w_k1_KO_s100
                                 + 0.690088 w_k1_MAT_s100
                                 + 0.966597 w_k1_MSFT_s100
                                 + 0.910232 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.88689 w_k1_ADM_s100
                                 + 0.988172 w_k1_AXP_s100
                                 + 1.47307 w_k1_DIS_s100
                                 + 1.06059 w_k1_FDX_s100
                                 + 1.25253 w_k1_IBM_s100
                                 + 0.902215 w_k1_INTC_s100
                                 + 1.28091 w_k1_KO_s100 + 1.39317 w_k1_MAT_s100
                                 + 1.02933 w_k1_MSFT_s100
                                 + 1.18748 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.890785 w_k1_ADM_s100
                                 + 0.745905 w_k1_AXP_s100
                                 + 1.01551 w_k1_DIS_s100
                                 + 0.82803 w_k1_FDX_s100
                                 + 1.40014 w_k1_IBM_s100
                                 + 0.80554 w_k1_INTC_s100
                                 + 1.17606 w_k1_KO_s100
                                 + 0.915006 w_k1_MAT_s100
                                 + 0.815588 w_k1_MSFT_s100
                                 + 0.945124 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.41546 w_k1_ADM_s100
                                 + 1.17122 w_k1_AXP_s100
                                 + 1.53355 w_k1_DIS_s100
                                 + 1.34235 w_k1_FDX_s100
                                 + 1.02034 w_k1_IBM_s100
                                 + 1.13895 w_k1_INTC_s100
                                 + 1.20822 w_k1_KO_s100 + 1.11552 w_k1_MAT_s100
                                 + 1.30664 w_k1_MSFT_s100
                                 + 1.15025 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.801764 w_k1_ADM_s100
                                 + 0.768119 w_k1_AXP_s100
                                 + 0.788669 w_k1_DIS_s100
                                 + 0.839853 w_k1_FDX_s100
                                 + 0.98666 w_k1_IBM_s100
                                 + 0.777356 w_k1_INTC_s100
                                 + 0.864409 w_k1_KO_s100 + 0.975 w_k1_MAT_s100
                                 + 0.833471 w_k1_MSFT_s100
                                 + 0.815657 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.859663 w_k1_ADM_s100
                                 + 0.729693 w_k1_AXP_s100
                                 + 0.84171 w_k1_DIS_s100
                                 + 0.883715 w_k1_FDX_s100
                                 + 0.734245 w_k1_IBM_s100
                                 + 0.784997 w_k1_INTC_s100
                                 + 0.844739 w_k1_KO_s100
                                 + 0.75226 w_k1_MAT_s100
                                 + 0.848188 w_k1_MSFT_s100
                                 + 0.852719 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.25036 w_k1_ADM_s100
                                 + 1.24119 w_k1_AXP_s100
                                 + 1.46382 w_k1_DIS_s100
                                 + 1.38704 w_k1_FDX_s100
                                 + 1.19424 w_k1_IBM_s100
                                 + 1.06614 w_k1_INTC_s100
                                 + 1.26012 w_k1_KO_s100 + 1.51305 w_k1_MAT_s100
                                 + 1.26893 w_k1_MSFT_s100 + 1.2024 w_k1_PG_s100
                                 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.23019 w_k1_ADM_s101
                                 + 1.33171 w_k1_AXP_s101
                                 + 1.23848 w_k1_DIS_s101
                                 + 1.07709 w_k1_FDX_s101
                                 + 1.12232 w_k1_IBM_s101
                                 + 1.14055 w_k1_INTC_s101
                                 + 1.35393 w_k1_KO_s101 + 1.33718 w_k1_MAT_s101
                                 + 1.08962 w_k1_MSFT_s101
                                 + 0.990717 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.875902 w_k1_ADM_s101
                                 + 0.849918 w_k1_AXP_s101
                                 + 0.656881 w_k1_DIS_s101
                                 + 0.730837 w_k1_FDX_s101
                                 + 1.12832 w_k1_IBM_s101
                                 + 1.01893 w_k1_INTC_s101
                                 + 1.01623 w_k1_KO_s101 + 1.04577 w_k1_MAT_s101
                                 + 0.890802 w_k1_MSFT_s101
                                 + 1.02786 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.16435 w_k1_ADM_s101
                                 + 1.08236 w_k1_AXP_s101 + 1.7851 w_k1_DIS_s101
                                 + 1.22338 w_k1_FDX_s101
                                 + 1.40252 w_k1_IBM_s101
                                 + 1.20894 w_k1_INTC_s101
                                 + 1.74268 w_k1_KO_s101 + 1.26618 w_k1_MAT_s101
                                 + 1.69605 w_k1_MSFT_s101
                                 + 1.30794 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.892278 w_k1_ADM_s101
                                 + 0.859903 w_k1_AXP_s101
                                 + 1.09512 w_k1_DIS_s101
                                 + 0.880957 w_k1_FDX_s101
                                 + 1.16563 w_k1_IBM_s101
                                 + 1.23929 w_k1_INTC_s101
                                 + 1.28453 w_k1_KO_s101 + 1.02409 w_k1_MAT_s101
                                 + 1.15678 w_k1_MSFT_s101
                                 + 1.03808 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.24721 w_k1_ADM_s101
                                 + 1.05398 w_k1_AXP_s101
                                 + 1.46584 w_k1_DIS_s101
                                 + 1.19816 w_k1_FDX_s101
                                 + 0.948984 w_k1_IBM_s101
                                 + 1.22412 w_k1_INTC_s101
                                 + 1.22917 w_k1_KO_s101 + 1.10517 w_k1_MAT_s101
                                 + 1.40103 w_k1_MSFT_s101 + 1.2588 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.17145 w_k1_ADM_s101
                                 + 0.982108 w_k1_AXP_s101 + 0.95 w_k1_DIS_s101
                                 + 0.988158 w_k1_FDX_s101
                                 + 0.862012 w_k1_IBM_s101
                                 + 0.935441 w_k1_INTC_s101
                                 + 0.982087 w_k1_KO_s101
                                 + 0.938971 w_k1_MAT_s101
                                 + 0.949292 w_k1_MSFT_s101
                                 + 0.999403 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.00729 w_k1_ADM_s101
                                 + 1.59403 w_k1_AXP_s101
                                 + 1.33546 w_k1_DIS_s101
                                 + 1.12768 w_k1_FDX_s101
                                 + 1.41219 w_k1_IBM_s101
                                 + 1.43555 w_k1_INTC_s101
                                 + 1.19864 w_k1_KO_s101 + 1.58496 w_k1_MAT_s101
                                 + 1.23918 w_k1_MSFT_s101
                                 + 1.10494 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.811452 w_k1_ADM_s101
                                 + 0.590578 w_k1_AXP_s101
                                 + 0.643338 w_k1_DIS_s101
                                 + 0.712088 w_k1_FDX_s101
                                 + 0.748163 w_k1_IBM_s101
                                 + 0.843467 w_k1_INTC_s101
                                 + 0.808982 w_k1_KO_s101
                                 + 0.795015 w_k1_MAT_s101
                                 + 0.856578 w_k1_MSFT_s101
                                 + 0.755879 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.15314 w_k1_ADM_s102
                                 + 1.28033 w_k1_AXP_s102
                                 + 1.04392 w_k1_DIS_s102
                                 + 1.11096 w_k1_FDX_s102
                                 + 1.50606 w_k1_IBM_s102
                                 + 1.02456 w_k1_INTC_s102
                                 + 1.19321 w_k1_KO_s102 + 1.7194 w_k1_MAT_s102
                                 + 0.976996 w_k1_MSFT_s102
                                 + 1.16591 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.13872 w_k1_ADM_s102
                                 + 1.10153 w_k1_AXP_s102
                                 + 1.13206 w_k1_DIS_s102
                                 + 1.00122 w_k1_FDX_s102
                                 + 1.24765 w_k1_IBM_s102
                                 + 1.05354 w_k1_INTC_s102
                                 + 1.10341 w_k1_KO_s102 + 1.14216 w_k1_MAT_s102
                                 + 1.07915 w_k1_MSFT_s102
                                 + 1.10942 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.06636 w_k1_ADM_s102
                                 + 1.06362 w_k1_AXP_s102
                                 + 0.861266 w_k1_DIS_s102
                                 + 1.01532 w_k1_FDX_s102
                                 + 0.941611 w_k1_IBM_s102
                                 + 0.909583 w_k1_INTC_s102
                                 + 0.932256 w_k1_KO_s102
                                 + 1.00638 w_k1_MAT_s102
                                 + 0.978237 w_k1_MSFT_s102
                                 + 1.10369 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.06905 w_k1_ADM_s102
                                 + 1.32856 w_k1_AXP_s102
                                 + 1.20253 w_k1_DIS_s102
                                 + 1.24459 w_k1_FDX_s102
                                 + 0.776717 w_k1_IBM_s102
                                 + 1.22901 w_k1_INTC_s102
                                 + 0.934101 w_k1_KO_s102
                                 + 0.719248 w_k1_MAT_s102
                                 + 1.12778 w_k1_MSFT_s102
                                 + 1.00489 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.859458 w_k1_ADM_s102
                                 + 0.894058 w_k1_AXP_s102
                                 + 0.692533 w_k1_DIS_s102
                                 + 0.877789 w_k1_FDX_s102
                                 + 0.699044 w_k1_IBM_s102
                                 + 0.978443 w_k1_INTC_s102
                                 + 0.743663 w_k1_KO_s102
                                 + 0.613097 w_k1_MAT_s102
                                 + 0.891765 w_k1_MSFT_s102
                                 + 0.964874 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.19327 w_k1_ADM_s102
                                 + 1.09034 w_k1_AXP_s102
                                 + 1.30643 w_k1_DIS_s102
                                 + 1.02533 w_k1_FDX_s102
                                 + 1.05735 w_k1_IBM_s102
                                 + 1.09008 w_k1_INTC_s102
                                 + 1.18376 w_k1_KO_s102 + 1.14431 w_k1_MAT_s102
                                 + 1.11591 w_k1_MSFT_s102
                                 + 1.16202 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.994047 w_k1_ADM_s102
                                 + 0.775757 w_k1_AXP_s102
                                 + 0.67841 w_k1_DIS_s102
                                 + 0.765158 w_k1_FDX_s102
                                 + 0.946877 w_k1_IBM_s102
                                 + 0.853265 w_k1_INTC_s102
                                 + 0.928932 w_k1_KO_s102
                                 + 1.04101 w_k1_MAT_s102
                                 + 0.865265 w_k1_MSFT_s102
                                 + 0.895574 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.5803 w_k1_ADM_s102
                                 + 1.65682 w_k1_AXP_s102 + 1.291 w_k1_DIS_s102
                                 + 1.25368 w_k1_FDX_s102
                                 + 1.18713 w_k1_IBM_s102
                                 + 1.01145 w_k1_INTC_s102
                                 + 1.19284 w_k1_KO_s102 + 1.76904 w_k1_MAT_s102
                                 + 1.05744 w_k1_MSFT_s102
                                 + 1.21478 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.907828 w_k1_ADM_s103
                                 + 0.794188 w_k1_AXP_s103
                                 + 0.837344 w_k1_DIS_s103
                                 + 0.915744 w_k1_FDX_s103
                                 + 0.785757 w_k1_IBM_s103
                                 + 0.904272 w_k1_INTC_s103
                                 + 1.03143 w_k1_KO_s103 + 0.93143 w_k1_MAT_s103
                                 + 1.00592 w_k1_MSFT_s103
                                 + 1.05254 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.734947 w_k1_ADM_s103
                                 + 0.613631 w_k1_AXP_s103
                                 + 0.57673 w_k1_DIS_s103
                                 + 0.791078 w_k1_FDX_s103
                                 + 0.683134 w_k1_IBM_s103
                                 + 0.848217 w_k1_INTC_s103
                                 + 0.670119 w_k1_KO_s103
                                 + 0.67411 w_k1_MAT_s103
                                 + 0.770513 w_k1_MSFT_s103
                                 + 0.817913 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.21019 w_k1_ADM_s103
                                 + 0.941846 w_k1_AXP_s103
                                 + 1.07539 w_k1_DIS_s103
                                 + 1.02078 w_k1_FDX_s103
                                 + 1.04996 w_k1_IBM_s103
                                 + 0.966127 w_k1_INTC_s103
                                 + 1.11294 w_k1_KO_s103
                                 + 0.954769 w_k1_MAT_s103
                                 + 1.12324 w_k1_MSFT_s103
                                 + 1.19626 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.51945 w_k1_ADM_s103
                                 + 1.3087 w_k1_AXP_s103 + 1.36638 w_k1_DIS_s103
                                 + 1.30165 w_k1_FDX_s103
                                 + 1.24727 w_k1_IBM_s103
                                 + 1.02231 w_k1_INTC_s103
                                 + 1.32266 w_k1_KO_s103 + 1.22107 w_k1_MAT_s103
                                 + 1.23346 w_k1_MSFT_s103
                                 + 1.39306 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.06822 w_k1_ADM_s103
                                 + 0.820218 w_k1_AXP_s103
                                 + 0.857324 w_k1_DIS_s103
                                 + 0.794565 w_k1_FDX_s103
                                 + 1.34048 w_k1_IBM_s103
                                 + 0.971939 w_k1_INTC_s103
                                 + 1.3137 w_k1_KO_s103 + 1.15158 w_k1_MAT_s103
                                 + 0.906678 w_k1_MSFT_s103
                                 + 1.11166 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.877063 w_k1_ADM_s103
                                 + 0.722643 w_k1_AXP_s103
                                 + 0.781036 w_k1_DIS_s103
                                 + 0.816747 w_k1_FDX_s103
                                 + 1.03835 w_k1_IBM_s103
                                 + 0.913352 w_k1_INTC_s103
                                 + 0.984697 w_k1_KO_s103
                                 + 0.685394 w_k1_MAT_s103
                                 + 0.812397 w_k1_MSFT_s103
                                 + 1.00604 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.52962 w_k1_ADM_s103
                                 + 1.36539 w_k1_AXP_s103
                                 + 1.06364 w_k1_DIS_s103 + 1.2402 w_k1_FDX_s103
                                 + 0.901058 w_k1_IBM_s103
                                 + 1.12238 w_k1_INTC_s103
                                 + 1.05293 w_k1_KO_s103
                                 + 0.965344 w_k1_MAT_s103
                                 + 1.21914 w_k1_MSFT_s103
                                 + 1.13964 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.22927 w_k1_ADM_s103
                                 + 1.03915 w_k1_AXP_s103
                                 + 0.849714 w_k1_DIS_s103
                                 + 1.09243 w_k1_FDX_s103
                                 + 0.612339 w_k1_IBM_s103
                                 + 0.961119 w_k1_INTC_s103
                                 + 0.907985 w_k1_KO_s103
                                 + 0.757752 w_k1_MAT_s103
                                 + 1.0501 w_k1_MSFT_s103 + 1.06137 w_k1_PG_s103
                                 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.916929 w_k1_ADM_s104
                                 + 0.899546 w_k1_AXP_s104
                                 + 0.909276 w_k1_DIS_s104
                                 + 0.893256 w_k1_FDX_s104
                                 + 1.02473 w_k1_IBM_s104
                                 + 0.949534 w_k1_INTC_s104
                                 + 0.955684 w_k1_KO_s104
                                 + 1.13822 w_k1_MAT_s104
                                 + 1.08105 w_k1_MSFT_s104
                                 + 0.852843 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.937698 w_k1_ADM_s104
                                 + 0.666571 w_k1_AXP_s104
                                 + 0.570364 w_k1_DIS_s104
                                 + 0.751402 w_k1_FDX_s104
                                 + 0.630038 w_k1_IBM_s104
                                 + 0.772393 w_k1_INTC_s104
                                 + 0.6764 w_k1_KO_s104 + 0.650889 w_k1_MAT_s104
                                 + 0.797983 w_k1_MSFT_s104
                                 + 0.788403 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.49473 w_k1_ADM_s104
                                 + 1.23469 w_k1_AXP_s104
                                 + 1.33693 w_k1_DIS_s104
                                 + 1.22842 w_k1_FDX_s104
                                 + 0.826947 w_k1_IBM_s104
                                 + 1.16917 w_k1_INTC_s104
                                 + 1.24014 w_k1_KO_s104
                                 + 0.936701 w_k1_MAT_s104
                                 + 1.15325 w_k1_MSFT_s104
                                 + 1.08358 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.17411 w_k1_ADM_s104
                                 + 0.91676 w_k1_AXP_s104
                                 + 0.901018 w_k1_DIS_s104
                                 + 0.985359 w_k1_FDX_s104
                                 + 0.749943 w_k1_IBM_s104
                                 + 1.23875 w_k1_INTC_s104
                                 + 0.946635 w_k1_KO_s104
                                 + 0.657096 w_k1_MAT_s104
                                 + 1.00646 w_k1_MSFT_s104
                                 + 0.946448 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.790108 w_k1_ADM_s104
                                 + 0.630395 w_k1_AXP_s104
                                 + 0.587078 w_k1_DIS_s104
                                 + 0.735476 w_k1_FDX_s104
                                 + 0.776852 w_k1_IBM_s104
                                 + 0.859477 w_k1_INTC_s104
                                 + 0.750203 w_k1_KO_s104
                                 + 0.771341 w_k1_MAT_s104
                                 + 0.792604 w_k1_MSFT_s104
                                 + 0.74356 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.07232 w_k1_ADM_s104
                                 + 1.17794 w_k1_AXP_s104
                                 + 1.07277 w_k1_DIS_s104 + 1.1692 w_k1_FDX_s104
                                 + 1.1203 w_k1_IBM_s104
                                 + 1.27497 w_k1_INTC_s104
                                 + 1.16388 w_k1_KO_s104 + 1.54866 w_k1_MAT_s104
                                 + 1.12715 w_k1_MSFT_s104
                                 + 1.05549 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.03067 w_k1_ADM_s104
                                 + 0.644462 w_k1_AXP_s104
                                 + 0.699884 w_k1_DIS_s104
                                 + 0.8439 w_k1_FDX_s104
                                 + 0.534078 w_k1_IBM_s104
                                 + 0.878826 w_k1_INTC_s104
                                 + 0.69808 w_k1_KO_s104
                                 + 0.505252 w_k1_MAT_s104
                                 + 0.945585 w_k1_MSFT_s104
                                 + 0.798602 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.59065 w_k1_ADM_s104
                                 + 1.48354 w_k1_AXP_s104
                                 + 1.58555 w_k1_DIS_s104
                                 + 1.28724 w_k1_FDX_s104 + 1.6612 w_k1_IBM_s104
                                 + 1.17919 w_k1_INTC_s104
                                 + 1.44168 w_k1_KO_s104 + 2.10739 w_k1_MAT_s104
                                 + 1.30034 w_k1_MSFT_s104 + 1.2512 w_k1_PG_s104
                                 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.18946 w_k1_ADM_s105
                                 + 0.956234 w_k1_AXP_s105
                                 + 0.962661 w_k1_DIS_s105
                                 + 1.00648 w_k1_FDX_s105
                                 + 0.982963 w_k1_IBM_s105
                                 + 0.926544 w_k1_INTC_s105
                                 + 0.936439 w_k1_KO_s105
                                 + 0.824307 w_k1_MAT_s105
                                 + 0.958491 w_k1_MSFT_s105
                                 + 1.03503 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.952073 w_k1_ADM_s105
                                 + 0.837534 w_k1_AXP_s105
                                 + 0.795466 w_k1_DIS_s105
                                 + 0.862249 w_k1_FDX_s105
                                 + 0.757026 w_k1_IBM_s105
                                 + 0.864867 w_k1_INTC_s105
                                 + 0.748717 w_k1_KO_s105
                                 + 0.783055 w_k1_MAT_s105
                                 + 0.896288 w_k1_MSFT_s105
                                 + 0.93909 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.32345 w_k1_ADM_s105
                                 + 1.6779 w_k1_AXP_s105 + 1.34543 w_k1_DIS_s105
                                 + 1.18412 w_k1_FDX_s105
                                 + 1.49581 w_k1_IBM_s105
                                 + 1.16278 w_k1_INTC_s105
                                 + 1.25984 w_k1_KO_s105 + 1.67291 w_k1_MAT_s105
                                 + 1.05403 w_k1_MSFT_s105 + 1.2086 w_k1_PG_s105
                                 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.02661 w_k1_ADM_s105
                                 + 1.18403 w_k1_AXP_s105
                                 + 0.984721 w_k1_DIS_s105
                                 + 0.935425 w_k1_FDX_s105
                                 + 1.29944 w_k1_IBM_s105
                                 + 0.943675 w_k1_INTC_s105
                                 + 0.940936 w_k1_KO_s105
                                 + 1.36288 w_k1_MAT_s105
                                 + 0.915638 w_k1_MSFT_s105 + 1.138 w_k1_PG_s105
                                 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.913226 w_k1_ADM_s105
                                 + 1.47133 w_k1_AXP_s105
                                 + 0.95596 w_k1_DIS_s105
                                 + 1.11895 w_k1_FDX_s105
                                 + 0.876817 w_k1_IBM_s105
                                 + 1.10728 w_k1_INTC_s105
                                 + 0.834668 w_k1_KO_s105
                                 + 0.906248 w_k1_MAT_s105
                                 + 0.963994 w_k1_MSFT_s105
                                 + 1.10499 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.28383 w_k1_ADM_s105
                                 + 1.36005 w_k1_AXP_s105
                                 + 1.50926 w_k1_DIS_s105
                                 + 1.29098 w_k1_FDX_s105
                                 + 1.07865 w_k1_IBM_s105
                                 + 0.987155 w_k1_INTC_s105
                                 + 1.29214 w_k1_KO_s105 + 1.44092 w_k1_MAT_s105
                                 + 1.1988 w_k1_MSFT_s105 + 1.45577 w_k1_PG_s105
                                 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.92583 w_k1_ADM_s105
                                 + 0.853279 w_k1_AXP_s105
                                 + 0.829097 w_k1_DIS_s105
                                 + 0.796847 w_k1_FDX_s105
                                 + 0.907892 w_k1_IBM_s105
                                 + 0.9745 w_k1_INTC_s105
                                 + 0.943646 w_k1_KO_s105
                                 + 0.911217 w_k1_MAT_s105
                                 + 0.839941 w_k1_MSFT_s105
                                 + 0.887855 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.894777 w_k1_ADM_s105
                                 + 0.874921 w_k1_AXP_s105
                                 + 0.697433 w_k1_DIS_s105
                                 + 0.844923 w_k1_FDX_s105
                                 + 1.02021 w_k1_IBM_s105
                                 + 0.85743 w_k1_INTC_s105
                                 + 0.914004 w_k1_KO_s105
                                 + 0.957625 w_k1_MAT_s105
                                 + 0.811373 w_k1_MSFT_s105
                                 + 1.02603 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.944396 w_k1_ADM_s106
                                 + 0.976282 w_k1_AXP_s106
                                 + 0.951093 w_k1_DIS_s106
                                 + 0.898001 w_k1_FDX_s106
                                 + 1.06002 w_k1_IBM_s106
                                 + 0.918897 w_k1_INTC_s106
                                 + 1.12778 w_k1_KO_s106 + 1.22937 w_k1_MAT_s106
                                 + 0.865996 w_k1_MSFT_s106
                                 + 1.00567 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.00201 w_k1_ADM_s106
                                 + 1.02874 w_k1_AXP_s106 + 1.341 w_k1_DIS_s106
                                 + 0.996121 w_k1_FDX_s106
                                 + 1.12267 w_k1_IBM_s106
                                 + 1.13605 w_k1_INTC_s106
                                 + 1.15721 w_k1_KO_s106
                                 + 0.693147 w_k1_MAT_s106
                                 + 1.015 w_k1_MSFT_s106 + 1.06524 w_k1_PG_s106
                                 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.18501 w_k1_ADM_s106
                                 + 1.01284 w_k1_AXP_s106
                                 + 0.832956 w_k1_DIS_s106
                                 + 0.981655 w_k1_FDX_s106
                                 + 0.711598 w_k1_IBM_s106
                                 + 1.10866 w_k1_INTC_s106
                                 + 0.796524 w_k1_KO_s106
                                 + 0.506567 w_k1_MAT_s106
                                 + 0.794209 w_k1_MSFT_s106
                                 + 0.907429 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.25874 w_k1_ADM_s106
                                 + 1.26406 w_k1_AXP_s106
                                 + 1.14658 w_k1_DIS_s106
                                 + 1.16883 w_k1_FDX_s106
                                 + 0.921906 w_k1_IBM_s106
                                 + 1.23138 w_k1_INTC_s106
                                 + 0.985754 w_k1_KO_s106
                                 + 0.893949 w_k1_MAT_s106
                                 + 1.11974 w_k1_MSFT_s106
                                 + 1.07775 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.11736 w_k1_ADM_s106
                                 + 1.34464 w_k1_AXP_s106
                                 + 1.53078 w_k1_DIS_s106 + 1.1913 w_k1_FDX_s106
                                 + 1.45952 w_k1_IBM_s106 + 1.162 w_k1_INTC_s106
                                 + 1.43784 w_k1_KO_s106 + 1.39834 w_k1_MAT_s106
                                 + 1.14347 w_k1_MSFT_s106
                                 + 1.27466 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.4837 w_k1_ADM_s106
                                 + 1.62638 w_k1_AXP_s106
                                 + 1.74873 w_k1_DIS_s106
                                 + 1.49895 w_k1_FDX_s106
                                 + 0.970256 w_k1_IBM_s106
                                 + 1.24057 w_k1_INTC_s106
                                 + 1.34097 w_k1_KO_s106 + 1.24762 w_k1_MAT_s106
                                 + 1.16742 w_k1_MSFT_s106
                                 + 1.29764 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.677587 w_k1_ADM_s106
                                 + 0.821602 w_k1_AXP_s106
                                 + 0.698885 w_k1_DIS_s106
                                 + 0.695556 w_k1_FDX_s106
                                 + 1.22311 w_k1_IBM_s106
                                 + 0.756638 w_k1_INTC_s106
                                 + 0.945783 w_k1_KO_s106
                                 + 1.12869 w_k1_MAT_s106
                                 + 0.674232 w_k1_MSFT_s106
                                 + 0.865647 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.998962 w_k1_ADM_s106
                                 + 0.834602 w_k1_AXP_s106
                                 + 0.688416 w_k1_DIS_s106
                                 + 0.77659 w_k1_FDX_s106
                                 + 0.821547 w_k1_IBM_s106
                                 + 0.878708 w_k1_INTC_s106
                                 + 0.831046 w_k1_KO_s106
                                 + 0.734005 w_k1_MAT_s106
                                 + 0.786284 w_k1_MSFT_s106
                                 + 0.882996 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.0377 w_k1_ADM_s107
                                 + 1.07327 w_k1_AXP_s107
                                 + 0.947657 w_k1_DIS_s107
                                 + 1.01375 w_k1_FDX_s107
                                 + 1.03744 w_k1_IBM_s107
                                 + 1.07308 w_k1_INTC_s107
                                 + 1.03552 w_k1_KO_s107 + 1.05788 w_k1_MAT_s107
                                 + 0.923237 w_k1_MSFT_s107
                                 + 1.05693 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.860949 w_k1_ADM_s107
                                 + 0.757914 w_k1_AXP_s107
                                 + 1.00154 w_k1_DIS_s107
                                 + 0.98185 w_k1_FDX_s107
                                 + 0.699802 w_k1_IBM_s107
                                 + 0.974626 w_k1_INTC_s107
                                 + 0.835587 w_k1_KO_s107
                                 + 0.520098 w_k1_MAT_s107
                                 + 0.970878 w_k1_MSFT_s107
                                 + 0.88738 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.902546 w_k1_ADM_s107
                                 + 0.669442 w_k1_AXP_s107
                                 + 0.772332 w_k1_DIS_s107
                                 + 0.755461 w_k1_FDX_s107
                                 + 0.932573 w_k1_IBM_s107
                                 + 0.727991 w_k1_INTC_s107
                                 + 0.85866 w_k1_KO_s107
                                 + 0.878766 w_k1_MAT_s107
                                 + 0.697916 w_k1_MSFT_s107
                                 + 0.878199 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.41315 w_k1_ADM_s107
                                 + 1.55756 w_k1_AXP_s107
                                 + 1.55053 w_k1_DIS_s107
                                 + 1.36784 w_k1_FDX_s107 + 1.0251 w_k1_IBM_s107
                                 + 1.37064 w_k1_INTC_s107
                                 + 1.09151 w_k1_KO_s107 + 1.00594 w_k1_MAT_s107
                                 + 1.18414 w_k1_MSFT_s107
                                 + 1.09254 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.869368 w_k1_ADM_s107
                                 + 0.855762 w_k1_AXP_s107
                                 + 0.774259 w_k1_DIS_s107
                                 + 0.846853 w_k1_FDX_s107
                                 + 0.742549 w_k1_IBM_s107
                                 + 0.983455 w_k1_INTC_s107
                                 + 0.729286 w_k1_KO_s107
                                 + 0.620995 w_k1_MAT_s107
                                 + 0.702945 w_k1_MSFT_s107
                                 + 0.841138 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.712403 w_k1_ADM_s107
                                 + 0.535519 w_k1_AXP_s107
                                 + 0.571782 w_k1_DIS_s107
                                 + 0.695659 w_k1_FDX_s107
                                 + 0.553303 w_k1_IBM_s107
                                 + 0.825194 w_k1_INTC_s107
                                 + 0.619841 w_k1_KO_s107
                                 + 0.445983 w_k1_MAT_s107
                                 + 0.640723 w_k1_MSFT_s107
                                 + 0.772052 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.23422 w_k1_ADM_s107
                                 + 1.23042 w_k1_AXP_s107
                                 + 1.40962 w_k1_DIS_s107
                                 + 1.21467 w_k1_FDX_s107
                                 + 1.33818 w_k1_IBM_s107
                                 + 1.10616 w_k1_INTC_s107
                                 + 1.28247 w_k1_KO_s107 + 1.4595 w_k1_MAT_s107
                                 + 1.06366 w_k1_MSFT_s107
                                 + 1.29525 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.918824 w_k1_ADM_s107
                                 + 0.957952 w_k1_AXP_s107
                                 + 0.975214 w_k1_DIS_s107
                                 + 0.854722 w_k1_FDX_s107
                                 + 1.28591 w_k1_IBM_s107
                                 + 0.820722 w_k1_INTC_s107
                                 + 1.13975 w_k1_KO_s107 + 1.27674 w_k1_MAT_s107
                                 + 0.892379 w_k1_MSFT_s107
                                 + 1.00082 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.931125 w_k1_ADM_s110
                                 + 1.0441 w_k1_AXP_s110
                                 + 0.831513 w_k1_DIS_s110
                                 + 0.947812 w_k1_FDX_s110
                                 + 0.731252 w_k1_IBM_s110
                                 + 0.905417 w_k1_INTC_s110
                                 + 0.732015 w_k1_KO_s110
                                 + 0.746121 w_k1_MAT_s110
                                 + 0.859757 w_k1_MSFT_s110
                                 + 0.9976 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.824203 w_k1_ADM_s110
                                 + 0.799924 w_k1_AXP_s110
                                 + 0.717852 w_k1_DIS_s110
                                 + 0.815749 w_k1_FDX_s110
                                 + 1.11911 w_k1_IBM_s110
                                 + 0.814794 w_k1_INTC_s110
                                 + 0.793732 w_k1_KO_s110
                                 + 0.796069 w_k1_MAT_s110
                                 + 0.711633 w_k1_MSFT_s110
                                 + 0.952369 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.769794 w_k1_ADM_s110
                                 + 0.647625 w_k1_AXP_s110
                                 + 0.689085 w_k1_DIS_s110
                                 + 0.749037 w_k1_FDX_s110
                                 + 0.80616 w_k1_IBM_s110
                                 + 0.743651 w_k1_INTC_s110
                                 + 0.708706 w_k1_KO_s110
                                 + 0.586529 w_k1_MAT_s110
                                 + 0.703782 w_k1_MSFT_s110
                                 + 0.871862 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.754999 w_k1_ADM_s110
                                 + 0.936488 w_k1_AXP_s110
                                 + 1.07493 w_k1_DIS_s110
                                 + 0.994421 w_k1_FDX_s110
                                 + 0.989568 w_k1_IBM_s110
                                 + 0.999785 w_k1_INTC_s110
                                 + 0.937113 w_k1_KO_s110
                                 + 1.05238 w_k1_MAT_s110
                                 + 0.996389 w_k1_MSFT_s110
                                 + 1.07027 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.978336 w_k1_ADM_s110
                                 + 0.892271 w_k1_AXP_s110
                                 + 0.834868 w_k1_DIS_s110
                                 + 0.925628 w_k1_FDX_s110
                                 + 0.573161 w_k1_IBM_s110
                                 + 0.922253 w_k1_INTC_s110
                                 + 0.745062 w_k1_KO_s110
                                 + 0.554723 w_k1_MAT_s110
                                 + 0.948696 w_k1_MSFT_s110
                                 + 0.862816 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.883314 w_k1_ADM_s110
                                 + 1.03087 w_k1_AXP_s110
                                 + 1.34268 w_k1_DIS_s110
                                 + 1.01388 w_k1_FDX_s110
                                 + 1.43709 w_k1_IBM_s110
                                 + 0.983162 w_k1_INTC_s110
                                 + 1.19482 w_k1_KO_s110 + 1.61662 w_k1_MAT_s110
                                 + 0.972788 w_k1_MSFT_s110
                                 + 1.07861 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.20444 w_k1_ADM_s110
                                 + 1.11612 w_k1_AXP_s110
                                 + 0.943778 w_k1_DIS_s110
                                 + 1.09975 w_k1_FDX_s110
                                 + 0.934636 w_k1_IBM_s110
                                 + 1.0661 w_k1_INTC_s110
                                 + 0.851614 w_k1_KO_s110
                                 + 1.23163 w_k1_MAT_s110 + 0.89 w_k1_MSFT_s110
                                 + 1.02867 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.25639 w_k1_ADM_s110
                                 + 1.22792 w_k1_AXP_s110
                                 + 1.38039 w_k1_DIS_s110
                                 + 1.24116 w_k1_FDX_s110
                                 + 1.05871 w_k1_IBM_s110
                                 + 1.14906 w_k1_INTC_s110 + 1.0749 w_k1_KO_s110
                                 + 0.943759 w_k1_MAT_s110
                                 + 1.13222 w_k1_MSFT_s110
                                 + 1.25378 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.24294 w_k1_ADM_s111
                                 + 1.40968 w_k1_AXP_s111
                                 + 1.17543 w_k1_DIS_s111
                                 + 1.21409 w_k1_FDX_s111
                                 + 1.27697 w_k1_IBM_s111
                                 + 1.07681 w_k1_INTC_s111
                                 + 1.20921 w_k1_KO_s111 + 1.29585 w_k1_MAT_s111
                                 + 1.25064 w_k1_MSFT_s111
                                 + 1.11528 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.976046 w_k1_ADM_s111
                                 + 0.955452 w_k1_AXP_s111
                                 + 1.19591 w_k1_DIS_s111
                                 + 1.08027 w_k1_FDX_s111
                                 + 1.39904 w_k1_IBM_s111
                                 + 1.13314 w_k1_INTC_s111 + 1.1262 w_k1_KO_s111
                                 + 1.00164 w_k1_MAT_s111
                                 + 1.09539 w_k1_MSFT_s111 + 1.097 w_k1_PG_s111
                                 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.857959 w_k1_ADM_s111
                                 + 1.13182 w_k1_AXP_s111
                                 + 0.88925 w_k1_DIS_s111
                                 + 0.902595 w_k1_FDX_s111
                                 + 1.7032 w_k1_IBM_s111
                                 + 1.08359 w_k1_INTC_s111
                                 + 1.09774 w_k1_KO_s111 + 1.6702 w_k1_MAT_s111
                                 + 0.871322 w_k1_MSFT_s111
                                 + 0.975972 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.807091 w_k1_ADM_s111
                                 + 0.799667 w_k1_AXP_s111
                                 + 0.60378 w_k1_DIS_s111
                                 + 0.753394 w_k1_FDX_s111
                                 + 1.09187 w_k1_IBM_s111
                                 + 0.806963 w_k1_INTC_s111
                                 + 0.824071 w_k1_KO_s111
                                 + 1.05883 w_k1_MAT_s111
                                 + 0.771901 w_k1_MSFT_s111
                                 + 0.833292 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.95834 w_k1_ADM_s111
                                 + 0.621034 w_k1_AXP_s111
                                 + 0.739414 w_k1_DIS_s111
                                 + 0.77179 w_k1_FDX_s111
                                 + 0.825863 w_k1_IBM_s111
                                 + 0.900738 w_k1_INTC_s111
                                 + 0.806093 w_k1_KO_s111
                                 + 0.624322 w_k1_MAT_s111
                                 + 0.828252 w_k1_MSFT_s111
                                 + 0.831399 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.938542 w_k1_ADM_s111
                                 + 0.728798 w_k1_AXP_s111
                                 + 0.84449 w_k1_DIS_s111
                                 + 1.00875 w_k1_FDX_s111
                                 + 0.561816 w_k1_IBM_s111
                                 + 0.855706 w_k1_INTC_s111
                                 + 0.727832 w_k1_KO_s111
                                 + 0.491255 w_k1_MAT_s111
                                 + 0.959426 w_k1_MSFT_s111
                                 + 0.86815 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.09553 w_k1_ADM_s111
                                 + 1.15737 w_k1_AXP_s111
                                 + 1.18005 w_k1_DIS_s111
                                 + 1.28196 w_k1_FDX_s111
                                 + 0.793594 w_k1_IBM_s111
                                 + 1.30878 w_k1_INTC_s111
                                 + 1.05036 w_k1_KO_s111
                                 + 0.727159 w_k1_MAT_s111
                                 + 1.15253 w_k1_MSFT_s111
                                 + 1.00574 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.85784 w_k1_ADM_s111
                                 + 0.962091 w_k1_AXP_s111
                                 + 0.83926 w_k1_DIS_s111
                                 + 0.934711 w_k1_FDX_s111
                                 + 0.898474 w_k1_IBM_s111
                                 + 1.01674 w_k1_INTC_s111
                                 + 0.971888 w_k1_KO_s111
                                 + 0.878577 w_k1_MAT_s111
                                 + 0.97887 w_k1_MSFT_s111
                                 + 0.892852 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.068 w_k1_ADM_s112
                                 + 0.956813 w_k1_AXP_s112
                                 + 0.962229 w_k1_DIS_s112
                                 + 0.937735 w_k1_FDX_s112
                                 + 1.41361 w_k1_IBM_s112
                                 + 1.07449 w_k1_INTC_s112
                                 + 1.27407 w_k1_KO_s112 + 1.35903 w_k1_MAT_s112
                                 + 0.957223 w_k1_MSFT_s112
                                 + 1.11365 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.27458 w_k1_ADM_s112
                                 + 1.086 w_k1_AXP_s112 + 1.63571 w_k1_DIS_s112
                                 + 1.35125 w_k1_FDX_s112 + 1.0776 w_k1_IBM_s112
                                 + 1.02386 w_k1_INTC_s112 + 1.2331 w_k1_KO_s112
                                 + 0.992934 w_k1_MAT_s112
                                 + 1.16549 w_k1_MSFT_s112
                                 + 1.04203 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.918286 w_k1_ADM_s112
                                 + 0.931874 w_k1_AXP_s112
                                 + 0.699401 w_k1_DIS_s112
                                 + 0.817937 w_k1_FDX_s112
                                 + 1.09301 w_k1_IBM_s112
                                 + 0.845985 w_k1_INTC_s112
                                 + 0.821641 w_k1_KO_s112
                                 + 1.22795 w_k1_MAT_s112
                                 + 0.75908 w_k1_MSFT_s112
                                 + 0.912823 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.17811 w_k1_ADM_s112
                                 + 1.23453 w_k1_AXP_s112
                                 + 0.968697 w_k1_DIS_s112
                                 + 1.08065 w_k1_FDX_s112
                                 + 1.01225 w_k1_IBM_s112
                                 + 1.05807 w_k1_INTC_s112
                                 + 0.91071 w_k1_KO_s112 + 1.05787 w_k1_MAT_s112
                                 + 0.842433 w_k1_MSFT_s112
                                 + 0.970687 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.42504 w_k1_ADM_s112
                                 + 1.1977 w_k1_AXP_s112 + 1.16197 w_k1_DIS_s112
                                 + 1.21575 w_k1_FDX_s112
                                 + 1.16793 w_k1_IBM_s112
                                 + 1.05086 w_k1_INTC_s112
                                 + 0.956113 w_k1_KO_s112
                                 + 1.09395 w_k1_MAT_s112
                                 + 1.04464 w_k1_MSFT_s112 + 1.1862 w_k1_PG_s112
                                 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.01864 w_k1_ADM_s112
                                 + 0.907614 w_k1_AXP_s112
                                 + 0.998582 w_k1_DIS_s112
                                 + 0.994753 w_k1_FDX_s112
                                 + 0.754197 w_k1_IBM_s112
                                 + 0.875747 w_k1_INTC_s112
                                 + 0.92344 w_k1_KO_s112
                                 + 0.689343 w_k1_MAT_s112
                                 + 0.833712 w_k1_MSFT_s112
                                 + 0.90933 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.779987 w_k1_ADM_s112
                                 + 0.58831 w_k1_AXP_s112
                                 + 0.581827 w_k1_DIS_s112
                                 + 0.75909 w_k1_FDX_s112
                                 + 0.764849 w_k1_IBM_s112
                                 + 0.819135 w_k1_INTC_s112
                                 + 0.661708 w_k1_KO_s112
                                 + 0.763987 w_k1_MAT_s112
                                 + 0.682832 w_k1_MSFT_s112
                                 + 0.764005 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.40033 w_k1_ADM_s112
                                 + 1.49729 w_k1_AXP_s112
                                 + 1.21333 w_k1_DIS_s112
                                 + 1.21672 w_k1_FDX_s112
                                 + 1.55154 w_k1_IBM_s112
                                 + 1.01712 w_k1_INTC_s112
                                 + 1.40923 w_k1_KO_s112 + 1.9508 w_k1_MAT_s112
                                 + 1.03479 w_k1_MSFT_s112
                                 + 1.22196 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.742426 w_k1_ADM_s113
                                 + 0.661434 w_k1_AXP_s113
                                 + 0.682746 w_k1_DIS_s113
                                 + 0.751165 w_k1_FDX_s113
                                 + 0.743599 w_k1_IBM_s113
                                 + 0.837211 w_k1_INTC_s113
                                 + 0.774252 w_k1_KO_s113
                                 + 0.537409 w_k1_MAT_s113
                                 + 0.830059 w_k1_MSFT_s113
                                 + 0.728365 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.0237 w_k1_ADM_s113
                                 + 1.23028 w_k1_AXP_s113
                                 + 1.29235 w_k1_DIS_s113
                                 + 1.15557 w_k1_FDX_s113
                                 + 1.15792 w_k1_IBM_s113
                                 + 1.1311 w_k1_INTC_s113 + 1.0555 w_k1_KO_s113
                                 + 1.09521 w_k1_MAT_s113
                                 + 1.30923 w_k1_MSFT_s113
                                 + 0.975743 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.989621 w_k1_ADM_s113
                                 + 0.988416 w_k1_AXP_s113
                                 + 0.996613 w_k1_DIS_s113
                                 + 1.10189 w_k1_FDX_s113
                                 + 0.66718 w_k1_IBM_s113
                                 + 0.955876 w_k1_INTC_s113
                                 + 0.753654 w_k1_KO_s113
                                 + 0.668998 w_k1_MAT_s113
                                 + 0.973386 w_k1_MSFT_s113
                                 + 0.915493 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.02962 w_k1_ADM_s113
                                 + 1.22756 w_k1_AXP_s113
                                 + 1.51074 w_k1_DIS_s113 + 1.1745 w_k1_FDX_s113
                                 + 1.49179 w_k1_IBM_s113
                                 + 1.53339 w_k1_INTC_s113
                                 + 1.23315 w_k1_KO_s113 + 1.23812 w_k1_MAT_s113
                                 + 1.22669 w_k1_MSFT_s113
                                 + 1.15804 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.870885 w_k1_ADM_s113
                                 + 0.876067 w_k1_AXP_s113
                                 + 0.896584 w_k1_DIS_s113
                                 + 0.923359 w_k1_FDX_s113
                                 + 0.889017 w_k1_IBM_s113
                                 + 1.1322 w_k1_INTC_s113
                                 + 0.862174 w_k1_KO_s113
                                 + 0.785188 w_k1_MAT_s113
                                 + 0.998773 w_k1_MSFT_s113
                                 + 0.91301 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.874515 w_k1_ADM_s113
                                 + 0.896994 w_k1_AXP_s113
                                 + 0.93252 w_k1_DIS_s113
                                 + 0.894583 w_k1_FDX_s113
                                 + 1.44012 w_k1_IBM_s113
                                 + 0.952541 w_k1_INTC_s113
                                 + 0.921219 w_k1_KO_s113
                                 + 0.971961 w_k1_MAT_s113
                                 + 1.01311 w_k1_MSFT_s113
                                 + 0.954349 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.25072 w_k1_ADM_s113
                                 + 1.55491 w_k1_AXP_s113
                                 + 1.83691 w_k1_DIS_s113
                                 + 1.32577 w_k1_FDX_s113
                                 + 1.98252 w_k1_IBM_s113
                                 + 1.24719 w_k1_INTC_s113
                                 + 1.47328 w_k1_KO_s113 + 1.62301 w_k1_MAT_s113
                                 + 1.26034 w_k1_MSFT_s113
                                 + 1.34105 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.18292 w_k1_ADM_s113
                                 + 1.37107 w_k1_AXP_s113
                                 + 1.39058 w_k1_DIS_s113
                                 + 1.26915 w_k1_FDX_s113
                                 + 0.857465 w_k1_IBM_s113
                                 + 1.32292 w_k1_INTC_s113
                                 + 1.01637 w_k1_KO_s113
                                 + 0.859023 w_k1_MAT_s113
                                 + 1.32522 w_k1_MSFT_s113
                                 + 1.10082 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.777445 w_k1_ADM_s114
                                 + 0.845862 w_k1_AXP_s114
                                 + 0.856505 w_k1_DIS_s114
                                 + 0.890374 w_k1_FDX_s114
                                 + 1.22179 w_k1_IBM_s114
                                 + 0.870267 w_k1_INTC_s114
                                 + 0.854496 w_k1_KO_s114
                                 + 1.18638 w_k1_MAT_s114
                                 + 0.906063 w_k1_MSFT_s114
                                 + 0.913968 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.18715 w_k1_ADM_s114
                                 + 1.41139 w_k1_AXP_s114
                                 + 1.49165 w_k1_DIS_s114
                                 + 1.57998 w_k1_FDX_s114
                                 + 0.893195 w_k1_IBM_s114
                                 + 1.60774 w_k1_INTC_s114
                                 + 1.16176 w_k1_KO_s114
                                 + 0.966407 w_k1_MAT_s114
                                 + 1.43148 w_k1_MSFT_s114
                                 + 1.06352 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.24545 w_k1_ADM_s114
                                 + 1.10784 w_k1_AXP_s114
                                 + 1.31502 w_k1_DIS_s114
                                 + 1.13661 w_k1_FDX_s114
                                 + 1.89476 w_k1_IBM_s114
                                 + 1.12869 w_k1_INTC_s114
                                 + 1.51326 w_k1_KO_s114 + 1.76117 w_k1_MAT_s114
                                 + 1.33024 w_k1_MSFT_s114
                                 + 1.23287 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.20996 w_k1_ADM_s114
                                 + 1.10167 w_k1_AXP_s114
                                 + 1.04523 w_k1_DIS_s114
                                 + 1.10354 w_k1_FDX_s114
                                 + 1.09839 w_k1_IBM_s114
                                 + 1.09229 w_k1_INTC_s114
                                 + 0.993887 w_k1_KO_s114
                                 + 1.09686 w_k1_MAT_s114
                                 + 1.09072 w_k1_MSFT_s114
                                 + 1.04762 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.793084 w_k1_ADM_s114
                                 + 0.75751 w_k1_AXP_s114
                                 + 0.707507 w_k1_DIS_s114
                                 + 0.912195 w_k1_FDX_s114
                                 + 0.714124 w_k1_IBM_s114
                                 + 1.01234 w_k1_INTC_s114
                                 + 0.715758 w_k1_KO_s114
                                 + 0.519786 w_k1_MAT_s114
                                 + 0.819069 w_k1_MSFT_s114
                                 + 0.758083 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.931333 w_k1_ADM_s114
                                 + 0.777565 w_k1_AXP_s114
                                 + 1.03441 w_k1_DIS_s114
                                 + 0.948568 w_k1_FDX_s114
                                 + 1.01494 w_k1_IBM_s114
                                 + 0.949372 w_k1_INTC_s114
                                 + 0.932007 w_k1_KO_s114
                                 + 0.916634 w_k1_MAT_s114
                                 + 0.989489 w_k1_MSFT_s114
                                 + 0.983621 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.983208 w_k1_ADM_s114
                                 + 1.25412 w_k1_AXP_s114
                                 + 1.81997 w_k1_DIS_s114
                                 + 1.46208 w_k1_FDX_s114
                                 + 1.45325 w_k1_IBM_s114
                                 + 1.39241 w_k1_INTC_s114
                                 + 1.37917 w_k1_KO_s114 + 1.32995 w_k1_MAT_s114
                                 + 1.28799 w_k1_MSFT_s114
                                 + 1.25165 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.922088 w_k1_ADM_s114
                                 + 1.0338 w_k1_AXP_s114 + 1.06346 w_k1_DIS_s114
                                 + 1.17807 w_k1_FDX_s114
                                 + 0.830669 w_k1_IBM_s114
                                 + 1.2095 w_k1_INTC_s114
                                 + 0.957155 w_k1_KO_s114
                                 + 0.796758 w_k1_MAT_s114
                                 + 1.0915 w_k1_MSFT_s114 + 1.0726 w_k1_PG_s114
                                 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.912065 w_k1_ADM_s115
                                 + 1.04656 w_k1_AXP_s115
                                 + 0.93287 w_k1_DIS_s115
                                 + 1.03091 w_k1_FDX_s115
                                 + 0.640345 w_k1_IBM_s115
                                 + 0.875512 w_k1_INTC_s115
                                 + 0.750182 w_k1_KO_s115
                                 + 1.35525 w_k1_MAT_s115
                                 + 0.882151 w_k1_MSFT_s115
                                 + 0.941608 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.735229 w_k1_ADM_s115
                                 + 1.09645 w_k1_AXP_s115
                                 + 0.980314 w_k1_DIS_s115
                                 + 0.911973 w_k1_FDX_s115
                                 + 1.26112 w_k1_IBM_s115
                                 + 1.06434 w_k1_INTC_s115
                                 + 1.09846 w_k1_KO_s115 + 1.56744 w_k1_MAT_s115
                                 + 0.943148 w_k1_MSFT_s115
                                 + 1.01753 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.731418 w_k1_ADM_s115
                                 + 0.795099 w_k1_AXP_s115
                                 + 0.722819 w_k1_DIS_s115
                                 + 0.726218 w_k1_FDX_s115
                                 + 1.11395 w_k1_IBM_s115 + 1.03 w_k1_INTC_s115
                                 + 0.940393 w_k1_KO_s115
                                 + 1.05827 w_k1_MAT_s115
                                 + 0.846533 w_k1_MSFT_s115
                                 + 0.759877 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.11409 w_k1_ADM_s115
                                 + 1.3819 w_k1_AXP_s115 + 1.12781 w_k1_DIS_s115
                                 + 1.04465 w_k1_FDX_s115
                                 + 1.53437 w_k1_IBM_s115
                                 + 0.911815 w_k1_INTC_s115
                                 + 1.17419 w_k1_KO_s115 + 2.78462 w_k1_MAT_s115
                                 + 0.976676 w_k1_MSFT_s115
                                 + 1.09998 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.694035 w_k1_ADM_s115
                                 + 0.942458 w_k1_AXP_s115
                                 + 0.819439 w_k1_DIS_s115
                                 + 0.875433 w_k1_FDX_s115
                                 + 0.867774 w_k1_IBM_s115
                                 + 0.887738 w_k1_INTC_s115
                                 + 0.883946 w_k1_KO_s115
                                 + 1.34277 w_k1_MAT_s115
                                 + 0.865699 w_k1_MSFT_s115
                                 + 0.89106 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.93707 w_k1_ADM_s115
                                 + 1.16812 w_k1_AXP_s115
                                 + 1.05312 w_k1_DIS_s115 + 1.0989 w_k1_FDX_s115
                                 + 1.03222 w_k1_IBM_s115
                                 + 1.1191 w_k1_INTC_s115 + 1.02938 w_k1_KO_s115
                                 + 1.32532 w_k1_MAT_s115
                                 + 0.902211 w_k1_MSFT_s115
                                 + 0.953482 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.704748 w_k1_ADM_s115
                                 + 0.772416 w_k1_AXP_s115
                                 + 0.546224 w_k1_DIS_s115
                                 + 0.709801 w_k1_FDX_s115
                                 + 0.754903 w_k1_IBM_s115
                                 + 0.811387 w_k1_INTC_s115
                                 + 0.690327 w_k1_KO_s115
                                 + 0.835734 w_k1_MAT_s115
                                 + 0.716852 w_k1_MSFT_s115
                                 + 0.769243 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.13821 w_k1_ADM_s115
                                 + 1.26693 w_k1_AXP_s115
                                 + 1.46555 w_k1_DIS_s115
                                 + 1.34622 w_k1_FDX_s115
                                 + 1.11093 w_k1_IBM_s115
                                 + 1.21464 w_k1_INTC_s115
                                 + 1.34563 w_k1_KO_s115 + 1.58113 w_k1_MAT_s115
                                 + 1.31769 w_k1_MSFT_s115
                                 + 1.17665 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.961406 w_k1_ADM_s116
                                 + 0.953263 w_k1_AXP_s116
                                 + 0.864782 w_k1_DIS_s116
                                 + 1.05775 w_k1_FDX_s116
                                 + 0.748528 w_k1_IBM_s116
                                 + 1.07705 w_k1_INTC_s116
                                 + 0.831337 w_k1_KO_s116
                                 + 0.625697 w_k1_MAT_s116
                                 + 0.938102 w_k1_MSFT_s116
                                 + 1.01173 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.0051 w_k1_ADM_s116
                                 + 0.745853 w_k1_AXP_s116
                                 + 1.03831 w_k1_DIS_s116
                                 + 0.83786 w_k1_FDX_s116 + 1.2334 w_k1_IBM_s116
                                 + 0.975776 w_k1_INTC_s116
                                 + 1.21397 w_k1_KO_s116
                                 + 0.996005 w_k1_MAT_s116
                                 + 0.904304 w_k1_MSFT_s116
                                 + 0.972318 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.940411 w_k1_ADM_s116
                                 + 0.624963 w_k1_AXP_s116
                                 + 0.609134 w_k1_DIS_s116
                                 + 0.745968 w_k1_FDX_s116
                                 + 0.672079 w_k1_IBM_s116
                                 + 0.930602 w_k1_INTC_s116
                                 + 0.649056 w_k1_KO_s116
                                 + 0.376497 w_k1_MAT_s116
                                 + 0.684093 w_k1_MSFT_s116
                                 + 0.80854 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.34222 w_k1_ADM_s116
                                 + 1.32037 w_k1_AXP_s116 + 1.2141 w_k1_DIS_s116
                                 + 1.14886 w_k1_FDX_s116
                                 + 1.34064 w_k1_IBM_s116
                                 + 1.21153 w_k1_INTC_s116
                                 + 1.19124 w_k1_KO_s116 + 1.24095 w_k1_MAT_s116
                                 + 1.00778 w_k1_MSFT_s116 + 1.2808 w_k1_PG_s116
                                 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.06786 w_k1_ADM_s116
                                 + 1.04857 w_k1_AXP_s116
                                 + 0.752097 w_k1_DIS_s116
                                 + 0.868711 w_k1_FDX_s116
                                 + 0.863729 w_k1_IBM_s116
                                 + 0.975714 w_k1_INTC_s116
                                 + 0.757367 w_k1_KO_s116
                                 + 0.909494 w_k1_MAT_s116
                                 + 0.843216 w_k1_MSFT_s116
                                 + 0.969415 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.02848 w_k1_ADM_s116
                                 + 1.14873 w_k1_AXP_s116
                                 + 1.05922 w_k1_DIS_s116
                                 + 1.10871 w_k1_FDX_s116
                                 + 0.986608 w_k1_IBM_s116
                                 + 1.14043 w_k1_INTC_s116
                                 + 0.995536 w_k1_KO_s116
                                 + 0.918815 w_k1_MAT_s116
                                 + 1.04034 w_k1_MSFT_s116
                                 + 1.13743 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.910488 w_k1_ADM_s116
                                 + 0.663309 w_k1_AXP_s116
                                 + 0.757323 w_k1_DIS_s116
                                 + 0.788989 w_k1_FDX_s116
                                 + 0.819374 w_k1_IBM_s116
                                 + 0.855422 w_k1_INTC_s116
                                 + 0.783385 w_k1_KO_s116
                                 + 0.707755 w_k1_MAT_s116
                                 + 0.88864 w_k1_MSFT_s116
                                 + 0.949909 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.20238 w_k1_ADM_s116
                                 + 0.96069 w_k1_AXP_s116
                                 + 1.41396 w_k1_DIS_s116
                                 + 1.13018 w_k1_FDX_s116
                                 + 1.22483 w_k1_IBM_s116
                                 + 1.25752 w_k1_INTC_s116
                                 + 1.31332 w_k1_KO_s116
                                 + 0.802045 w_k1_MAT_s116
                                 + 1.15503 w_k1_MSFT_s116
                                 + 1.14665 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.18944 w_k1_ADM_s117
                                 + 1.48713 w_k1_AXP_s117
                                 + 1.53077 w_k1_DIS_s117
                                 + 1.23682 w_k1_FDX_s117
                                 + 1.19232 w_k1_IBM_s117
                                 + 1.4398 w_k1_INTC_s117 + 1.30676 w_k1_KO_s117
                                 + 1.1993 w_k1_MAT_s117
                                 + 1.31035 w_k1_MSFT_s117
                                 + 1.31656 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.830788 w_k1_ADM_s117
                                 + 0.6941 w_k1_AXP_s117
                                 + 0.796402 w_k1_DIS_s117
                                 + 0.788784 w_k1_FDX_s117
                                 + 0.892672 w_k1_IBM_s117
                                 + 1.00085 w_k1_INTC_s117
                                 + 0.83844 w_k1_KO_s117 + 0.68586 w_k1_MAT_s117
                                 + 0.823504 w_k1_MSFT_s117
                                 + 1.0492 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.821693 w_k1_ADM_s117
                                 + 0.967946 w_k1_AXP_s117
                                 + 0.816813 w_k1_DIS_s117
                                 + 0.946141 w_k1_FDX_s117
                                 + 0.679718 w_k1_IBM_s117
                                 + 1.06754 w_k1_INTC_s117
                                 + 0.816464 w_k1_KO_s117
                                 + 0.919563 w_k1_MAT_s117
                                 + 0.903497 w_k1_MSFT_s117
                                 + 1.01824 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.672891 w_k1_ADM_s117
                                 + 0.799422 w_k1_AXP_s117
                                 + 0.820954 w_k1_DIS_s117
                                 + 0.772356 w_k1_FDX_s117
                                 + 1.26698 w_k1_IBM_s117
                                 + 0.89663 w_k1_INTC_s117
                                 + 0.922096 w_k1_KO_s117
                                 + 0.975559 w_k1_MAT_s117
                                 + 0.716787 w_k1_MSFT_s117
                                 + 0.984209 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.31877 w_k1_ADM_s117
                                 + 1.58322 w_k1_AXP_s117
                                 + 1.30218 w_k1_DIS_s117
                                 + 1.40052 w_k1_FDX_s117
                                 + 0.78353 w_k1_IBM_s117
                                 + 1.40818 w_k1_INTC_s117
                                 + 1.02511 w_k1_KO_s117
                                 + 0.871816 w_k1_MAT_s117
                                 + 1.07559 w_k1_MSFT_s117
                                 + 1.10942 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.00168 w_k1_ADM_s117
                                 + 1.04391 w_k1_AXP_s117
                                 + 1.13162 w_k1_DIS_s117
                                 + 0.977605 w_k1_FDX_s117
                                 + 1.67402 w_k1_IBM_s117
                                 + 1.16131 w_k1_INTC_s117
                                 + 1.43604 w_k1_KO_s117 + 1.41344 w_k1_MAT_s117
                                 + 1.0391 w_k1_MSFT_s117 + 1.12393 w_k1_PG_s117
                                 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.817141 w_k1_ADM_s117
                                 + 1.15226 w_k1_AXP_s117
                                 + 0.995666 w_k1_DIS_s117
                                 + 1.03865 w_k1_FDX_s117
                                 + 1.16119 w_k1_IBM_s117
                                 + 1.09606 w_k1_INTC_s117
                                 + 1.01123 w_k1_KO_s117 + 1.0165 w_k1_MAT_s117
                                 + 0.941892 w_k1_MSFT_s117
                                 + 1.08136 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.855932 w_k1_ADM_s117
                                 + 1.01096 w_k1_AXP_s117
                                 + 1.18971 w_k1_DIS_s117
                                 + 1.14579 w_k1_FDX_s117
                                 + 0.900943 w_k1_IBM_s117
                                 + 1.17949 w_k1_INTC_s117
                                 + 1.25867 w_k1_KO_s117
                                 + 0.913801 w_k1_MAT_s117
                                 + 1.11524 w_k1_MSFT_s117
                                 + 1.10056 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0324581981404934 w_k1_ADM_s000
                                 + 0.0182536454976186 w_k1_ADM_s001
                                 + 0.0193312498147182 w_k1_ADM_s002
                                 + 0.0226087200059713 w_k1_ADM_s003
                                 + 0.0472378959581505 w_k1_ADM_s004
                                 + 0.0181018306340216 w_k1_ADM_s005
                                 + 0.061821518897749 w_k1_ADM_s006
                                 + 0.0610924204533611 w_k1_ADM_s007
                                 + 0.0167853775991401 w_k1_ADM_s010
                                 + 0.0118869258788676 w_k1_ADM_s011
                                 + 0.00491379138179432 w_k1_ADM_s012
                                 + 0.0067980133656967 w_k1_ADM_s013
                                 + 0.014976701223483 w_k1_ADM_s014
                                 + 0.00810870614937895 w_k1_ADM_s015
                                 + 0.0151846226077015 w_k1_ADM_s016
                                 + 0.011017360337125 w_k1_ADM_s017
                                 + 0.0310573913147312 w_k1_ADM_s100
                                 + 0.0411048279169668 w_k1_ADM_s101
                                 + 0.017947031228404 w_k1_ADM_s102
                                 + 0.0208116925805516 w_k1_ADM_s103
                                 + 0.0498263143091148 w_k1_ADM_s104
                                 + 0.0452709894461943 w_k1_ADM_s105
                                 + 0.0562090366017224 w_k1_ADM_s106
                                 + 0.0717067179941141 w_k1_ADM_s107
                                 + 0.0390351476519507 w_k1_ADM_s110
                                 + 0.0487714765505524 w_k1_ADM_s111
                                 + 0.0221944667350955 w_k1_ADM_s112
                                 + 0.0711709313005606 w_k1_ADM_s113
                                 + 0.0506933108321928 w_k1_ADM_s114
                                 + 0.0290956345260659 w_k1_ADM_s115
                                 + 0.0248233104139141 w_k1_ADM_s116
                                 + 0.0346758768289008 w_k1_ADM_s117
                                 + 0.031337312898988 w_k1_AXP_s000
                                 + 0.0188118974056456 w_k1_AXP_s001
                                 + 0.0220554153856255 w_k1_AXP_s002
                                 + 0.0228894656524436 w_k1_AXP_s003
                                 + 0.0411205467327463 w_k1_AXP_s004
                                 + 0.0175634354394423 w_k1_AXP_s005
                                 + 0.0552205790854641 w_k1_AXP_s006
                                 + 0.0639401085256024 w_k1_AXP_s007
                                 + 0.0156525095373844 w_k1_AXP_s010
                                 + 0.010770899692338 w_k1_AXP_s011
                                 + 0.00516268074314494 w_k1_AXP_s012
                                 + 0.00751300017578471 w_k1_AXP_s013
                                 + 0.0171337350693452 w_k1_AXP_s014
                                 + 0.00644392609046637 w_k1_AXP_s015
                                 + 0.0119060551826236 w_k1_AXP_s016
                                 + 0.0121532195561329 w_k1_AXP_s017
                                 + 0.0293832037649221 w_k1_AXP_s100
                                 + 0.0392306151394143 w_k1_AXP_s101
                                 + 0.0178368595350971 w_k1_AXP_s102
                                 + 0.017515043946332 w_k1_AXP_s103
                                 + 0.0448108648653109 w_k1_AXP_s104
                                 + 0.0475594530811844 w_k1_AXP_s105
                                 + 0.0584595609854208 w_k1_AXP_s106
                                 + 0.0700515281403811 w_k1_AXP_s107
                                 + 0.0395200629934651 w_k1_AXP_s110
                                 + 0.0491827251303292 w_k1_AXP_s111
                                 + 0.0206128214744094 w_k1_AXP_s112
                                 + 0.0768766993830616 w_k1_AXP_s113
                                 + 0.0517172454616209 w_k1_AXP_s114
                                 + 0.0358687257098452 w_k1_AXP_s115
                                 + 0.0230762091154484 w_k1_AXP_s116
                                 + 0.0405180160509287 w_k1_AXP_s117
                                 + 0.0332174794398848 w_k1_DIS_s000
                                 + 0.0215906920625656 w_k1_DIS_s001
                                 + 0.0204650637096065 w_k1_DIS_s002
                                 + 0.0238320362170695 w_k1_DIS_s003
                                 + 0.0475940161721433 w_k1_DIS_s004
                                 + 0.0175334923745758 w_k1_DIS_s005
                                 + 0.0590750575986895 w_k1_DIS_s006
                                 + 0.0646864762763239 w_k1_DIS_s007
                                 + 0.0165443094656115 w_k1_DIS_s010
                                 + 0.0103300627991692 w_k1_DIS_s011
                                 + 0.0050977477015358 w_k1_DIS_s012
                                 + 0.00749720638437952 w_k1_DIS_s013
                                 + 0.0161099703758827 w_k1_DIS_s014
                                 + 0.00712728301857645 w_k1_DIS_s015
                                 + 0.0121799786680601 w_k1_DIS_s016
                                 + 0.0120912095894254 w_k1_DIS_s017
                                 + 0.035587206696578 w_k1_DIS_s100
                                 + 0.0410949338088849 w_k1_DIS_s101
                                 + 0.0157858299054276 w_k1_DIS_s102
                                 + 0.0172273712743791 w_k1_DIS_s103
                                 + 0.0448013729408396 w_k1_DIS_s104
                                 + 0.0436932826204216 w_k1_DIS_s105
                                 + 0.0577340023192756 w_k1_DIS_s106
                                 + 0.072456268578824 w_k1_DIS_s107
                                 + 0.041483378034978 w_k1_DIS_s110
                                 + 0.04824034802367 w_k1_DIS_s111
                                 + 0.0201059532679057 w_k1_DIS_s112
                                 + 0.0814787222898659 w_k1_DIS_s113
                                 + 0.055550102757804 w_k1_DIS_s114
                                 + 0.033068248316987 w_k1_DIS_s115
                                 + 0.0227254147477637 w_k1_DIS_s116
                                 + 0.0407166610808496 w_k1_DIS_s117
                                 + 0.0317817046990013 w_k1_FDX_s000
                                 + 0.0220752842342657 w_k1_FDX_s001
                                 + 0.0207891161557596 w_k1_FDX_s002
                                 + 0.0232360838467566 w_k1_FDX_s003
                                 + 0.044674910534958 w_k1_FDX_s004
                                 + 0.0165959601020575 w_k1_FDX_s005
                                 + 0.0564888195593325 w_k1_FDX_s006
                                 + 0.0643697812869732 w_k1_FDX_s007
                                 + 0.0160513033891069 w_k1_FDX_s010
                                 + 0.0104273330063832 w_k1_FDX_s011
                                 + 0.00480610412327059 w_k1_FDX_s012
                                 + 0.00693530546258438 w_k1_FDX_s013
                                 + 0.0153486370968916 w_k1_FDX_s014
                                 + 0.00734940364386664 w_k1_FDX_s015
                                 + 0.0128211049016741 w_k1_FDX_s016
                                 + 0.011460902371908 w_k1_FDX_s017
                                 + 0.0326338161284247 w_k1_FDX_s100
                                 + 0.0377645834064777 w_k1_FDX_s101
                                 + 0.0162247896342304 w_k1_FDX_s102
                                 + 0.0182339218705856 w_k1_FDX_s103
                                 + 0.0445834691066353 w_k1_FDX_s104
                                 + 0.0425868808765444 w_k1_FDX_s105
                                 + 0.0533822021578297 w_k1_FDX_s106
                                 + 0.0696488636448929 w_k1_FDX_s107
                                 + 0.0399623827412521 w_k1_FDX_s110
                                 + 0.0504820075846854 w_k1_FDX_s111
                                 + 0.0205819571162496 w_k1_FDX_s112
                                 + 0.0768543034582088 w_k1_FDX_s113
                                 + 0.0579345966401209 w_k1_FDX_s114
                                 + 0.0330735091603311 w_k1_FDX_s115
                                 + 0.0231850781712331 w_k1_FDX_s116
                                 + 0.0389039910649464 w_k1_FDX_s117
                                 + 0.0321783603928812 w_k1_IBM_s000
                                 + 0.0213093382836027 w_k1_IBM_s001
                                 + 0.0225034033631854 w_k1_IBM_s002
                                 + 0.0237441598459022 w_k1_IBM_s003
                                 + 0.0460343083745595 w_k1_IBM_s004
                                 + 0.0173705068387056 w_k1_IBM_s005
                                 + 0.0563210686414531 w_k1_IBM_s006
                                 + 0.0638258280017992 w_k1_IBM_s007
                                 + 0.0169655417390257 w_k1_IBM_s010
                                 + 0.0102358466954466 w_k1_IBM_s011
                                 + 0.00479524355791128 w_k1_IBM_s012
                                 + 0.00715084490751796 w_k1_IBM_s013
                                 + 0.0152698321900664 w_k1_IBM_s014
                                 + 0.00728141099486875 w_k1_IBM_s015
                                 + 0.0125998991637724 w_k1_IBM_s016
                                 + 0.0122336874124042 w_k1_IBM_s017
                                 + 0.0319924776454827 w_k1_IBM_s100
                                 + 0.0404192621350918 w_k1_IBM_s101
                                 + 0.016550231597917 w_k1_IBM_s102
                                 + 0.0177625971456995 w_k1_IBM_s103
                                 + 0.0454791928693545 w_k1_IBM_s104
                                 + 0.043256462276239 w_k1_IBM_s105
                                 + 0.0559893627231551 w_k1_IBM_s106
                                 + 0.0720797435581592 w_k1_IBM_s107
                                 + 0.0398821762229258 w_k1_IBM_s110
                                 + 0.0519349740042653 w_k1_IBM_s111
                                 + 0.0215865659048766 w_k1_IBM_s112
                                 + 0.0769924496318326 w_k1_IBM_s113
                                 + 0.0559801634343804 w_k1_IBM_s114
                                 + 0.0344262912265884 w_k1_IBM_s115
                                 + 0.0228749474052669 w_k1_IBM_s116
                                 + 0.0393818743776771 w_k1_IBM_s117
                                 + 0.0360746142772511 w_k1_INTC_s000
                                 + 0.0204691193551989 w_k1_INTC_s001
                                 + 0.0237164641759922 w_k1_INTC_s002
                                 + 0.0228684878184063 w_k1_INTC_s003
                                 + 0.0453585606443323 w_k1_INTC_s004
                                 + 0.0163034212322689 w_k1_INTC_s005
                                 + 0.0532484078727261 w_k1_INTC_s006
                                 + 0.0681882035976946 w_k1_INTC_s007
                                 + 0.0174441552583618 w_k1_INTC_s010
                                 + 0.00986412499601613 w_k1_INTC_s011
                                 + 0.00532143413230029 w_k1_INTC_s012
                                 + 0.00643600440317646 w_k1_INTC_s013
                                 + 0.0125431896623876 w_k1_INTC_s014
                                 + 0.00744963410608511 w_k1_INTC_s015
                                 + 0.0109946164867211 w_k1_INTC_s016
                                 + 0.0107695498192817 w_k1_INTC_s017
                                 + 0.0277650778884959 w_k1_INTC_s100
                                 + 0.0421060604865276 w_k1_INTC_s101
                                 + 0.0160279964555123 w_k1_INTC_s102
                                 + 0.0175099584984292 w_k1_INTC_s103
                                 + 0.0468263863544097 w_k1_INTC_s104
                                 + 0.0399084245076944 w_k1_INTC_s105
                                 + 0.054674458379487 w_k1_INTC_s106
                                 + 0.0704972289043247 w_k1_INTC_s107
                                 + 0.038634733664202 w_k1_INTC_s110
                                 + 0.0515686181964746 w_k1_INTC_s111
                                 + 0.0194752406062425 w_k1_INTC_s112
                                 + 0.0818332805911095 w_k1_INTC_s113
                                 + 0.0592624370341733 w_k1_INTC_s114
                                 + 0.0331433697395601 w_k1_INTC_s115
                                 + 0.0249639437052019 w_k1_INTC_s116
                                 + 0.0436362256490606 w_k1_INTC_s117
                                 + 0.0334053126256229 w_k1_KO_s000
                                 + 0.0225112472150579 w_k1_KO_s001
                                 + 0.0224973365750705 w_k1_KO_s002
                                 + 0.0231678146160492 w_k1_KO_s003
                                 + 0.0475930881944547 w_k1_KO_s004
                                 + 0.0172725904150622 w_k1_KO_s005
                                 + 0.0592966955339615 w_k1_KO_s006
                                 + 0.0574845434701907 w_k1_KO_s007
                                 + 0.0176224990152637 w_k1_KO_s010
                                 + 0.0107002275386214 w_k1_KO_s011
                                 + 0.00501564171459349 w_k1_KO_s012
                                 + 0.00749847933484313 w_k1_KO_s013
                                 + 0.0153335651176963 w_k1_KO_s014
                                 + 0.00781519965879351 w_k1_KO_s015
                                 + 0.0128349604609004 w_k1_KO_s016
                                 + 0.0103848647754196 w_k1_KO_s017
                                 + 0.0328376814555125 w_k1_KO_s100
                                 + 0.0436339327509207 w_k1_KO_s101
                                 + 0.016272298105003 w_k1_KO_s102
                                 + 0.0191976018363669 w_k1_KO_s103
                                 + 0.0457669621336776 w_k1_KO_s104
                                 + 0.0415623840625423 w_k1_KO_s105
                                 + 0.0569178884281252 w_k1_KO_s106
                                 + 0.0705996341659244 w_k1_KO_s107
                                 + 0.036575609860842 w_k1_KO_s110
                                 + 0.0488686995034953 w_k1_KO_s111
                                 + 0.0203836202018998 w_k1_KO_s112
                                 + 0.0708849798559532 w_k1_KO_s113
                                 + 0.0518970168510107 w_k1_KO_s114
                                 + 0.0334987399933282 w_k1_KO_s115
                                 + 0.0224225648267077 w_k1_KO_s116
                                 + 0.0408081016972699 w_k1_KO_s117
                                 + 0.0348905599812677 w_k1_MAT_s000
                                 + 0.0204511691911789 w_k1_MAT_s001
                                 + 0.0253559388362332 w_k1_MAT_s002
                                 + 0.0265885920093669 w_k1_MAT_s003
                                 + 0.0391247802644221 w_k1_MAT_s004
                                 + 0.0182990687450659 w_k1_MAT_s005
                                 + 0.0562895070250007 w_k1_MAT_s006
                                 + 0.0565687030276413 w_k1_MAT_s007
                                 + 0.0149323255473574 w_k1_MAT_s010
                                 + 0.00995296303664754 w_k1_MAT_s011
                                 + 0.00511209903838108 w_k1_MAT_s012
                                 + 0.00832751506208765 w_k1_MAT_s013
                                 + 0.0180000338881295 w_k1_MAT_s014
                                 + 0.00589829828372906 w_k1_MAT_s015
                                 + 0.0121066248220604 w_k1_MAT_s016
                                 + 0.0120457571170451 w_k1_MAT_s017
                                 + 0.0327312358773165 w_k1_MAT_s100
                                 + 0.0422683834006044 w_k1_MAT_s101
                                 + 0.0181854394437704 w_k1_MAT_s102
                                 + 0.0166495264028794 w_k1_MAT_s103
                                 + 0.0533806036883054 w_k1_MAT_s104
                                 + 0.0468213478385668 w_k1_MAT_s105
                                 + 0.053653940779728 w_k1_MAT_s106
                                 + 0.0707727034384823 w_k1_MAT_s107
                                 + 0.0400679353829216 w_k1_MAT_s110
                                 + 0.0464890092563152 w_k1_MAT_s111
                                 + 0.0219475800609663 w_k1_MAT_s112
                                 + 0.0664706609155083 w_k1_MAT_s113
                                 + 0.0511503384378309 w_k1_MAT_s114
                                 + 0.0491676774868814 w_k1_MAT_s115
                                 + 0.0197121314428446 w_k1_MAT_s116
                                 + 0.0373140323492233 w_k1_MAT_s117
                                 + 0.0358995525316116 w_k1_MSFT_s000
                                 + 0.0213440036085605 w_k1_MSFT_s001
                                 + 0.0212165729068756 w_k1_MSFT_s002
                                 + 0.0236834658419651 w_k1_MSFT_s003
                                 + 0.0426098531032793 w_k1_MSFT_s004
                                 + 0.0169859363332279 w_k1_MSFT_s005
                                 + 0.0568097103853032 w_k1_MSFT_s006
                                 + 0.0572930526121125 w_k1_MSFT_s007
                                 + 0.0159752747827582 w_k1_MSFT_s010
                                 + 0.0102239571922732 w_k1_MSFT_s011
                                 + 0.00553074826699432 w_k1_MSFT_s012
                                 + 0.00677695245801901 w_k1_MSFT_s013
                                 + 0.0145864892826407 w_k1_MSFT_s014
                                 + 0.00790915020165036 w_k1_MSFT_s015
                                 + 0.0120843006251894 w_k1_MSFT_s016
                                 + 0.0106986953294567 w_k1_MSFT_s017
                                 + 0.0305096898758528 w_k1_MSFT_s100
                                 + 0.0425375777713687 w_k1_MSFT_s101
                                 + 0.0159893238998176 w_k1_MSFT_s102
                                 + 0.0185357963561026 w_k1_MSFT_s103
                                 + 0.0466490522161491 w_k1_MSFT_s104
                                 + 0.040591148095617 w_k1_MSFT_s105
                                 + 0.0493928917488917 w_k1_MSFT_s106
                                 + 0.0633904077515162 w_k1_MSFT_s107
                                 + 0.0367312782685484 w_k1_MSFT_s110
                                 + 0.0498469567094802 w_k1_MSFT_s111
                                 + 0.01805907978944 w_k1_MSFT_s112
                                 + 0.0809889648738918 w_k1_MSFT_s113
                                 + 0.0555787630095108 w_k1_MSFT_s114
                                 + 0.0311783108697896 w_k1_MSFT_s115
                                 + 0.0220943129388742 w_k1_MSFT_s116
                                 + 0.0381785140866734 w_k1_MSFT_s117
                                 + 0.0332794633399495 w_k1_PG_s000
                                 + 0.0197541488473477 w_k1_PG_s001
                                 + 0.0216734238589067 w_k1_PG_s002
                                 + 0.0239818392744874 w_k1_PG_s003
                                 + 0.0428269387748237 w_k1_PG_s004
                                 + 0.0173049708454806 w_k1_PG_s005
                                 + 0.0544439156161933 w_k1_PG_s006
                                 + 0.0584513902062929 w_k1_PG_s007
                                 + 0.0161040527064586 w_k1_PG_s010
                                 + 0.010901531560211 w_k1_PG_s011
                                 + 0.00519124278642821 w_k1_PG_s012
                                 + 0.00734232629660933 w_k1_PG_s013
                                 + 0.0158906623779055 w_k1_PG_s014
                                 + 0.00739228356271257 w_k1_PG_s015
                                 + 0.0129424998658992 w_k1_PG_s016
                                 + 0.0108396377512787 w_k1_PG_s017
                                 + 0.0307178770852092 w_k1_PG_s100
                                 + 0.0402303403107583 w_k1_PG_s101
                                 + 0.0171440608625522 w_k1_PG_s102
                                 + 0.0200748308740403 w_k1_PG_s103
                                 + 0.0425478253799661 w_k1_PG_s104
                                 + 0.0468194938889445 w_k1_PG_s105
                                 + 0.0551159435302265 w_k1_PG_s106
                                 + 0.0721887877437892 w_k1_PG_s107
                                 + 0.0414117693469871 w_k1_PG_s110
                                 + 0.0473103967269603 w_k1_PG_s111
                                 + 0.0201687913752946 w_k1_PG_s112
                                 + 0.071323540460416 w_k1_PG_s113
                                 + 0.0524334056608005 w_k1_PG_s114
                                 + 0.0318737197201914 w_k1_PG_s115
                                 + 0.024661739472712 w_k1_PG_s116
                                 + 0.0418334695285726 w_k1_PG_s117 >= 1.06
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
