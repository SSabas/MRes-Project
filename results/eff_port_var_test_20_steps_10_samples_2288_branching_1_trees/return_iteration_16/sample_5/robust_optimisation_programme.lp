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
 w_k1_ADM_s0_return_constraint:  1.22952 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.937566 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.08718 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.917686 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.24378 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.853135 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.849435 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.767372 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.95319 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.08251 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.13405 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.03053 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.07737 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.13796 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.12052 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.0792 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.764017 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.1926 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.869519 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.939272 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.03828 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.02569 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.972809 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.17849 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.925904 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.814906 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.996346 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.14397 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.735777 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.77419 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.21747 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.38781 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.10659 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.829484 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.02445 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.009 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.910817 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.757367 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.36508 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.95536 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.29405 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.856549 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.22871 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.759922 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.706206 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.846752 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.775013 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.780287 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.53856 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.00558 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.966708 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.19385 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.08005 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.12425 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.654709 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.41123 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.624918 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.891789 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.3371 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.15056 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.913738 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.39747 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.876435 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.762951 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.06548 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.23324 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.745135 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.664925 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.57215 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.41604 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.94174 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.741928 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.20659 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.00397 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.09457 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.550695 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.38213 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.879666 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.48359 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.899109 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.39179 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.854917 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.659361 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.754503 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.626677 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.953824 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.54426 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.16074 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.827295 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.25687 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.33335 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.14488 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.593071 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.72073 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.563093 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.883436 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.05111 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.11163 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.909065 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.36512 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.830838 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.803011 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.832383 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.26406 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.721149 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.62769 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.35638 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.49963 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.834452 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.649861 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.13498 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.886506 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.921307 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.422601 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.25275 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.934703 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.21348 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.894157 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.18458 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.863596 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.774158 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.801481 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.831107 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.03991 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.2447 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.00597 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.914048 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.25673 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.2067 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.21717 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.717095 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.34355 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.772186 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.908562 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.991834 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.07707 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.893764 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.13038 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.04292 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.786099 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.977605 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.24076 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.853325 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.774592 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.23497 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.43207 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.01417 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.756952 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.0099 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.920529 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.887953 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.615175 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.19299 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.970628 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.24826 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.938394 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.13426 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.936548 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.781323 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.42488 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.57293 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.566106 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.65864 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.31446 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.975263 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.879128 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.07404 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.786716 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.898089 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.47238 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.561474 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.998058 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.84527 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.23524 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.19277 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.69974 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.586776 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.22651 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.872502 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.993407 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.651004 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.812883 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.77555 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.930277 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.581878 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.731651 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.12776 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.951158 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.45985 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.597083 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.20835 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.00213 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.0947 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.953285 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.07562 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.891956 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.874572 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.833468 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.876683 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.939259 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.46403 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.97939 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.867478 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.24829 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.946871 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.02931 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.784759 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.08275 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.762999 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.799512 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.07375 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.02981 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.901226 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.01486 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.949695 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.783374 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.01388 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.35121 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.812886 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.759369 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.132 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.08875 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.944158 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.780723 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.897293 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.863589 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.91819 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.636217 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.23575 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.942197 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.28791 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.876828 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.23008 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.944042 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.749102 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.858832 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.689853 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.732643 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.55079 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.17172 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.950242 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.05261 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.20391 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.03524 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.829275 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.54568 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.647316 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.938492 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.37899 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.08846 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.08738 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.44953 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.700954 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.952008 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.888183 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.05162 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.643932 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.713308 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.44893 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.1288 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.706668 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.715351 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.12589 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.820279 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.06112 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.556338 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.28491 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.883671 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.47003 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.882811 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.24193 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.06589 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.680561 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.2871 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.517389 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.408612 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.42388 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.21027 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.913691 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.996479 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.09062 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.861767 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.791984 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.71335 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.51104 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.06156 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.86121 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.26704 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.15831 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.87588 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.718035 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.27568 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.989864 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.0334 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.69474 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.760831 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.77177 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.891556 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.55419 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.584994 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.1189 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.876553 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.12875 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.416164 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.19218 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.943481 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.26117 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.848208 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.1966 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.898634 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.75891 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.721752 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.820454 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.06966 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.37171 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.915384 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.821622 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.23528 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.3358 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.20859 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.763614 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.4269 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.851177 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.05895 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.15227 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.08575 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.870523 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.08703 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.95221 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.788054 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.907393 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.16678 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.850845 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.793101 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.09616 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.28887 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.03245 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.805766 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.972628 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.897922 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.913368 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.642337 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.14908 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.944835 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.20107 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.94881 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.10646 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.926279 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.804411 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.839046 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.830785 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.92951 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.22891 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.01994 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.892895 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.00806 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.02456 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.08567 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.846086 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.42504 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.794729 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.991785 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.13843 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.06436 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.01835 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.25188 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.943344 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.96032 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.00708 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.12333 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.893899 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.858106 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.25302 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.1798 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.963248 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.838611 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.0584 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.998933 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.982332 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.715295 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.152000347040235 z_k1_s0000
                                 - 0.0640001345221747 z_k1_s0001
                                 - 0.290000850616715 z_k1_s0002
                                 - 0.126000795179375 z_k1_s0003
                                 - 0.0670001167215905 z_k1_s0004
                                 - 0.232000239692255 z_k1_s0005
                                 - 0.0990002941829571 z_k1_s0006
                                 - 0.0720002339876559 z_k1_s0007
                                 - 0.0859999428891777 z_k1_s0010
                                 - 0.0700000417144766 z_k1_s0011
                                 - 0.113000167959117 z_k1_s0012
                                 - 0.0690000333785529 z_k1_s0013
                                 - 0.0789998845377119 z_k1_s0014
                                 - 0.0899999762328725 z_k1_s0015
                                 - 0.0600000357552657 z_k1_s0016
                                 - 0.207000177535685 z_k1_s0017
                                 - 0.0639999789033684 z_k1_s0020
                                 - 0.0830002827418413 z_k1_s0021
                                 - 0.0490000217150756 z_k1_s0022
                                 - 0.0790001980916382 z_k1_s0023
                                 - 0.203999732160754 z_k1_s0024
                                 - 0.0579999829280524 z_k1_s0025
                                 - 0.0600000252531539 z_k1_s0026
                                 - 0.0579999829280524 z_k1_s0027
                                 - 0.173000298724083 z_k1_s0030
                                 - 0.118000043383802 z_k1_s0031
                                 - 0.162999726843197 z_k1_s0032
                                 - 0.31699997939525 z_k1_s0033
                                 - 0.0890001189319874 z_k1_s0034
                                 - 0.0730001287240392 z_k1_s0035
                                 - 0.152000369376243 z_k1_s0036
                                 - 0.0710001299480457 z_k1_s0037
                                 - 0.0620001749380226 z_k1_s0040
                                 - 0.131000094134379 z_k1_s0041
                                 - 0.0990006015229363 z_k1_s0042
                                 - 0.0900002134744318 z_k1_s0043
                                 - 0.242000505386607 z_k1_s0044
                                 - 0.112999897039046 z_k1_s0045
                                 - 0.0800002755330765 z_k1_s0046
                                 - 0.148000548336303 z_k1_s0047
                                 - 0.236999950205656 z_k1_s0050
                                 - 0.152000656673145 z_k1_s0051
                                 - 0.13100009621093 z_k1_s0052
                                 - 0.174000818823318 z_k1_s0053
                                 - 0.128999617832859 z_k1_s0054
                                 - 0.0750000891475366 z_k1_s0055
                                 - 0.280000672818044 z_k1_s0056
                                 - 0.0970001237974951 z_k1_s0057
                                 - 0.107999716675503 z_k1_s0060
                                 - 0.158999965299054 z_k1_s0061
                                 - 0.172999908960744 z_k1_s0062
                                 - 0.0589998079598625 z_k1_s0063
                                 - 0.113999995101157 z_k1_s0064
                                 - 0.0949997841181804 z_k1_s0065
                                 - 0.245999766482022 z_k1_s0066
                                 - 0.104999683362402 z_k1_s0067
                                 - 0.30699919677064 z_k1_s0070
                                 - 0.188999499839261 z_k1_s0071
                                 - 0.170999829025645 z_k1_s0072
                                 - 0.265998845218785 z_k1_s0073
                                 - 0.445999031350583 z_k1_s0074
                                 - 0.142999877360601 z_k1_s0075
                                 - 0.109999785269844 z_k1_s0076
                                 - 0.106999840134242 z_k1_s0077
                                 - 0.106999977281856 z_k1_s0100
                                 - 0.149000027355111 z_k1_s0101
                                 - 0.140999566960183 z_k1_s0102
                                 - 0.140999566960183 z_k1_s0103
                                 - 0.16800020029302 z_k1_s0104
                                 - 0.0590000553124287 z_k1_s0105
                                 - 0.212000102464993 z_k1_s0106
                                 - 0.0750000293022379 z_k1_s0107
                                 - 0.678000448254283 z_k1_s0110
                                 - 0.597001938649476 z_k1_s0111
                                 - 0.345000176009011 z_k1_s0112
                                 - 0.45800081125168 z_k1_s0113
                                 - 0.346000308122728 z_k1_s0114
                                 - 0.865999220819673 z_k1_s0115
                                 - 0.544002083426151 z_k1_s0116
                                 - 0.370000115650206 z_k1_s0117
                                 - 0.122999886324925 z_k1_s0120
                                 - 0.10099993807167 z_k1_s0121
                                 - 0.228999837726867 z_k1_s0122
                                 - 0.150000399908158 z_k1_s0123
                                 - 0.14299966364616 z_k1_s0124
                                 - 0.175000466559517 z_k1_s0125
                                 - 0.257000022776331 z_k1_s0126
                                 - 0.202000152143192 z_k1_s0127
                                 - 0.117999988900023 z_k1_s0130
                                 - 0.1950002516569 z_k1_s0131
                                 - 0.101000054099192 z_k1_s0132
                                 - 0.0920000969458129 z_k1_s0133
                                 - 0.0870000578161388 z_k1_s0134
                                 - 0.279999925661057 z_k1_s0135
                                 - 0.147000668972271 z_k1_s0136
                                 - 0.396000097149271 z_k1_s0137
                                 - 0.34199911227856 z_k1_s0140
                                 - 0.13399991815347 z_k1_s0141
                                 - 0.15299984812451 z_k1_s0142
                                 - 0.330000549601495 z_k1_s0143
                                 - 0.15299984812451 z_k1_s0144
                                 - 0.20199972638664 z_k1_s0145
                                 - 0.42499926745711 z_k1_s0146
                                 - 0.124999970946126 z_k1_s0147
                                 - 0.381999204820027 z_k1_s0150
                                 - 1.02299556300201 z_k1_s0151
                                 - 1.02199601790633 z_k1_s0152
                                 - 0.343999400516766 z_k1_s0153
                                 - 0.77200098519365 z_k1_s0154
                                 - 0.804996331331341 z_k1_s0155
                                 - 0.604999269363172 z_k1_s0156
                                 - 0.22599973562821 z_k1_s0157
                                 - 0.243000258756655 z_k1_s0160
                                 - 0.370000333510118 z_k1_s0161
                                 - 0.191000626755337 z_k1_s0162
                                 - 0.129000010353501 z_k1_s0163
                                 - 0.0930000264525288 z_k1_s0164
                                 - 0.338999453759057 z_k1_s0165
                                 - 0.159000024154318 z_k1_s0166
                                 - 0.109000001152956 z_k1_s0167
                                 - 0.269000924403475 z_k1_s0170
                                 - 0.379000869675133 z_k1_s0171
                                 - 0.158000487000778 z_k1_s0172
                                 - 0.197000613561114 z_k1_s0173
                                 - 0.376000396784658 z_k1_s0174
                                 - 0.309000539592209 z_k1_s0175
                                 - 0.235000247990411 z_k1_s0176
                                 - 0.084000195398981 z_k1_s0177
                                 - 0.741000227483987 z_k1_s1000
                                 - 0.306999812639727 z_k1_s1001
                                 - 0.841000389503384 z_k1_s1002
                                 - 0.329999813864198 z_k1_s1003
                                 - 0.27899998351424 z_k1_s1004
                                 - 0.188999837696783 z_k1_s1005
                                 - 0.472999649112039 z_k1_s1006
                                 - 0.443998268646761 z_k1_s1007
                                 - 0.734995936276042 z_k1_s1010
                                 - 0.773999225276634 z_k1_s1011
                                 - 0.982995122495488 z_k1_s1012
                                 - 0.968998834327506 z_k1_s1013
                                 - 0.413998867801173 z_k1_s1014
                                 - 0.528998522123809 z_k1_s1015
                                 - 0.715999370338344 z_k1_s1016
                                 - 0.491998711421522 z_k1_s1017
                                 - 0.359000826519661 z_k1_s1020
                                 - 0.958999838154971 z_k1_s1021
                                 - 0.24000064751611 z_k1_s1022
                                 - 0.899003185907627 z_k1_s1023
                                 - 0.206000539072129 z_k1_s1024
                                 - 0.304000686486818 z_k1_s1025
                                 - 0.746003299560859 z_k1_s1026
                                 - 0.29800062016132 z_k1_s1027
                                 - 0.239999869184551 z_k1_s1030
                                 - 0.16699992618091 z_k1_s1031
                                 - 0.628999777601129 z_k1_s1032
                                 - 0.299999836480688 z_k1_s1033
                                 - 0.238999990176235 z_k1_s1034
                                 - 0.342999279636463 z_k1_s1035
                                 - 0.171999837422288 z_k1_s1036
                                 - 0.490998923660377 z_k1_s1037
                                 - 0.459999534637378 z_k1_s1040
                                 - 0.247000240533861 z_k1_s1041
                                 - 0.204000105178514 z_k1_s1042
                                 - 0.17200001062105 z_k1_s1043
                                 - 0.39500037928694 z_k1_s1044
                                 - 0.211000034631651 z_k1_s1045
                                 - 0.362000522293459 z_k1_s1046
                                 - 0.602999799609694 z_k1_s1047
                                 - 1.38200024899205 z_k1_s1050
                                 - 1.11599691269139 z_k1_s1051
                                 - 1.11900048508858 z_k1_s1052
                                 - 0.794000892702255 z_k1_s1053
                                 - 1.71699869035859 z_k1_s1054
                                 - 0.976996805753631 z_k1_s1055
                                 - 0.630000616211539 z_k1_s1056
                                 - 0.9709983669662 z_k1_s1057
                                 - 0.432998960839165 z_k1_s1060
                                 - 0.301999886937625 z_k1_s1061
                                 - 0.249000009779021 z_k1_s1062
                                 - 0.369998727446914 z_k1_s1063
                                 - 0.294000041830647 z_k1_s1064
                                 - 0.390999852577521 z_k1_s1065
                                 - 0.286999980987069 z_k1_s1066
                                 - 0.815999282509666 z_k1_s1067
                                 - 0.951999618779893 z_k1_s1070
                                 - 0.245999760829305 z_k1_s1071
                                 - 0.287999971234225 z_k1_s1072
                                 - 0.370999283377648 z_k1_s1073
                                 - 0.385999818379252 z_k1_s1074
                                 - 0.355000858708555 z_k1_s1075
                                 - 0.428999206051231 z_k1_s1076
                                 - 0.192000088122781 z_k1_s1077
                                 - 1.08599543704756 z_k1_s1100
                                 - 0.422999193819923 z_k1_s1101
                                 - 0.230999167299822 z_k1_s1102
                                 - 0.701000441538659 z_k1_s1103
                                 - 0.476998963935555 z_k1_s1104
                                 - 0.723996958031709 z_k1_s1105
                                 - 1.07399536757773 z_k1_s1106
                                 - 0.708996871194417 z_k1_s1107
                                 - 0.652998615242656 z_k1_s1110
                                 - 0.649001419733103 z_k1_s1111
                                 - 0.314999952155418 z_k1_s1112
                                 - 0.259000115816697 z_k1_s1113
                                 - 0.678999586557196 z_k1_s1114
                                 - 0.354001409127228 z_k1_s1115
                                 - 0.243999985104517 z_k1_s1116
                                 - 0.338000059487415 z_k1_s1117
                                 - 0.58100040725748 z_k1_s1120
                                 - 1.5660048402202 z_k1_s1121
                                 - 0.764998365573827 z_k1_s1122
                                 - 0.762997915069797 z_k1_s1123
                                 - 0.376000617852444 z_k1_s1124
                                 - 1.25900091993321 z_k1_s1125
                                 - 1.27700497446948 z_k1_s1126
                                 - 0.661001032196916 z_k1_s1127
                                 - 0.307999992418571 z_k1_s1130
                                 - 0.481998749645156 z_k1_s1131
                                 - 0.475000653851768 z_k1_s1132
                                 - 0.489998610552116 z_k1_s1133
                                 - 0.179000096293596 z_k1_s1134
                                 - 0.833001559439042 z_k1_s1135
                                 - 0.423999758069695 z_k1_s1136
                                 - 0.374000627401194 z_k1_s1137
                                 - 0.592997017037095 z_k1_s1140
                                 - 0.533000255617729 z_k1_s1141
                                 - 0.324999382561571 z_k1_s1142
                                 - 0.458998889047568 z_k1_s1143
                                 - 1.01899837655225 z_k1_s1144
                                 - 0.407999114597577 z_k1_s1145
                                 - 0.905998425811263 z_k1_s1146
                                 - 0.351999073947344 z_k1_s1147
                                 - 0.773000160880057 z_k1_s1150
                                 - 1.10700055823007 z_k1_s1151
                                 - 0.993002855378196 z_k1_s1152
                                 - 1.06799838014779 z_k1_s1153
                                 - 0.567000599874081 z_k1_s1154
                                 - 1.74299832809277 z_k1_s1155
                                 - 0.757000257462146 z_k1_s1156
                                 - 1.0370009807767 z_k1_s1157
                                 - 0.820999568346039 z_k1_s1160
                                 - 1.02199932862318 z_k1_s1161
                                 - 0.339999979947476 z_k1_s1162
                                 - 0.247000228837184 z_k1_s1163
                                 - 0.730001407016917 z_k1_s1164
                                 - 0.75300186139698 z_k1_s1165
                                 - 0.229000007209334 z_k1_s1166
                                 - 0.260000217712821 z_k1_s1167
                                 - 0.268999325448697 z_k1_s1170
                                 - 0.226999330843337 z_k1_s1171
                                 - 0.373000272090418 z_k1_s1172
                                 - 0.219000067870793 z_k1_s1173
                                 - 0.406998987723493 z_k1_s1174
                                 - 0.27299996493484 z_k1_s1175
                                 - 0.280999003907413 z_k1_s1176
                                 - 0.19100007146722 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.922925 w_k1_ADM_s000
                                 + 0.72914 w_k1_AXP_s000
                                 + 0.572182 w_k1_DIS_s000
                                 + 0.800648 w_k1_FDX_s000
                                 + 0.759395 w_k1_IBM_s000
                                 + 0.808143 w_k1_INTC_s000
                                 + 0.691227 w_k1_KO_s000
                                 + 0.603249 w_k1_MAT_s000
                                 + 0.801763 w_k1_MSFT_s000
                                 + 0.796828 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.792666 w_k1_ADM_s000
                                 + 0.750112 w_k1_AXP_s000
                                 + 0.631413 w_k1_DIS_s000
                                 + 0.72131 w_k1_FDX_s000
                                 + 1.06817 w_k1_IBM_s000
                                 + 0.762959 w_k1_INTC_s000
                                 + 0.870426 w_k1_KO_s000
                                 + 0.837383 w_k1_MAT_s000
                                 + 0.718761 w_k1_MSFT_s000
                                 + 0.898798 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.952598 w_k1_ADM_s000
                                 + 1.19221 w_k1_AXP_s000
                                 + 1.17449 w_k1_DIS_s000
                                 + 1.11231 w_k1_FDX_s000
                                 + 1.05143 w_k1_IBM_s000
                                 + 1.11558 w_k1_INTC_s000
                                 + 1.04682 w_k1_KO_s000
                                 + 0.953057 w_k1_MAT_s000
                                 + 1.05572 w_k1_MSFT_s000
                                 + 1.08996 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.08529 w_k1_ADM_s000
                                 + 1.26595 w_k1_AXP_s000
                                 + 1.04625 w_k1_DIS_s000
                                 + 1.10125 w_k1_FDX_s000
                                 + 1.38107 w_k1_IBM_s000
                                 + 1.09754 w_k1_INTC_s000 + 1.1311 w_k1_KO_s000
                                 + 1.08211 w_k1_MAT_s000
                                 + 0.986487 w_k1_MSFT_s000
                                 + 1.11429 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.76932 w_k1_ADM_s000
                                 + 0.895131 w_k1_AXP_s000
                                 + 1.52708 w_k1_DIS_s000
                                 + 1.05822 w_k1_FDX_s000
                                 + 1.06149 w_k1_IBM_s000
                                 + 1.10362 w_k1_INTC_s000
                                 + 1.17311 w_k1_KO_s000
                                 + 0.905477 w_k1_MAT_s000
                                 + 1.16048 w_k1_MSFT_s000
                                 + 1.11303 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.806309 w_k1_ADM_s000
                                 + 0.806615 w_k1_AXP_s000
                                 + 0.985268 w_k1_DIS_s000
                                 + 1.01164 w_k1_FDX_s000
                                 + 0.816593 w_k1_IBM_s000
                                 + 0.935212 w_k1_INTC_s000
                                 + 0.889041 w_k1_KO_s000
                                 + 0.524655 w_k1_MAT_s000
                                 + 1.02499 w_k1_MSFT_s000
                                 + 0.982147 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.26801 w_k1_ADM_s000
                                 + 1.18062 w_k1_AXP_s000
                                 + 1.37463 w_k1_DIS_s000
                                 + 1.08336 w_k1_FDX_s000
                                 + 1.80013 w_k1_IBM_s000
                                 + 0.943303 w_k1_INTC_s000
                                 + 1.51453 w_k1_KO_s000 + 1.3126 w_k1_MAT_s000
                                 + 1.16592 w_k1_MSFT_s000
                                 + 1.15018 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.853028 w_k1_ADM_s000
                                 + 0.812378 w_k1_AXP_s000
                                 + 0.754789 w_k1_DIS_s000
                                 + 0.81051 w_k1_FDX_s000
                                 + 1.31888 w_k1_IBM_s000
                                 + 0.605299 w_k1_INTC_s000
                                 + 1.03902 w_k1_KO_s000 + 1.17284 w_k1_MAT_s000
                                 + 0.780478 w_k1_MSFT_s000
                                 + 0.891955 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.11917 w_k1_ADM_s001
                                 + 0.941707 w_k1_AXP_s001
                                 + 0.858305 w_k1_DIS_s001
                                 + 0.965007 w_k1_FDX_s001
                                 + 0.66785 w_k1_IBM_s001
                                 + 0.939619 w_k1_INTC_s001
                                 + 0.929644 w_k1_KO_s001
                                 + 0.701106 w_k1_MAT_s001
                                 + 0.980181 w_k1_MSFT_s001
                                 + 1.0793 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.714215 w_k1_ADM_s001
                                 + 0.547817 w_k1_AXP_s001
                                 + 0.689943 w_k1_DIS_s001
                                 + 0.712006 w_k1_FDX_s001
                                 + 0.755147 w_k1_IBM_s001
                                 + 0.877555 w_k1_INTC_s001
                                 + 0.906007 w_k1_KO_s001
                                 + 0.506152 w_k1_MAT_s001
                                 + 0.995836 w_k1_MSFT_s001
                                 + 0.945312 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.03886 w_k1_ADM_s001
                                 + 1.19797 w_k1_AXP_s001
                                 + 0.952107 w_k1_DIS_s001
                                 + 0.949909 w_k1_FDX_s001
                                 + 0.907811 w_k1_IBM_s001
                                 + 0.87633 w_k1_INTC_s001
                                 + 1.06826 w_k1_KO_s001 + 1.14178 w_k1_MAT_s001
                                 + 1.04156 w_k1_MSFT_s001
                                 + 0.916404 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.1424 w_k1_ADM_s001
                                 + 1.58721 w_k1_AXP_s001
                                 + 1.51541 w_k1_DIS_s001
                                 + 1.28382 w_k1_FDX_s001
                                 + 1.26922 w_k1_IBM_s001
                                 + 1.17326 w_k1_INTC_s001
                                 + 1.30722 w_k1_KO_s001 + 1.59826 w_k1_MAT_s001
                                 + 1.12407 w_k1_MSFT_s001
                                 + 1.10226 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.972184 w_k1_ADM_s001
                                 + 1.12158 w_k1_AXP_s001
                                 + 1.14252 w_k1_DIS_s001
                                 + 0.868448 w_k1_FDX_s001
                                 + 1.62828 w_k1_IBM_s001
                                 + 1.07255 w_k1_INTC_s001
                                 + 1.37208 w_k1_KO_s001 + 1.15686 w_k1_MAT_s001
                                 + 0.972421 w_k1_MSFT_s001
                                 + 1.14348 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.649524 w_k1_ADM_s001
                                 + 0.690685 w_k1_AXP_s001
                                 + 0.841516 w_k1_DIS_s001
                                 + 0.788799 w_k1_FDX_s001
                                 + 1.01156 w_k1_IBM_s001
                                 + 0.967175 w_k1_INTC_s001
                                 + 0.978071 w_k1_KO_s001
                                 + 0.688799 w_k1_MAT_s001
                                 + 0.931663 w_k1_MSFT_s001
                                 + 0.909038 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.881198 w_k1_ADM_s001
                                 + 0.89606 w_k1_AXP_s001
                                 + 0.585745 w_k1_DIS_s001
                                 + 0.742219 w_k1_FDX_s001
                                 + 0.832468 w_k1_IBM_s001
                                 + 0.894044 w_k1_INTC_s001
                                 + 0.894811 w_k1_KO_s001
                                 + 0.980182 w_k1_MAT_s001
                                 + 0.787123 w_k1_MSFT_s001
                                 + 0.947541 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.05648 w_k1_ADM_s001
                                 + 1.19481 w_k1_AXP_s001
                                 + 1.16003 w_k1_DIS_s001
                                 + 1.08917 w_k1_FDX_s001
                                 + 1.10628 w_k1_IBM_s001
                                 + 1.08812 w_k1_INTC_s001
                                 + 1.24351 w_k1_KO_s001
                                 + 0.921983 w_k1_MAT_s001
                                 + 1.23307 w_k1_MSFT_s001
                                 + 1.17852 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.864247 w_k1_ADM_s002
                                 + 0.822167 w_k1_AXP_s002
                                 + 0.61267 w_k1_DIS_s002
                                 + 0.794312 w_k1_FDX_s002
                                 + 1.0099 w_k1_IBM_s002 + 1.0051 w_k1_INTC_s002
                                 + 0.760314 w_k1_KO_s002
                                 + 1.39348 w_k1_MAT_s002
                                 + 0.873337 w_k1_MSFT_s002
                                 + 0.855452 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.844703 w_k1_ADM_s002
                                 + 1.12002 w_k1_AXP_s002
                                 + 0.909081 w_k1_DIS_s002
                                 + 1.07153 w_k1_FDX_s002
                                 + 1.33977 w_k1_IBM_s002
                                 + 1.17078 w_k1_INTC_s002
                                 + 1.14882 w_k1_KO_s002 + 1.57844 w_k1_MAT_s002
                                 + 1.04524 w_k1_MSFT_s002
                                 + 0.944058 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.05055 w_k1_ADM_s002
                                 + 1.39253 w_k1_AXP_s002
                                 + 1.43331 w_k1_DIS_s002 + 1.1892 w_k1_FDX_s002
                                 + 1.51964 w_k1_IBM_s002
                                 + 0.909522 w_k1_INTC_s002
                                 + 1.27647 w_k1_KO_s002 + 1.51992 w_k1_MAT_s002
                                 + 1.0129 w_k1_MSFT_s002 + 1.00457 w_k1_PG_s002
                                 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.03453 w_k1_ADM_s002
                                 + 1.55399 w_k1_AXP_s002
                                 + 1.27389 w_k1_DIS_s002
                                 + 1.17145 w_k1_FDX_s002
                                 + 1.22183 w_k1_IBM_s002
                                 + 1.18282 w_k1_INTC_s002
                                 + 1.15455 w_k1_KO_s002 + 1.36916 w_k1_MAT_s002
                                 + 1.11337 w_k1_MSFT_s002
                                 + 0.986797 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.97469 w_k1_ADM_s002
                                 + 0.846285 w_k1_AXP_s002
                                 + 0.810096 w_k1_DIS_s002 + 0.948 w_k1_FDX_s002
                                 + 0.864286 w_k1_IBM_s002
                                 + 0.863641 w_k1_INTC_s002
                                 + 0.811525 w_k1_KO_s002
                                 + 0.892298 w_k1_MAT_s002
                                 + 0.850185 w_k1_MSFT_s002
                                 + 0.811049 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.26175 w_k1_ADM_s002
                                 + 1.51014 w_k1_AXP_s002
                                 + 1.73714 w_k1_DIS_s002
                                 + 1.57705 w_k1_FDX_s002
                                 + 1.20036 w_k1_IBM_s002
                                 + 1.37486 w_k1_INTC_s002
                                 + 1.18844 w_k1_KO_s002 + 1.40122 w_k1_MAT_s002
                                 + 1.47859 w_k1_MSFT_s002
                                 + 1.29908 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.01107 w_k1_ADM_s002
                                 + 0.838844 w_k1_AXP_s002
                                 + 1.12388 w_k1_DIS_s002
                                 + 1.10663 w_k1_FDX_s002
                                 + 0.844414 w_k1_IBM_s002
                                 + 1.00095 w_k1_INTC_s002
                                 + 0.950152 w_k1_KO_s002
                                 + 0.845356 w_k1_MAT_s002
                                 + 1.12943 w_k1_MSFT_s002
                                 + 1.02652 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.35237 w_k1_ADM_s002
                                 + 1.20608 w_k1_AXP_s002
                                 + 1.21993 w_k1_DIS_s002
                                 + 1.28759 w_k1_FDX_s002
                                 + 0.948716 w_k1_IBM_s002
                                 + 1.12358 w_k1_INTC_s002
                                 + 0.945866 w_k1_KO_s002
                                 + 0.956567 w_k1_MAT_s002
                                 + 1.19645 w_k1_MSFT_s002
                                 + 1.01672 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.878652 w_k1_ADM_s003
                                 + 1.02673 w_k1_AXP_s003
                                 + 0.769245 w_k1_DIS_s003
                                 + 1.01406 w_k1_FDX_s003
                                 + 0.905607 w_k1_IBM_s003
                                 + 0.915165 w_k1_INTC_s003
                                 + 0.929105 w_k1_KO_s003
                                 + 1.45812 w_k1_MAT_s003
                                 + 0.844791 w_k1_MSFT_s003
                                 + 0.977626 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.33302 w_k1_ADM_s003
                                 + 1.49557 w_k1_AXP_s003
                                 + 1.27324 w_k1_DIS_s003
                                 + 1.32753 w_k1_FDX_s003
                                 + 1.36359 w_k1_IBM_s003
                                 + 1.14348 w_k1_INTC_s003 + 1.3598 w_k1_KO_s003
                                 + 1.8492 w_k1_MAT_s003
                                 + 0.974981 w_k1_MSFT_s003
                                 + 1.09986 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.08713 w_k1_ADM_s003
                                 + 0.99809 w_k1_AXP_s003
                                 + 1.03492 w_k1_DIS_s003
                                 + 0.957387 w_k1_FDX_s003
                                 + 1.57742 w_k1_IBM_s003
                                 + 1.04214 w_k1_INTC_s003
                                 + 1.40311 w_k1_KO_s003 + 2.16567 w_k1_MAT_s003
                                 + 0.891018 w_k1_MSFT_s003
                                 + 0.912877 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.779117 w_k1_ADM_s003
                                 + 0.743268 w_k1_AXP_s003
                                 + 0.789041 w_k1_DIS_s003
                                 + 0.808251 w_k1_FDX_s003
                                 + 1.00007 w_k1_IBM_s003
                                 + 0.834314 w_k1_INTC_s003
                                 + 0.909279 w_k1_KO_s003
                                 + 1.24763 w_k1_MAT_s003
                                 + 0.767323 w_k1_MSFT_s003
                                 + 0.991588 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.862393 w_k1_ADM_s003
                                 + 0.771789 w_k1_AXP_s003
                                 + 0.572713 w_k1_DIS_s003
                                 + 0.859702 w_k1_FDX_s003
                                 + 0.648103 w_k1_IBM_s003
                                 + 0.717747 w_k1_INTC_s003
                                 + 0.721115 w_k1_KO_s003
                                 + 0.907972 w_k1_MAT_s003
                                 + 0.690595 w_k1_MSFT_s003
                                 + 0.810123 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.34035 w_k1_ADM_s003
                                 + 0.968787 w_k1_AXP_s003
                                 + 1.0339 w_k1_DIS_s003 + 1.31484 w_k1_FDX_s003
                                 + 0.782274 w_k1_IBM_s003
                                 + 1.37643 w_k1_INTC_s003
                                 + 0.935302 w_k1_KO_s003
                                 + 0.628622 w_k1_MAT_s003
                                 + 1.04331 w_k1_MSFT_s003
                                 + 0.910121 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.00995 w_k1_ADM_s003
                                 + 1.10748 w_k1_AXP_s003
                                 + 1.45967 w_k1_DIS_s003
                                 + 1.16942 w_k1_FDX_s003
                                 + 1.20827 w_k1_IBM_s003
                                 + 1.30145 w_k1_INTC_s003
                                 + 1.28869 w_k1_KO_s003 + 1.12313 w_k1_MAT_s003
                                 + 1.05834 w_k1_MSFT_s003
                                 + 1.02698 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.21336 w_k1_ADM_s003
                                 + 1.43313 w_k1_AXP_s003 + 1.2337 w_k1_DIS_s003
                                 + 1.09956 w_k1_FDX_s003
                                 + 1.68144 w_k1_IBM_s003
                                 + 1.23847 w_k1_INTC_s003
                                 + 1.36179 w_k1_KO_s003 + 2.33017 w_k1_MAT_s003
                                 + 0.918232 w_k1_MSFT_s003
                                 + 1.07459 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.44507 w_k1_ADM_s004
                                 + 1.50986 w_k1_AXP_s004
                                 + 2.16469 w_k1_DIS_s004
                                 + 1.69698 w_k1_FDX_s004
                                 + 0.999294 w_k1_IBM_s004
                                 + 1.27312 w_k1_INTC_s004
                                 + 1.40575 w_k1_KO_s004 + 1.3143 w_k1_MAT_s004
                                 + 1.21789 w_k1_MSFT_s004
                                 + 1.23045 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.04223 w_k1_ADM_s004
                                 + 0.669817 w_k1_AXP_s004
                                 + 0.635812 w_k1_DIS_s004
                                 + 0.722788 w_k1_FDX_s004
                                 + 0.906271 w_k1_IBM_s004
                                 + 0.583757 w_k1_INTC_s004
                                 + 0.714113 w_k1_KO_s004
                                 + 1.02772 w_k1_MAT_s004
                                 + 0.613393 w_k1_MSFT_s004
                                 + 0.790806 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.33444 w_k1_ADM_s004
                                 + 1.14436 w_k1_AXP_s004
                                 + 0.691609 w_k1_DIS_s004
                                 + 0.953798 w_k1_FDX_s004
                                 + 0.857933 w_k1_IBM_s004
                                 + 0.786179 w_k1_INTC_s004
                                 + 0.616496 w_k1_KO_s004
                                 + 0.901016 w_k1_MAT_s004
                                 + 0.768414 w_k1_MSFT_s004
                                 + 0.861571 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.13691 w_k1_ADM_s004
                                 + 1.24971 w_k1_AXP_s004
                                 + 1.29714 w_k1_DIS_s004
                                 + 1.22194 w_k1_FDX_s004
                                 + 1.76667 w_k1_IBM_s004
                                 + 0.889952 w_k1_INTC_s004
                                 + 1.34901 w_k1_KO_s004 + 1.7981 w_k1_MAT_s004
                                 + 0.938824 w_k1_MSFT_s004
                                 + 0.985045 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.04975 w_k1_ADM_s004
                                 + 0.836156 w_k1_AXP_s004
                                 + 0.922012 w_k1_DIS_s004
                                 + 0.998309 w_k1_FDX_s004
                                 + 0.888408 w_k1_IBM_s004
                                 + 0.93518 w_k1_INTC_s004
                                 + 0.921464 w_k1_KO_s004
                                 + 0.910003 w_k1_MAT_s004
                                 + 0.859665 w_k1_MSFT_s004
                                 + 0.924612 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.23165 w_k1_ADM_s004
                                 + 0.883063 w_k1_AXP_s004
                                 + 1.11686 w_k1_DIS_s004
                                 + 1.17952 w_k1_FDX_s004
                                 + 0.580504 w_k1_IBM_s004
                                 + 0.850005 w_k1_INTC_s004
                                 + 0.9473 w_k1_KO_s004 + 0.925379 w_k1_MAT_s004
                                 + 1.06659 w_k1_MSFT_s004
                                 + 1.07975 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.832095 w_k1_ADM_s004
                                 + 0.760387 w_k1_AXP_s004
                                 + 0.818946 w_k1_DIS_s004
                                 + 0.810331 w_k1_FDX_s004
                                 + 1.53158 w_k1_IBM_s004
                                 + 0.760738 w_k1_INTC_s004
                                 + 1.1201 w_k1_KO_s004 + 1.63159 w_k1_MAT_s004
                                 + 0.761068 w_k1_MSFT_s004
                                 + 0.87888 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.17611 w_k1_ADM_s004
                                 + 1.15221 w_k1_AXP_s004
                                 + 1.31432 w_k1_DIS_s004
                                 + 1.17356 w_k1_FDX_s004
                                 + 1.24306 w_k1_IBM_s004
                                 + 1.16289 w_k1_INTC_s004
                                 + 1.24761 w_k1_KO_s004 + 1.866 w_k1_MAT_s004
                                 + 1.05462 w_k1_MSFT_s004
                                 + 1.06672 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.12042 w_k1_ADM_s005
                                 + 1.0846 w_k1_AXP_s005 + 1.25907 w_k1_DIS_s005
                                 + 1.10373 w_k1_FDX_s005
                                 + 1.03462 w_k1_IBM_s005
                                 + 0.993296 w_k1_INTC_s005
                                 + 1.05103 w_k1_KO_s005
                                 + 0.931061 w_k1_MAT_s005
                                 + 1.1711 w_k1_MSFT_s005
                                 + 0.947615 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.05645 w_k1_ADM_s005
                                 + 1.22393 w_k1_AXP_s005
                                 + 1.12098 w_k1_DIS_s005
                                 + 0.999852 w_k1_FDX_s005
                                 + 1.28188 w_k1_IBM_s005
                                 + 1.27938 w_k1_INTC_s005
                                 + 1.14905 w_k1_KO_s005 + 1.43047 w_k1_MAT_s005
                                 + 1.33892 w_k1_MSFT_s005
                                 + 1.10208 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.931599 w_k1_ADM_s005
                                 + 0.681155 w_k1_AXP_s005
                                 + 0.606694 w_k1_DIS_s005
                                 + 0.789503 w_k1_FDX_s005
                                 + 0.702679 w_k1_IBM_s005
                                 + 0.877879 w_k1_INTC_s005
                                 + 0.757286 w_k1_KO_s005
                                 + 0.393148 w_k1_MAT_s005
                                 + 0.811822 w_k1_MSFT_s005
                                 + 0.868869 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.999448 w_k1_ADM_s005
                                 + 0.940719 w_k1_AXP_s005
                                 + 0.784431 w_k1_DIS_s005
                                 + 0.922629 w_k1_FDX_s005
                                 + 1.03763 w_k1_IBM_s005
                                 + 1.01007 w_k1_INTC_s005
                                 + 0.893095 w_k1_KO_s005
                                 + 0.940465 w_k1_MAT_s005
                                 + 0.846672 w_k1_MSFT_s005
                                 + 0.962946 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.25512 w_k1_ADM_s005
                                 + 1.0354 w_k1_AXP_s005 + 1.10183 w_k1_DIS_s005
                                 + 1.24443 w_k1_FDX_s005
                                 + 0.777103 w_k1_IBM_s005
                                 + 1.11745 w_k1_INTC_s005
                                 + 0.978058 w_k1_KO_s005
                                 + 0.681173 w_k1_MAT_s005
                                 + 1.27202 w_k1_MSFT_s005
                                 + 1.18632 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.44703 w_k1_ADM_s005
                                 + 1.30072 w_k1_AXP_s005 + 1.5585 w_k1_DIS_s005
                                 + 1.41414 w_k1_FDX_s005
                                 + 1.04908 w_k1_IBM_s005
                                 + 1.29549 w_k1_INTC_s005
                                 + 1.35625 w_k1_KO_s005 + 1.00921 w_k1_MAT_s005
                                 + 1.35644 w_k1_MSFT_s005
                                 + 1.25406 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.789944 w_k1_ADM_s005
                                 + 0.87003 w_k1_AXP_s005
                                 + 0.881275 w_k1_DIS_s005
                                 + 0.904366 w_k1_FDX_s005
                                 + 0.941918 w_k1_IBM_s005
                                 + 1.2137 w_k1_INTC_s005
                                 + 0.886871 w_k1_KO_s005
                                 + 0.617464 w_k1_MAT_s005
                                 + 0.928665 w_k1_MSFT_s005
                                 + 0.94954 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.12496 w_k1_ADM_s005
                                 + 1.2888 w_k1_AXP_s005 + 1.30959 w_k1_DIS_s005
                                 + 0.983386 w_k1_FDX_s005
                                 + 1.67556 w_k1_IBM_s005
                                 + 1.14442 w_k1_INTC_s005
                                 + 1.38565 w_k1_KO_s005 + 1.96198 w_k1_MAT_s005
                                 + 1.20691 w_k1_MSFT_s005
                                 + 1.17431 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.950206 w_k1_ADM_s006
                                 + 1.19914 w_k1_AXP_s006
                                 + 0.922426 w_k1_DIS_s006
                                 + 0.98507 w_k1_FDX_s006
                                 + 1.37883 w_k1_IBM_s006
                                 + 0.920645 w_k1_INTC_s006
                                 + 0.913867 w_k1_KO_s006
                                 + 1.50555 w_k1_MAT_s006
                                 + 0.872459 w_k1_MSFT_s006
                                 + 1.01552 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.20533 w_k1_ADM_s006
                                 + 0.977539 w_k1_AXP_s006
                                 + 1.44004 w_k1_DIS_s006
                                 + 1.23478 w_k1_FDX_s006
                                 + 1.17649 w_k1_IBM_s006
                                 + 1.45751 w_k1_INTC_s006
                                 + 1.27366 w_k1_KO_s006
                                 + 0.772676 w_k1_MAT_s006
                                 + 1.28751 w_k1_MSFT_s006 + 1.0907 w_k1_PG_s006
                                 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.751065 w_k1_ADM_s006
                                 + 0.706601 w_k1_AXP_s006
                                 + 0.696563 w_k1_DIS_s006
                                 + 0.756912 w_k1_FDX_s006
                                 + 0.787398 w_k1_IBM_s006
                                 + 0.871465 w_k1_INTC_s006
                                 + 0.705793 w_k1_KO_s006
                                 + 0.553669 w_k1_MAT_s006
                                 + 0.836987 w_k1_MSFT_s006
                                 + 0.819494 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.10759 w_k1_ADM_s006
                                 + 1.51547 w_k1_AXP_s006 + 1.6963 w_k1_DIS_s006
                                 + 1.35369 w_k1_FDX_s006
                                 + 1.28415 w_k1_IBM_s006
                                 + 1.38721 w_k1_INTC_s006
                                 + 1.35857 w_k1_KO_s006 + 1.64814 w_k1_MAT_s006
                                 + 1.44757 w_k1_MSFT_s006
                                 + 1.35907 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.973346 w_k1_ADM_s006
                                 + 0.883917 w_k1_AXP_s006
                                 + 0.862454 w_k1_DIS_s006
                                 + 0.874281 w_k1_FDX_s006
                                 + 1.11133 w_k1_IBM_s006
                                 + 1.14392 w_k1_INTC_s006 + 1.0678 w_k1_KO_s006
                                 + 0.96422 w_k1_MAT_s006
                                 + 0.981292 w_k1_MSFT_s006
                                 + 1.01129 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.41304 w_k1_ADM_s006
                                 + 1.20285 w_k1_AXP_s006
                                 + 1.12191 w_k1_DIS_s006
                                 + 1.21198 w_k1_FDX_s006
                                 + 0.922487 w_k1_IBM_s006
                                 + 1.22676 w_k1_INTC_s006
                                 + 0.93203 w_k1_KO_s006 + 1.16005 w_k1_MAT_s006
                                 + 1.17438 w_k1_MSFT_s006 + 1.2349 w_k1_PG_s006
                                 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.07252 w_k1_ADM_s006
                                 + 0.912376 w_k1_AXP_s006
                                 + 0.91941 w_k1_DIS_s006
                                 + 0.934812 w_k1_FDX_s006
                                 + 0.886052 w_k1_IBM_s006
                                 + 0.906945 w_k1_INTC_s006
                                 + 0.879762 w_k1_KO_s006
                                 + 0.865584 w_k1_MAT_s006
                                 + 0.964799 w_k1_MSFT_s006
                                 + 1.05905 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.21817 w_k1_ADM_s006
                                 + 0.963271 w_k1_AXP_s006
                                 + 1.23637 w_k1_DIS_s006
                                 + 1.11638 w_k1_FDX_s006
                                 + 0.745874 w_k1_IBM_s006
                                 + 1.16261 w_k1_INTC_s006
                                 + 0.905925 w_k1_KO_s006
                                 + 0.911461 w_k1_MAT_s006
                                 + 1.34155 w_k1_MSFT_s006
                                 + 1.04689 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.05504 w_k1_ADM_s007
                                 + 1.03164 w_k1_AXP_s007
                                 + 0.892068 w_k1_DIS_s007
                                 + 0.906334 w_k1_FDX_s007
                                 + 1.34942 w_k1_IBM_s007
                                 + 0.877002 w_k1_INTC_s007
                                 + 1.05979 w_k1_KO_s007 + 1.14615 w_k1_MAT_s007
                                 + 0.872568 w_k1_MSFT_s007
                                 + 0.972965 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.50691 w_k1_ADM_s007
                                 + 1.2947 w_k1_AXP_s007 + 1.51408 w_k1_DIS_s007
                                 + 1.38135 w_k1_FDX_s007
                                 + 0.967545 w_k1_IBM_s007
                                 + 1.19158 w_k1_INTC_s007
                                 + 1.25801 w_k1_KO_s007
                                 + 0.891463 w_k1_MAT_s007
                                 + 1.18703 w_k1_MSFT_s007
                                 + 1.24902 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.21632 w_k1_ADM_s007
                                 + 1.05331 w_k1_AXP_s007
                                 + 0.664617 w_k1_DIS_s007
                                 + 0.934992 w_k1_FDX_s007
                                 + 0.875786 w_k1_IBM_s007
                                 + 0.861467 w_k1_INTC_s007
                                 + 0.778772 w_k1_KO_s007
                                 + 1.02323 w_k1_MAT_s007
                                 + 0.849511 w_k1_MSFT_s007
                                 + 1.05454 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.58966 w_k1_ADM_s007
                                 + 1.2527 w_k1_AXP_s007 + 1.27421 w_k1_DIS_s007
                                 + 1.18172 w_k1_FDX_s007
                                 + 1.47892 w_k1_IBM_s007
                                 + 1.08964 w_k1_INTC_s007
                                 + 1.20464 w_k1_KO_s007 + 1.11075 w_k1_MAT_s007
                                 + 1.10759 w_k1_MSFT_s007
                                 + 1.28971 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.876524 w_k1_ADM_s007
                                 + 0.817811 w_k1_AXP_s007
                                 + 0.956699 w_k1_DIS_s007
                                 + 0.953984 w_k1_FDX_s007
                                 + 0.859014 w_k1_IBM_s007
                                 + 1.03367 w_k1_INTC_s007
                                 + 0.927027 w_k1_KO_s007
                                 + 0.835925 w_k1_MAT_s007
                                 + 0.998206 w_k1_MSFT_s007
                                 + 0.901319 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.27732 w_k1_ADM_s007
                                 + 0.913043 w_k1_AXP_s007
                                 + 0.941314 w_k1_DIS_s007
                                 + 1.12306 w_k1_FDX_s007
                                 + 0.626931 w_k1_IBM_s007
                                 + 0.959072 w_k1_INTC_s007
                                 + 0.762318 w_k1_KO_s007
                                 + 0.374441 w_k1_MAT_s007
                                 + 0.800333 w_k1_MSFT_s007
                                 + 0.892819 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.959323 w_k1_ADM_s007
                                 + 0.658607 w_k1_AXP_s007
                                 + 0.537836 w_k1_DIS_s007
                                 + 0.676536 w_k1_FDX_s007
                                 + 0.708782 w_k1_IBM_s007
                                 + 0.751969 w_k1_INTC_s007
                                 + 0.657687 w_k1_KO_s007
                                 + 0.614294 w_k1_MAT_s007
                                 + 0.737106 w_k1_MSFT_s007
                                 + 0.834254 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.742063 w_k1_ADM_s007
                                 + 0.662663 w_k1_AXP_s007
                                 + 0.710693 w_k1_DIS_s007
                                 + 0.768865 w_k1_FDX_s007
                                 + 0.941092 w_k1_IBM_s007
                                 + 0.82612 w_k1_INTC_s007
                                 + 0.860383 w_k1_KO_s007
                                 + 0.671415 w_k1_MAT_s007
                                 + 0.772222 w_k1_MSFT_s007
                                 + 0.954739 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.953078 w_k1_ADM_s010
                                 + 0.587042 w_k1_AXP_s010
                                 + 0.759316 w_k1_DIS_s010
                                 + 0.809676 w_k1_FDX_s010
                                 + 0.784316 w_k1_IBM_s010
                                 + 0.956808 w_k1_INTC_s010
                                 + 0.917114 w_k1_KO_s010
                                 + 0.626915 w_k1_MAT_s010
                                 + 0.753037 w_k1_MSFT_s010
                                 + 0.897351 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.12238 w_k1_ADM_s010
                                 + 1.31159 w_k1_AXP_s010
                                 + 1.05764 w_k1_DIS_s010
                                 + 1.09034 w_k1_FDX_s010
                                 + 1.18059 w_k1_IBM_s010
                                 + 1.02123 w_k1_INTC_s010
                                 + 1.11912 w_k1_KO_s010 + 1.25875 w_k1_MAT_s010
                                 + 1.00574 w_k1_MSFT_s010
                                 + 1.18552 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.909723 w_k1_ADM_s010
                                 + 0.95392 w_k1_AXP_s010
                                 + 0.976293 w_k1_DIS_s010
                                 + 1.0681 w_k1_FDX_s010
                                 + 0.754445 w_k1_IBM_s010
                                 + 1.33329 w_k1_INTC_s010
                                 + 0.87545 w_k1_KO_s010
                                 + 0.540367 w_k1_MAT_s010
                                 + 1.02629 w_k1_MSFT_s010
                                 + 0.903722 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.942133 w_k1_ADM_s010
                                 + 1.09847 w_k1_AXP_s010
                                 + 0.992705 w_k1_DIS_s010
                                 + 0.929052 w_k1_FDX_s010
                                 + 1.42366 w_k1_IBM_s010
                                 + 1.16001 w_k1_INTC_s010 + 1.0121 w_k1_KO_s010
                                 + 1.15821 w_k1_MAT_s010
                                 + 0.911791 w_k1_MSFT_s010
                                 + 1.19764 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.800245 w_k1_ADM_s010
                                 + 0.756117 w_k1_AXP_s010
                                 + 1.03069 w_k1_DIS_s010
                                 + 0.911337 w_k1_FDX_s010
                                 + 1.13203 w_k1_IBM_s010
                                 + 0.996251 w_k1_INTC_s010
                                 + 1.14134 w_k1_KO_s010
                                 + 0.846377 w_k1_MAT_s010
                                 + 0.973487 w_k1_MSFT_s010
                                 + 1.08915 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.891797 w_k1_ADM_s010
                                 + 0.716134 w_k1_AXP_s010
                                 + 0.708245 w_k1_DIS_s010
                                 + 0.688306 w_k1_FDX_s010
                                 + 1.05747 w_k1_IBM_s010
                                 + 0.923618 w_k1_INTC_s010
                                 + 0.923196 w_k1_KO_s010
                                 + 0.864246 w_k1_MAT_s010
                                 + 0.722734 w_k1_MSFT_s010
                                 + 0.923374 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.07229 w_k1_ADM_s010
                                 + 1.41452 w_k1_AXP_s010
                                 + 1.41522 w_k1_DIS_s010
                                 + 1.30996 w_k1_FDX_s010
                                 + 1.08005 w_k1_IBM_s010
                                 + 1.55404 w_k1_INTC_s010
                                 + 1.21111 w_k1_KO_s010 + 1.13866 w_k1_MAT_s010
                                 + 1.40692 w_k1_MSFT_s010
                                 + 1.37129 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.785556 w_k1_ADM_s010
                                 + 0.826853 w_k1_AXP_s010
                                 + 0.626495 w_k1_DIS_s010
                                 + 0.817651 w_k1_FDX_s010
                                 + 0.960953 w_k1_IBM_s010
                                 + 1.04534 w_k1_INTC_s010
                                 + 0.934084 w_k1_KO_s010
                                 + 0.964983 w_k1_MAT_s010
                                 + 0.823431 w_k1_MSFT_s010
                                 + 0.938067 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.34236 w_k1_ADM_s011
                                 + 1.3312 w_k1_AXP_s011 + 1.09577 w_k1_DIS_s011
                                 + 1.20069 w_k1_FDX_s011
                                 + 0.991467 w_k1_IBM_s011
                                 + 1.02325 w_k1_INTC_s011
                                 + 0.992315 w_k1_KO_s011
                                 + 0.994803 w_k1_MAT_s011
                                 + 1.08686 w_k1_MSFT_s011
                                 + 1.00541 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.868232 w_k1_ADM_s011
                                 + 0.880468 w_k1_AXP_s011
                                 + 0.648133 w_k1_DIS_s011
                                 + 0.777069 w_k1_FDX_s011
                                 + 0.888604 w_k1_IBM_s011
                                 + 0.826466 w_k1_INTC_s011
                                 + 0.74963 w_k1_KO_s011 + 1.05265 w_k1_MAT_s011
                                 + 0.705335 w_k1_MSFT_s011
                                 + 0.774993 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.10158 w_k1_ADM_s011
                                 + 1.28686 w_k1_AXP_s011
                                 + 1.06329 w_k1_DIS_s011
                                 + 1.01291 w_k1_FDX_s011 + 1.3021 w_k1_IBM_s011
                                 + 0.952353 w_k1_INTC_s011
                                 + 1.17267 w_k1_KO_s011 + 1.29954 w_k1_MAT_s011
                                 + 0.931418 w_k1_MSFT_s011
                                 + 0.957749 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.57475 w_k1_ADM_s011
                                 + 1.6214 w_k1_AXP_s011 + 1.7123 w_k1_DIS_s011
                                 + 1.48584 w_k1_FDX_s011
                                 + 1.01787 w_k1_IBM_s011
                                 + 1.39231 w_k1_INTC_s011
                                 + 1.28456 w_k1_KO_s011 + 1.37482 w_k1_MAT_s011
                                 + 1.4914 w_k1_MSFT_s011 + 1.27379 w_k1_PG_s011
                                 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.990253 w_k1_ADM_s011
                                 + 0.686383 w_k1_AXP_s011
                                 + 0.626653 w_k1_DIS_s011
                                 + 0.785488 w_k1_FDX_s011
                                 + 0.50827 w_k1_IBM_s011
                                 + 0.977462 w_k1_INTC_s011
                                 + 0.595854 w_k1_KO_s011
                                 + 0.370404 w_k1_MAT_s011
                                 + 0.817671 w_k1_MSFT_s011
                                 + 0.770544 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.14084 w_k1_ADM_s011
                                 + 0.783169 w_k1_AXP_s011
                                 + 0.923883 w_k1_DIS_s011
                                 + 1.01812 w_k1_FDX_s011
                                 + 0.808197 w_k1_IBM_s011
                                 + 0.898852 w_k1_INTC_s011
                                 + 0.914694 w_k1_KO_s011
                                 + 0.778676 w_k1_MAT_s011
                                 + 0.99303 w_k1_MSFT_s011
                                 + 0.920473 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.02722 w_k1_ADM_s011
                                 + 0.949936 w_k1_AXP_s011
                                 + 1.51893 w_k1_DIS_s011
                                 + 1.19674 w_k1_FDX_s011
                                 + 1.15264 w_k1_IBM_s011
                                 + 1.09566 w_k1_INTC_s011
                                 + 1.21045 w_k1_KO_s011 + 1.24826 w_k1_MAT_s011
                                 + 1.22689 w_k1_MSFT_s011
                                 + 1.07567 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.709011 w_k1_ADM_s011
                                 + 0.799532 w_k1_AXP_s011
                                 + 0.845789 w_k1_DIS_s011
                                 + 0.855656 w_k1_FDX_s011
                                 + 0.869076 w_k1_IBM_s011
                                 + 1.03596 w_k1_INTC_s011
                                 + 0.777865 w_k1_KO_s011
                                 + 0.72194 w_k1_MAT_s011
                                 + 0.93748 w_k1_MSFT_s011
                                 + 0.81501 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.02657 w_k1_ADM_s012
                                 + 0.884739 w_k1_AXP_s012
                                 + 0.864424 w_k1_DIS_s012
                                 + 0.956527 w_k1_FDX_s012
                                 + 0.568935 w_k1_IBM_s012
                                 + 0.942131 w_k1_INTC_s012
                                 + 0.81671 w_k1_KO_s012 + 0.59471 w_k1_MAT_s012
                                 + 1.05628 w_k1_MSFT_s012
                                 + 0.877919 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.00217 w_k1_ADM_s012
                                 + 1.13021 w_k1_AXP_s012
                                 + 1.25775 w_k1_DIS_s012
                                 + 0.881156 w_k1_FDX_s012
                                 + 1.84237 w_k1_IBM_s012
                                 + 1.15883 w_k1_INTC_s012
                                 + 1.40621 w_k1_KO_s012 + 1.02817 w_k1_MAT_s012
                                 + 1.14193 w_k1_MSFT_s012
                                 + 0.950022 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.867785 w_k1_ADM_s012
                                 + 1.33724 w_k1_AXP_s012
                                 + 1.39402 w_k1_DIS_s012
                                 + 1.07419 w_k1_FDX_s012
                                 + 1.39261 w_k1_IBM_s012
                                 + 1.07692 w_k1_INTC_s012
                                 + 1.22519 w_k1_KO_s012 + 1.63459 w_k1_MAT_s012
                                 + 1.18446 w_k1_MSFT_s012
                                 + 1.12025 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.819968 w_k1_ADM_s012
                                 + 0.716563 w_k1_AXP_s012
                                 + 0.60315 w_k1_DIS_s012
                                 + 0.755168 w_k1_FDX_s012
                                 + 0.750372 w_k1_IBM_s012
                                 + 0.879183 w_k1_INTC_s012
                                 + 0.7441 w_k1_KO_s012 + 0.565724 w_k1_MAT_s012
                                 + 0.889373 w_k1_MSFT_s012
                                 + 0.788388 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.24235 w_k1_ADM_s012
                                 + 1.43235 w_k1_AXP_s012
                                 + 1.46289 w_k1_DIS_s012
                                 + 1.38939 w_k1_FDX_s012
                                 + 0.860284 w_k1_IBM_s012
                                 + 1.24157 w_k1_INTC_s012
                                 + 1.24781 w_k1_KO_s012 + 1.21722 w_k1_MAT_s012
                                 + 1.51648 w_k1_MSFT_s012
                                 + 1.17817 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.980195 w_k1_ADM_s012
                                 + 0.802258 w_k1_AXP_s012
                                 + 0.735984 w_k1_DIS_s012
                                 + 0.717642 w_k1_FDX_s012
                                 + 1.10598 w_k1_IBM_s012
                                 + 0.640588 w_k1_INTC_s012
                                 + 0.84735 w_k1_KO_s012
                                 + 0.940036 w_k1_MAT_s012
                                 + 0.73686 w_k1_MSFT_s012
                                 + 0.853634 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.975522 w_k1_ADM_s012
                                 + 1.10527 w_k1_AXP_s012
                                 + 1.10517 w_k1_DIS_s012
                                 + 0.979872 w_k1_FDX_s012
                                 + 1.03148 w_k1_IBM_s012
                                 + 0.971544 w_k1_INTC_s012
                                 + 0.963324 w_k1_KO_s012 + 1.052 w_k1_MAT_s012
                                 + 1.04426 w_k1_MSFT_s012
                                 + 1.02309 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.11689 w_k1_ADM_s012
                                 + 1.06426 w_k1_AXP_s012
                                 + 0.893371 w_k1_DIS_s012
                                 + 0.982492 w_k1_FDX_s012
                                 + 0.854988 w_k1_IBM_s012
                                 + 1.00034 w_k1_INTC_s012
                                 + 0.889016 w_k1_KO_s012
                                 + 0.820374 w_k1_MAT_s012
                                 + 0.980173 w_k1_MSFT_s012
                                 + 0.965374 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.1747 w_k1_ADM_s013
                                 + 1.37159 w_k1_AXP_s013
                                 + 0.836733 w_k1_DIS_s013
                                 + 1.16386 w_k1_FDX_s013
                                 + 0.984822 w_k1_IBM_s013
                                 + 1.19949 w_k1_INTC_s013
                                 + 0.880606 w_k1_KO_s013
                                 + 1.09899 w_k1_MAT_s013
                                 + 1.0236 w_k1_MSFT_s013
                                 + 0.889142 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.1794 w_k1_ADM_s013
                                 + 1.17857 w_k1_AXP_s013
                                 + 1.31499 w_k1_DIS_s013
                                 + 1.23866 w_k1_FDX_s013
                                 + 0.858153 w_k1_IBM_s013
                                 + 1.22208 w_k1_INTC_s013
                                 + 0.940424 w_k1_KO_s013
                                 + 0.587016 w_k1_MAT_s013
                                 + 1.15026 w_k1_MSFT_s013
                                 + 0.853289 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.4319 w_k1_ADM_s013
                                 + 1.7511 w_k1_AXP_s013 + 1.10388 w_k1_DIS_s013
                                 + 1.16996 w_k1_FDX_s013
                                 + 1.47492 w_k1_IBM_s013
                                 + 1.22023 w_k1_INTC_s013
                                 + 1.18009 w_k1_KO_s013 + 1.81839 w_k1_MAT_s013
                                 + 0.941788 w_k1_MSFT_s013
                                 + 0.990796 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.37605 w_k1_ADM_s013
                                 + 1.0164 w_k1_AXP_s013 + 1.48567 w_k1_DIS_s013
                                 + 1.08594 w_k1_FDX_s013 + 1.338 w_k1_IBM_s013
                                 + 0.988455 w_k1_INTC_s013
                                 + 1.40888 w_k1_KO_s013 + 1.29944 w_k1_MAT_s013
                                 + 1.2183 w_k1_MSFT_s013
                                 + 0.951974 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 2.01293 w_k1_ADM_s013
                                 + 1.70244 w_k1_AXP_s013
                                 + 1.88033 w_k1_DIS_s013
                                 + 1.57117 w_k1_FDX_s013
                                 + 1.20406 w_k1_IBM_s013
                                 + 1.64549 w_k1_INTC_s013
                                 + 1.36448 w_k1_KO_s013 + 1.11513 w_k1_MAT_s013
                                 + 1.39169 w_k1_MSFT_s013
                                 + 1.08273 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.826514 w_k1_ADM_s013
                                 + 0.901072 w_k1_AXP_s013
                                 + 1.02751 w_k1_DIS_s013
                                 + 0.91963 w_k1_FDX_s013 + 1.1514 w_k1_IBM_s013
                                 + 0.959344 w_k1_INTC_s013
                                 + 0.966756 w_k1_KO_s013
                                 + 0.693635 w_k1_MAT_s013
                                 + 0.825246 w_k1_MSFT_s013
                                 + 0.870014 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.952182 w_k1_ADM_s013
                                 + 0.970666 w_k1_AXP_s013
                                 + 0.723119 w_k1_DIS_s013
                                 + 0.786989 w_k1_FDX_s013
                                 + 1.11463 w_k1_IBM_s013
                                 + 0.928712 w_k1_INTC_s013
                                 + 0.847036 w_k1_KO_s013
                                 + 0.999175 w_k1_MAT_s013
                                 + 0.844179 w_k1_MSFT_s013
                                 + 0.791639 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.843197 w_k1_ADM_s013
                                 + 0.741032 w_k1_AXP_s013
                                 + 0.788457 w_k1_DIS_s013
                                 + 0.845768 w_k1_FDX_s013
                                 + 0.778167 w_k1_IBM_s013
                                 + 0.899286 w_k1_INTC_s013
                                 + 0.84857 w_k1_KO_s013
                                 + 0.756474 w_k1_MAT_s013
                                 + 0.935475 w_k1_MSFT_s013
                                 + 0.776766 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.872071 w_k1_ADM_s014
                                 + 1.21311 w_k1_AXP_s014
                                 + 1.03779 w_k1_DIS_s014
                                 + 0.988377 w_k1_FDX_s014
                                 + 1.30117 w_k1_IBM_s014
                                 + 1.14187 w_k1_INTC_s014
                                 + 1.07845 w_k1_KO_s014 + 1.10618 w_k1_MAT_s014
                                 + 0.964006 w_k1_MSFT_s014
                                 + 1.02362 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.945531 w_k1_ADM_s014
                                 + 1.08833 w_k1_AXP_s014
                                 + 1.52861 w_k1_DIS_s014
                                 + 1.19621 w_k1_FDX_s014
                                 + 1.01147 w_k1_IBM_s014
                                 + 1.08168 w_k1_INTC_s014
                                 + 1.26794 w_k1_KO_s014 + 1.13145 w_k1_MAT_s014
                                 + 1.21205 w_k1_MSFT_s014
                                 + 0.970596 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.632534 w_k1_ADM_s014
                                 + 0.641869 w_k1_AXP_s014
                                 + 0.576128 w_k1_DIS_s014
                                 + 0.720687 w_k1_FDX_s014
                                 + 0.842396 w_k1_IBM_s014
                                 + 0.768126 w_k1_INTC_s014
                                 + 0.642293 w_k1_KO_s014
                                 + 0.786664 w_k1_MAT_s014
                                 + 0.765016 w_k1_MSFT_s014
                                 + 0.808907 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.952178 w_k1_ADM_s014
                                 + 1.0622 w_k1_AXP_s014
                                 + 0.831915 w_k1_DIS_s014
                                 + 0.928371 w_k1_FDX_s014
                                 + 0.873061 w_k1_IBM_s014
                                 + 1.10672 w_k1_INTC_s014
                                 + 0.893312 w_k1_KO_s014 + 0.9764 w_k1_MAT_s014
                                 + 0.907947 w_k1_MSFT_s014
                                 + 0.890353 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.2269 w_k1_ADM_s014
                                 + 1.37813 w_k1_AXP_s014
                                 + 1.46406 w_k1_DIS_s014
                                 + 1.55489 w_k1_FDX_s014
                                 + 0.93407 w_k1_IBM_s014
                                 + 1.22314 w_k1_INTC_s014
                                 + 1.13166 w_k1_KO_s014 + 1.05953 w_k1_MAT_s014
                                 + 1.37452 w_k1_MSFT_s014
                                 + 1.05242 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.911314 w_k1_ADM_s014
                                 + 1.2667 w_k1_AXP_s014 + 1.70517 w_k1_DIS_s014
                                 + 1.34171 w_k1_FDX_s014
                                 + 1.42391 w_k1_IBM_s014
                                 + 1.33712 w_k1_INTC_s014
                                 + 1.35495 w_k1_KO_s014 + 1.2142 w_k1_MAT_s014
                                 + 1.35925 w_k1_MSFT_s014
                                 + 1.18339 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.897394 w_k1_ADM_s014
                                 + 0.943964 w_k1_AXP_s014
                                 + 1.01854 w_k1_DIS_s014
                                 + 1.10357 w_k1_FDX_s014
                                 + 0.698831 w_k1_IBM_s014
                                 + 1.22217 w_k1_INTC_s014
                                 + 0.85885 w_k1_KO_s014 + 0.62022 w_k1_MAT_s014
                                 + 1.13432 w_k1_MSFT_s014
                                 + 0.959018 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.590596 w_k1_ADM_s014
                                 + 0.639949 w_k1_AXP_s014
                                 + 0.841652 w_k1_DIS_s014
                                 + 0.703934 w_k1_FDX_s014
                                 + 1.26201 w_k1_IBM_s014
                                 + 0.713193 w_k1_INTC_s014
                                 + 1.03963 w_k1_KO_s014 + 1.01957 w_k1_MAT_s014
                                 + 0.752822 w_k1_MSFT_s014
                                 + 0.880365 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.39967 w_k1_ADM_s015
                                 + 1.46009 w_k1_AXP_s015
                                 + 1.30189 w_k1_DIS_s015
                                 + 1.44292 w_k1_FDX_s015
                                 + 0.883947 w_k1_IBM_s015
                                 + 1.32945 w_k1_INTC_s015
                                 + 1.28633 w_k1_KO_s015 + 0.9784 w_k1_MAT_s015
                                 + 1.23543 w_k1_MSFT_s015
                                 + 1.16992 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.00943 w_k1_ADM_s015
                                 + 1.20805 w_k1_AXP_s015
                                 + 0.977253 w_k1_DIS_s015
                                 + 0.964599 w_k1_FDX_s015
                                 + 1.36875 w_k1_IBM_s015
                                 + 1.05015 w_k1_INTC_s015
                                 + 1.15968 w_k1_KO_s015 + 1.22294 w_k1_MAT_s015
                                 + 0.886832 w_k1_MSFT_s015
                                 + 0.955696 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.14423 w_k1_ADM_s015
                                 + 1.0951 w_k1_AXP_s015 + 1.11922 w_k1_DIS_s015
                                 + 1.10815 w_k1_FDX_s015
                                 + 0.908084 w_k1_IBM_s015
                                 + 1.3416 w_k1_INTC_s015 + 1.05469 w_k1_KO_s015
                                 + 0.715871 w_k1_MAT_s015
                                 + 1.12282 w_k1_MSFT_s015
                                 + 0.957574 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.14251 w_k1_ADM_s015
                                 + 1.22917 w_k1_AXP_s015
                                 + 1.47697 w_k1_DIS_s015
                                 + 1.17802 w_k1_FDX_s015
                                 + 1.26453 w_k1_IBM_s015
                                 + 1.43327 w_k1_INTC_s015
                                 + 1.51335 w_k1_KO_s015 + 1.17924 w_k1_MAT_s015
                                 + 1.19357 w_k1_MSFT_s015
                                 + 1.21039 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.599194 w_k1_ADM_s015
                                 + 0.818706 w_k1_AXP_s015
                                 + 0.706486 w_k1_DIS_s015
                                 + 0.802764 w_k1_FDX_s015
                                 + 1.04722 w_k1_IBM_s015
                                 + 1.0566 w_k1_INTC_s015
                                 + 0.856165 w_k1_KO_s015
                                 + 1.08051 w_k1_MAT_s015
                                 + 0.851636 w_k1_MSFT_s015
                                 + 0.881176 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.965722 w_k1_ADM_s015
                                 + 0.932239 w_k1_AXP_s015
                                 + 0.860718 w_k1_DIS_s015
                                 + 0.992208 w_k1_FDX_s015
                                 + 0.775586 w_k1_IBM_s015
                                 + 0.982854 w_k1_INTC_s015
                                 + 0.873884 w_k1_KO_s015
                                 + 0.668893 w_k1_MAT_s015
                                 + 0.907991 w_k1_MSFT_s015
                                 + 0.955207 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.854139 w_k1_ADM_s015
                                 + 0.781313 w_k1_AXP_s015
                                 + 0.657233 w_k1_DIS_s015
                                 + 0.846208 w_k1_FDX_s015
                                 + 0.64046 w_k1_IBM_s015
                                 + 1.06477 w_k1_INTC_s015
                                 + 0.859266 w_k1_KO_s015
                                 + 0.486101 w_k1_MAT_s015
                                 + 0.795909 w_k1_MSFT_s015
                                 + 0.80999 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.20253 w_k1_ADM_s015
                                 + 1.24181 w_k1_AXP_s015
                                 + 1.37053 w_k1_DIS_s015
                                 + 1.00111 w_k1_FDX_s015 + 1.9944 w_k1_IBM_s015
                                 + 1.19757 w_k1_INTC_s015
                                 + 1.61528 w_k1_KO_s015 + 1.70824 w_k1_MAT_s015
                                 + 1.04772 w_k1_MSFT_s015
                                 + 1.21213 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.962281 w_k1_ADM_s016
                                 + 0.948896 w_k1_AXP_s016
                                 + 1.02268 w_k1_DIS_s016
                                 + 1.01122 w_k1_FDX_s016
                                 + 0.695116 w_k1_IBM_s016
                                 + 1.10975 w_k1_INTC_s016
                                 + 0.882985 w_k1_KO_s016
                                 + 0.735386 w_k1_MAT_s016
                                 + 0.937354 w_k1_MSFT_s016
                                 + 0.911565 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.836483 w_k1_ADM_s016
                                 + 0.864607 w_k1_AXP_s016
                                 + 0.99215 w_k1_DIS_s016
                                 + 0.931743 w_k1_FDX_s016
                                 + 0.982375 w_k1_IBM_s016
                                 + 1.00232 w_k1_INTC_s016
                                 + 0.943875 w_k1_KO_s016
                                 + 0.809434 w_k1_MAT_s016
                                 + 0.906903 w_k1_MSFT_s016
                                 + 0.898715 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.844574 w_k1_ADM_s016
                                 + 0.874488 w_k1_AXP_s016
                                 + 0.737987 w_k1_DIS_s016
                                 + 0.848683 w_k1_FDX_s016
                                 + 0.790407 w_k1_IBM_s016
                                 + 0.809066 w_k1_INTC_s016
                                 + 0.811708 w_k1_KO_s016
                                 + 1.03756 w_k1_MAT_s016
                                 + 0.931687 w_k1_MSFT_s016
                                 + 0.937084 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.910142 w_k1_ADM_s016
                                 + 0.487278 w_k1_AXP_s016
                                 + 0.610495 w_k1_DIS_s016
                                 + 0.643642 w_k1_FDX_s016
                                 + 0.854087 w_k1_IBM_s016
                                 + 0.617237 w_k1_INTC_s016
                                 + 0.778148 w_k1_KO_s016
                                 + 0.645133 w_k1_MAT_s016
                                 + 0.695969 w_k1_MSFT_s016
                                 + 0.857887 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.22635 w_k1_ADM_s016
                                 + 1.4609 w_k1_AXP_s016 + 1.95573 w_k1_DIS_s016
                                 + 1.52263 w_k1_FDX_s016
                                 + 1.44584 w_k1_IBM_s016
                                 + 1.11755 w_k1_INTC_s016
                                 + 1.46079 w_k1_KO_s016 + 1.83848 w_k1_MAT_s016
                                 + 1.38835 w_k1_MSFT_s016
                                 + 1.13995 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.08971 w_k1_ADM_s016
                                 + 1.19241 w_k1_AXP_s016
                                 + 1.32584 w_k1_DIS_s016
                                 + 1.20839 w_k1_FDX_s016
                                 + 1.07288 w_k1_IBM_s016
                                 + 1.16378 w_k1_INTC_s016
                                 + 1.12514 w_k1_KO_s016 + 1.26477 w_k1_MAT_s016
                                 + 1.14344 w_k1_MSFT_s016
                                 + 1.05176 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.16246 w_k1_ADM_s016
                                 + 0.911576 w_k1_AXP_s016
                                 + 1.11985 w_k1_DIS_s016
                                 + 0.901736 w_k1_FDX_s016
                                 + 1.42414 w_k1_IBM_s016
                                 + 0.781144 w_k1_INTC_s016
                                 + 1.11192 w_k1_KO_s016 + 1.58713 w_k1_MAT_s016
                                 + 0.970816 w_k1_MSFT_s016
                                 + 1.02861 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.26948 w_k1_ADM_s016
                                 + 1.64387 w_k1_AXP_s016
                                 + 1.54039 w_k1_DIS_s016 + 1.1471 w_k1_FDX_s016
                                 + 1.79952 w_k1_IBM_s016
                                 + 0.974781 w_k1_INTC_s016
                                 + 1.44719 w_k1_KO_s016 + 2.06669 w_k1_MAT_s016
                                 + 1.13626 w_k1_MSFT_s016
                                 + 1.13394 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.969851 w_k1_ADM_s017
                                 + 1.18026 w_k1_AXP_s017
                                 + 0.82794 w_k1_DIS_s017
                                 + 1.02769 w_k1_FDX_s017
                                 + 0.951311 w_k1_IBM_s017
                                 + 0.868047 w_k1_INTC_s017
                                 + 0.930233 w_k1_KO_s017
                                 + 1.20516 w_k1_MAT_s017
                                 + 1.06296 w_k1_MSFT_s017
                                 + 1.04345 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.707138 w_k1_ADM_s017
                                 + 0.741458 w_k1_AXP_s017
                                 + 0.671204 w_k1_DIS_s017
                                 + 0.800453 w_k1_FDX_s017
                                 + 0.755001 w_k1_IBM_s017
                                 + 0.847032 w_k1_INTC_s017
                                 + 0.824114 w_k1_KO_s017
                                 + 0.767502 w_k1_MAT_s017
                                 + 0.869667 w_k1_MSFT_s017
                                 + 0.835203 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.75434 w_k1_ADM_s017
                                 + 0.952308 w_k1_AXP_s017
                                 + 1.01202 w_k1_DIS_s017
                                 + 0.962522 w_k1_FDX_s017
                                 + 0.695516 w_k1_IBM_s017
                                 + 1.13532 w_k1_INTC_s017
                                 + 0.899644 w_k1_KO_s017
                                 + 0.701061 w_k1_MAT_s017
                                 + 1.08971 w_k1_MSFT_s017
                                 + 0.955795 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.04559 w_k1_ADM_s017
                                 + 1.04148 w_k1_AXP_s017
                                 + 1.14858 w_k1_DIS_s017
                                 + 0.993497 w_k1_FDX_s017
                                 + 1.12268 w_k1_IBM_s017
                                 + 0.986506 w_k1_INTC_s017
                                 + 1.17197 w_k1_KO_s017 + 1.30502 w_k1_MAT_s017
                                 + 1.07114 w_k1_MSFT_s017 + 1.09 w_k1_PG_s017
                                 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.845453 w_k1_ADM_s017
                                 + 0.839886 w_k1_AXP_s017
                                 + 1.13016 w_k1_DIS_s017
                                 + 1.02594 w_k1_FDX_s017
                                 + 0.998387 w_k1_IBM_s017
                                 + 1.10671 w_k1_INTC_s017
                                 + 1.12988 w_k1_KO_s017
                                 + 0.881902 w_k1_MAT_s017
                                 + 1.23391 w_k1_MSFT_s017
                                 + 1.03478 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.17486 w_k1_ADM_s017
                                 + 1.67361 w_k1_AXP_s017
                                 + 1.40757 w_k1_DIS_s017
                                 + 1.35089 w_k1_FDX_s017
                                 + 1.04203 w_k1_IBM_s017
                                 + 1.18816 w_k1_INTC_s017 + 1.1248 w_k1_KO_s017
                                 + 1.39148 w_k1_MAT_s017
                                 + 1.29602 w_k1_MSFT_s017
                                 + 1.07586 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.699565 w_k1_ADM_s017
                                 + 1.19848 w_k1_AXP_s017
                                 + 1.11705 w_k1_DIS_s017
                                 + 1.05803 w_k1_FDX_s017
                                 + 1.36124 w_k1_IBM_s017
                                 + 1.16527 w_k1_INTC_s017
                                 + 1.36599 w_k1_KO_s017 + 1.66382 w_k1_MAT_s017
                                 + 1.19615 w_k1_MSFT_s017
                                 + 1.09995 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.18362 w_k1_ADM_s017
                                 + 1.44769 w_k1_AXP_s017
                                 + 1.17484 w_k1_DIS_s017
                                 + 0.930019 w_k1_FDX_s017
                                 + 1.74138 w_k1_IBM_s017
                                 + 1.22908 w_k1_INTC_s017
                                 + 1.44792 w_k1_KO_s017 + 1.9142 w_k1_MAT_s017
                                 + 1.17446 w_k1_MSFT_s017
                                 + 1.18388 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.898035 w_k1_ADM_s100
                                 + 1.13862 w_k1_AXP_s100
                                 + 1.21239 w_k1_DIS_s100
                                 + 1.14802 w_k1_FDX_s100
                                 + 1.06012 w_k1_IBM_s100
                                 + 1.13408 w_k1_INTC_s100
                                 + 0.91296 w_k1_KO_s100 + 1.34498 w_k1_MAT_s100
                                 + 1.16756 w_k1_MSFT_s100
                                 + 1.10448 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.951205 w_k1_ADM_s100
                                 + 1.41251 w_k1_AXP_s100
                                 + 1.53407 w_k1_DIS_s100
                                 + 1.45375 w_k1_FDX_s100
                                 + 1.18343 w_k1_IBM_s100
                                 + 1.33128 w_k1_INTC_s100
                                 + 1.09099 w_k1_KO_s100 + 1.46509 w_k1_MAT_s100
                                 + 1.39565 w_k1_MSFT_s100
                                 + 1.27771 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.803848 w_k1_ADM_s100
                                 + 0.877274 w_k1_AXP_s100
                                 + 0.726321 w_k1_DIS_s100
                                 + 0.935118 w_k1_FDX_s100
                                 + 0.794849 w_k1_IBM_s100
                                 + 0.932653 w_k1_INTC_s100
                                 + 0.698122 w_k1_KO_s100
                                 + 0.97065 w_k1_MAT_s100
                                 + 0.857563 w_k1_MSFT_s100
                                 + 0.859564 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.781047 w_k1_ADM_s100
                                 + 0.694352 w_k1_AXP_s100
                                 + 0.728924 w_k1_DIS_s100
                                 + 0.80065 w_k1_FDX_s100
                                 + 0.759567 w_k1_IBM_s100
                                 + 0.863137 w_k1_INTC_s100
                                 + 0.730632 w_k1_KO_s100
                                 + 0.768825 w_k1_MAT_s100
                                 + 0.811674 w_k1_MSFT_s100
                                 + 0.761141 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.83153 w_k1_ADM_s100
                                 + 1.30449 w_k1_AXP_s100
                                 + 0.990604 w_k1_DIS_s100
                                 + 1.09268 w_k1_FDX_s100
                                 + 1.22534 w_k1_IBM_s100
                                 + 0.966175 w_k1_INTC_s100
                                 + 0.857892 w_k1_KO_s100 + 1.4674 w_k1_MAT_s100
                                 + 0.99876 w_k1_MSFT_s100
                                 + 0.968347 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.11657 w_k1_ADM_s100
                                 + 1.28847 w_k1_AXP_s100
                                 + 1.24362 w_k1_DIS_s100
                                 + 1.11483 w_k1_FDX_s100
                                 + 1.73761 w_k1_IBM_s100
                                 + 1.22345 w_k1_INTC_s100
                                 + 1.34372 w_k1_KO_s100 + 1.70124 w_k1_MAT_s100
                                 + 1.02029 w_k1_MSFT_s100
                                 + 1.15831 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.801417 w_k1_ADM_s100
                                 + 1.03353 w_k1_AXP_s100
                                 + 0.895488 w_k1_DIS_s100
                                 + 0.919571 w_k1_FDX_s100
                                 + 1.17298 w_k1_IBM_s100
                                 + 1.13224 w_k1_INTC_s100
                                 + 0.98657 w_k1_KO_s100 + 1.01521 w_k1_MAT_s100
                                 + 0.887304 w_k1_MSFT_s100
                                 + 0.977838 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.833849 w_k1_ADM_s100
                                 + 0.922292 w_k1_AXP_s100
                                 + 1.12051 w_k1_DIS_s100
                                 + 1.11527 w_k1_FDX_s100
                                 + 0.814754 w_k1_IBM_s100
                                 + 1.10169 w_k1_INTC_s100
                                 + 0.945693 w_k1_KO_s100
                                 + 0.694542 w_k1_MAT_s100
                                 + 1.30044 w_k1_MSFT_s100
                                 + 1.03251 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.12084 w_k1_ADM_s101
                                 + 1.07235 w_k1_AXP_s101
                                 + 0.853417 w_k1_DIS_s101
                                 + 1.03278 w_k1_FDX_s101
                                 + 0.824134 w_k1_IBM_s101
                                 + 1.04372 w_k1_INTC_s101
                                 + 0.812818 w_k1_KO_s101
                                 + 1.10089 w_k1_MAT_s101
                                 + 1.03989 w_k1_MSFT_s101
                                 + 1.02761 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.716469 w_k1_ADM_s101
                                 + 0.682283 w_k1_AXP_s101
                                 + 0.635236 w_k1_DIS_s101
                                 + 0.739101 w_k1_FDX_s101
                                 + 0.757128 w_k1_IBM_s101
                                 + 0.877543 w_k1_INTC_s101
                                 + 0.749071 w_k1_KO_s101
                                 + 0.674094 w_k1_MAT_s101
                                 + 0.86274 w_k1_MSFT_s101
                                 + 0.818587 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.960873 w_k1_ADM_s101
                                 + 1.4027 w_k1_AXP_s101 + 1.79596 w_k1_DIS_s101
                                 + 1.35596 w_k1_FDX_s101
                                 + 1.47192 w_k1_IBM_s101
                                 + 1.60294 w_k1_INTC_s101
                                 + 1.34773 w_k1_KO_s101 + 1.68337 w_k1_MAT_s101
                                 + 1.28347 w_k1_MSFT_s101
                                 + 1.22982 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.01398 w_k1_ADM_s101
                                 + 0.929126 w_k1_AXP_s101
                                 + 1.33891 w_k1_DIS_s101
                                 + 1.25698 w_k1_FDX_s101
                                 + 0.898794 w_k1_IBM_s101
                                 + 1.45975 w_k1_INTC_s101
                                 + 1.01263 w_k1_KO_s101
                                 + 0.727767 w_k1_MAT_s101
                                 + 1.32575 w_k1_MSFT_s101
                                 + 1.08935 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.12538 w_k1_ADM_s101
                                 + 1.08288 w_k1_AXP_s101
                                 + 1.05558 w_k1_DIS_s101
                                 + 1.07546 w_k1_FDX_s101
                                 + 1.23288 w_k1_IBM_s101
                                 + 1.20193 w_k1_INTC_s101
                                 + 0.928783 w_k1_KO_s101
                                 + 1.08473 w_k1_MAT_s101
                                 + 1.08684 w_k1_MSFT_s101 + 1.1157 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.789365 w_k1_ADM_s101
                                 + 0.685323 w_k1_AXP_s101
                                 + 0.842909 w_k1_DIS_s101
                                 + 0.926439 w_k1_FDX_s101
                                 + 0.87771 w_k1_IBM_s101
                                 + 0.98471 w_k1_INTC_s101
                                 + 0.824823 w_k1_KO_s101
                                 + 0.75436 w_k1_MAT_s101
                                 + 0.932939 w_k1_MSFT_s101
                                 + 0.880872 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.662606 w_k1_ADM_s101
                                 + 0.887883 w_k1_AXP_s101
                                 + 0.927196 w_k1_DIS_s101
                                 + 0.922892 w_k1_FDX_s101
                                 + 1.13779 w_k1_IBM_s101
                                 + 0.906834 w_k1_INTC_s101
                                 + 0.835741 w_k1_KO_s101 + 1.2124 w_k1_MAT_s101
                                 + 0.931157 w_k1_MSFT_s101
                                 + 0.953298 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.738452 w_k1_ADM_s101
                                 + 0.690517 w_k1_AXP_s101
                                 + 0.673941 w_k1_DIS_s101
                                 + 0.733858 w_k1_FDX_s101
                                 + 1.42003 w_k1_IBM_s101
                                 + 0.789586 w_k1_INTC_s101
                                 + 0.925018 w_k1_KO_s101
                                 + 1.12965 w_k1_MAT_s101
                                 + 0.784148 w_k1_MSFT_s101
                                 + 0.979673 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.34062 w_k1_ADM_s102
                                 + 1.27567 w_k1_AXP_s102
                                 + 0.896336 w_k1_DIS_s102
                                 + 1.12345 w_k1_FDX_s102
                                 + 0.865679 w_k1_IBM_s102
                                 + 0.998845 w_k1_INTC_s102
                                 + 0.844561 w_k1_KO_s102
                                 + 0.847006 w_k1_MAT_s102
                                 + 0.915748 w_k1_MSFT_s102
                                 + 1.00569 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.831055 w_k1_ADM_s102
                                 + 0.783335 w_k1_AXP_s102
                                 + 0.615609 w_k1_DIS_s102
                                 + 0.798852 w_k1_FDX_s102
                                 + 0.801326 w_k1_IBM_s102
                                 + 0.844036 w_k1_INTC_s102
                                 + 0.783261 w_k1_KO_s102
                                 + 0.739153 w_k1_MAT_s102
                                 + 0.782519 w_k1_MSFT_s102
                                 + 0.771176 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.834552 w_k1_ADM_s102
                                 + 0.630645 w_k1_AXP_s102
                                 + 1.06207 w_k1_DIS_s102
                                 + 0.913268 w_k1_FDX_s102
                                 + 0.948022 w_k1_IBM_s102
                                 + 0.946364 w_k1_INTC_s102 + 1.061 w_k1_KO_s102
                                 + 0.893653 w_k1_MAT_s102
                                 + 0.963006 w_k1_MSFT_s102
                                 + 0.956996 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.970214 w_k1_ADM_s102
                                 + 0.977414 w_k1_AXP_s102
                                 + 1.06431 w_k1_DIS_s102
                                 + 1.04223 w_k1_FDX_s102
                                 + 0.940612 w_k1_IBM_s102
                                 + 1.08492 w_k1_INTC_s102
                                 + 1.04466 w_k1_KO_s102
                                 + 0.910634 w_k1_MAT_s102
                                 + 1.04943 w_k1_MSFT_s102
                                 + 1.04455 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.01098 w_k1_ADM_s102
                                 + 1.44129 w_k1_AXP_s102
                                 + 0.910586 w_k1_DIS_s102
                                 + 0.989767 w_k1_FDX_s102
                                 + 1.34165 w_k1_IBM_s102
                                 + 1.27102 w_k1_INTC_s102
                                 + 1.18898 w_k1_KO_s102 + 1.26684 w_k1_MAT_s102
                                 + 0.906243 w_k1_MSFT_s102
                                 + 1.01649 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.898177 w_k1_ADM_s102
                                 + 0.837266 w_k1_AXP_s102
                                 + 0.783308 w_k1_DIS_s102
                                 + 1.01061 w_k1_FDX_s102
                                 + 0.567358 w_k1_IBM_s102
                                 + 1.04542 w_k1_INTC_s102
                                 + 0.80869 w_k1_KO_s102
                                 + 0.555432 w_k1_MAT_s102
                                 + 0.921175 w_k1_MSFT_s102
                                 + 0.809884 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.22087 w_k1_ADM_s102
                                 + 1.11641 w_k1_AXP_s102
                                 + 1.51336 w_k1_DIS_s102
                                 + 1.19151 w_k1_FDX_s102
                                 + 1.18292 w_k1_IBM_s102
                                 + 1.13015 w_k1_INTC_s102
                                 + 1.27823 w_k1_KO_s102
                                 + 0.851006 w_k1_MAT_s102
                                 + 1.16072 w_k1_MSFT_s102
                                 + 1.11286 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.38613 w_k1_ADM_s102
                                 + 1.30788 w_k1_AXP_s102
                                 + 1.74381 w_k1_DIS_s102
                                 + 1.46856 w_k1_FDX_s102
                                 + 0.902806 w_k1_IBM_s102
                                 + 1.36802 w_k1_INTC_s102
                                 + 1.21603 w_k1_KO_s102
                                 + 0.602468 w_k1_MAT_s102
                                 + 1.38591 w_k1_MSFT_s102
                                 + 1.26321 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.811065 w_k1_ADM_s103
                                 + 1.02303 w_k1_AXP_s103
                                 + 0.870661 w_k1_DIS_s103
                                 + 0.761557 w_k1_FDX_s103
                                 + 1.41502 w_k1_IBM_s103
                                 + 1.00161 w_k1_INTC_s103
                                 + 1.09388 w_k1_KO_s103 + 0.93481 w_k1_MAT_s103
                                 + 0.899561 w_k1_MSFT_s103
                                 + 1.00317 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.773739 w_k1_ADM_s103
                                 + 0.867245 w_k1_AXP_s103
                                 + 0.63677 w_k1_DIS_s103
                                 + 0.802325 w_k1_FDX_s103
                                 + 1.07163 w_k1_IBM_s103
                                 + 1.03262 w_k1_INTC_s103
                                 + 0.896496 w_k1_KO_s103
                                 + 0.903807 w_k1_MAT_s103
                                 + 0.814242 w_k1_MSFT_s103
                                 + 0.862536 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.873885 w_k1_ADM_s103
                                 + 1.01262 w_k1_AXP_s103
                                 + 0.885708 w_k1_DIS_s103
                                 + 1.01885 w_k1_FDX_s103
                                 + 0.813163 w_k1_IBM_s103
                                 + 1.2015 w_k1_INTC_s103
                                 + 0.936528 w_k1_KO_s103
                                 + 0.665283 w_k1_MAT_s103
                                 + 1.12897 w_k1_MSFT_s103
                                 + 0.948447 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.0233 w_k1_ADM_s103
                                 + 1.20666 w_k1_AXP_s103
                                 + 1.44541 w_k1_DIS_s103
                                 + 1.27988 w_k1_FDX_s103
                                 + 1.06023 w_k1_IBM_s103
                                 + 1.16272 w_k1_INTC_s103
                                 + 1.21649 w_k1_KO_s103
                                 + 0.962716 w_k1_MAT_s103
                                 + 1.3874 w_k1_MSFT_s103 + 1.09686 w_k1_PG_s103
                                 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.823133 w_k1_ADM_s103
                                 + 0.978269 w_k1_AXP_s103
                                 + 0.976232 w_k1_DIS_s103
                                 + 0.954876 w_k1_FDX_s103
                                 + 1.12218 w_k1_IBM_s103
                                 + 1.30809 w_k1_INTC_s103
                                 + 0.979821 w_k1_KO_s103
                                 + 0.825128 w_k1_MAT_s103
                                 + 1.03089 w_k1_MSFT_s103
                                 + 0.952324 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.18265 w_k1_ADM_s103
                                 + 1.19401 w_k1_AXP_s103 + 1.3881 w_k1_DIS_s103
                                 + 1.1461 w_k1_FDX_s103 + 1.59005 w_k1_IBM_s103
                                 + 1.20629 w_k1_INTC_s103
                                 + 1.46903 w_k1_KO_s103 + 1.69646 w_k1_MAT_s103
                                 + 1.14543 w_k1_MSFT_s103
                                 + 1.21508 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.979854 w_k1_ADM_s103
                                 + 1.44503 w_k1_AXP_s103
                                 + 1.14423 w_k1_DIS_s103
                                 + 1.08642 w_k1_FDX_s103
                                 + 1.11871 w_k1_IBM_s103
                                 + 1.01986 w_k1_INTC_s103
                                 + 1.20026 w_k1_KO_s103 + 1.32923 w_k1_MAT_s103
                                 + 1.07003 w_k1_MSFT_s103
                                 + 1.10035 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.743411 w_k1_ADM_s103
                                 + 0.675717 w_k1_AXP_s103
                                 + 0.621776 w_k1_DIS_s103
                                 + 0.816769 w_k1_FDX_s103
                                 + 0.694999 w_k1_IBM_s103
                                 + 0.963762 w_k1_INTC_s103
                                 + 0.74421 w_k1_KO_s103
                                 + 0.525344 w_k1_MAT_s103
                                 + 0.852612 w_k1_MSFT_s103
                                 + 0.759664 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.848878 w_k1_ADM_s104
                                 + 0.764933 w_k1_AXP_s104
                                 + 0.69755 w_k1_DIS_s104
                                 + 0.86527 w_k1_FDX_s104
                                 + 0.674901 w_k1_IBM_s104
                                 + 0.705911 w_k1_INTC_s104
                                 + 0.629686 w_k1_KO_s104
                                 + 0.708028 w_k1_MAT_s104
                                 + 0.827144 w_k1_MSFT_s104
                                 + 0.851395 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.22416 w_k1_ADM_s104
                                 + 1.41474 w_k1_AXP_s104
                                 + 1.48558 w_k1_DIS_s104
                                 + 1.41674 w_k1_FDX_s104
                                 + 1.16155 w_k1_IBM_s104
                                 + 1.49308 w_k1_INTC_s104
                                 + 1.30822 w_k1_KO_s104 + 1.31781 w_k1_MAT_s104
                                 + 1.30518 w_k1_MSFT_s104
                                 + 1.18694 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.1591 w_k1_ADM_s104
                                 + 1.02021 w_k1_AXP_s104
                                 + 1.34985 w_k1_DIS_s104 + 1.0812 w_k1_FDX_s104
                                 + 1.40515 w_k1_IBM_s104
                                 + 0.95855 w_k1_INTC_s104
                                 + 1.38963 w_k1_KO_s104 + 1.24582 w_k1_MAT_s104
                                 + 1.09574 w_k1_MSFT_s104
                                 + 1.04527 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.999448 w_k1_ADM_s104
                                 + 0.93788 w_k1_AXP_s104
                                 + 0.944772 w_k1_DIS_s104
                                 + 0.907518 w_k1_FDX_s104
                                 + 1.29072 w_k1_IBM_s104
                                 + 0.954525 w_k1_INTC_s104
                                 + 1.16303 w_k1_KO_s104 + 1.22029 w_k1_MAT_s104
                                 + 0.917549 w_k1_MSFT_s104
                                 + 1.06894 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.80453 w_k1_ADM_s104
                                 + 0.714354 w_k1_AXP_s104
                                 + 0.718704 w_k1_DIS_s104
                                 + 0.790607 w_k1_FDX_s104
                                 + 0.976333 w_k1_IBM_s104
                                 + 0.86165 w_k1_INTC_s104
                                 + 0.856155 w_k1_KO_s104
                                 + 0.888139 w_k1_MAT_s104
                                 + 0.878504 w_k1_MSFT_s104
                                 + 0.8646 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.18472 w_k1_ADM_s104
                                 + 1.2035 w_k1_AXP_s104 + 1.20893 w_k1_DIS_s104
                                 + 1.343 w_k1_FDX_s104 + 0.840464 w_k1_IBM_s104
                                 + 1.07803 w_k1_INTC_s104
                                 + 0.987461 w_k1_KO_s104
                                 + 1.13516 w_k1_MAT_s104
                                 + 1.07994 w_k1_MSFT_s104 + 1.1661 w_k1_PG_s104
                                 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.832237 w_k1_ADM_s104
                                 + 1.21448 w_k1_AXP_s104
                                 + 1.08008 w_k1_DIS_s104
                                 + 1.09219 w_k1_FDX_s104 + 1.1232 w_k1_IBM_s104
                                 + 1.15754 w_k1_INTC_s104
                                 + 1.04172 w_k1_KO_s104 + 1.39403 w_k1_MAT_s104
                                 + 0.972839 w_k1_MSFT_s104
                                 + 1.04812 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.970303 w_k1_ADM_s104
                                 + 0.978208 w_k1_AXP_s104
                                 + 0.995387 w_k1_DIS_s104
                                 + 1.04988 w_k1_FDX_s104
                                 + 0.857894 w_k1_IBM_s104
                                 + 0.921856 w_k1_INTC_s104
                                 + 0.942201 w_k1_KO_s104
                                 + 0.976817 w_k1_MAT_s104
                                 + 1.05857 w_k1_MSFT_s104 + 1.0014 w_k1_PG_s104
                                 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.818336 w_k1_ADM_s105
                                 + 1.03798 w_k1_AXP_s105
                                 + 1.06607 w_k1_DIS_s105
                                 + 1.08849 w_k1_FDX_s105
                                 + 1.04698 w_k1_IBM_s105
                                 + 0.977941 w_k1_INTC_s105
                                 + 1.08511 w_k1_KO_s105 + 1.21018 w_k1_MAT_s105
                                 + 1.17288 w_k1_MSFT_s105
                                 + 1.01822 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.689273 w_k1_ADM_s105
                                 + 0.602951 w_k1_AXP_s105
                                 + 0.599824 w_k1_DIS_s105
                                 + 0.709462 w_k1_FDX_s105
                                 + 0.807811 w_k1_IBM_s105
                                 + 0.630328 w_k1_INTC_s105
                                 + 0.612494 w_k1_KO_s105
                                 + 0.629247 w_k1_MAT_s105
                                 + 0.741159 w_k1_MSFT_s105
                                 + 0.798595 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.860462 w_k1_ADM_s105
                                 + 0.785623 w_k1_AXP_s105
                                 + 0.823278 w_k1_DIS_s105
                                 + 0.988739 w_k1_FDX_s105
                                 + 0.60712 w_k1_IBM_s105
                                 + 0.833682 w_k1_INTC_s105
                                 + 0.807717 w_k1_KO_s105
                                 + 0.444635 w_k1_MAT_s105
                                 + 0.884062 w_k1_MSFT_s105
                                 + 0.892479 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.15287 w_k1_ADM_s105
                                 + 1.53834 w_k1_AXP_s105
                                 + 1.32343 w_k1_DIS_s105
                                 + 1.19664 w_k1_FDX_s105
                                 + 1.34394 w_k1_IBM_s105
                                 + 1.02898 w_k1_INTC_s105
                                 + 1.23452 w_k1_KO_s105 + 1.44828 w_k1_MAT_s105
                                 + 1.10717 w_k1_MSFT_s105
                                 + 1.14661 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.947514 w_k1_ADM_s105
                                 + 0.951931 w_k1_AXP_s105
                                 + 1.16945 w_k1_DIS_s105
                                 + 1.06425 w_k1_FDX_s105
                                 + 0.838423 w_k1_IBM_s105
                                 + 0.757609 w_k1_INTC_s105
                                 + 0.903737 w_k1_KO_s105
                                 + 0.941409 w_k1_MAT_s105
                                 + 0.953993 w_k1_MSFT_s105
                                 + 1.10633 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.86384 w_k1_ADM_s105
                                 + 0.892126 w_k1_AXP_s105
                                 + 0.813782 w_k1_DIS_s105
                                 + 0.866342 w_k1_FDX_s105
                                 + 0.983813 w_k1_IBM_s105
                                 + 0.722178 w_k1_INTC_s105
                                 + 0.87302 w_k1_KO_s105 + 1.24737 w_k1_MAT_s105
                                 + 0.82978 w_k1_MSFT_s105
                                 + 0.934315 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.27126 w_k1_ADM_s105
                                 + 1.44575 w_k1_AXP_s105
                                 + 1.88379 w_k1_DIS_s105
                                 + 1.51659 w_k1_FDX_s105
                                 + 1.12358 w_k1_IBM_s105
                                 + 1.11798 w_k1_INTC_s105
                                 + 1.37694 w_k1_KO_s105 + 1.52925 w_k1_MAT_s105
                                 + 1.3801 w_k1_MSFT_s105 + 1.21111 w_k1_PG_s105
                                 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.805697 w_k1_ADM_s105
                                 + 0.866602 w_k1_AXP_s105
                                 + 0.983835 w_k1_DIS_s105
                                 + 0.905952 w_k1_FDX_s105
                                 + 1.42869 w_k1_IBM_s105
                                 + 0.70008 w_k1_INTC_s105
                                 + 1.11566 w_k1_KO_s105 + 1.54207 w_k1_MAT_s105
                                 + 0.860369 w_k1_MSFT_s105
                                 + 1.00951 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.714179 w_k1_ADM_s106
                                 + 0.711842 w_k1_AXP_s106
                                 + 0.601514 w_k1_DIS_s106
                                 + 0.717327 w_k1_FDX_s106
                                 + 0.635806 w_k1_IBM_s106
                                 + 0.820387 w_k1_INTC_s106
                                 + 0.655253 w_k1_KO_s106
                                 + 0.518259 w_k1_MAT_s106
                                 + 0.690014 w_k1_MSFT_s106
                                 + 0.771342 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.19792 w_k1_ADM_s106
                                 + 0.996383 w_k1_AXP_s106
                                 + 1.20627 w_k1_DIS_s106
                                 + 1.14232 w_k1_FDX_s106
                                 + 0.82052 w_k1_IBM_s106
                                 + 0.966693 w_k1_INTC_s106
                                 + 1.07794 w_k1_KO_s106
                                 + 0.759215 w_k1_MAT_s106
                                 + 1.06704 w_k1_MSFT_s106
                                 + 1.00418 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.07414 w_k1_ADM_s106
                                 + 0.78064 w_k1_AXP_s106
                                 + 1.51819 w_k1_DIS_s106
                                 + 1.07899 w_k1_FDX_s106
                                 + 1.17267 w_k1_IBM_s106
                                 + 1.13064 w_k1_INTC_s106
                                 + 1.48716 w_k1_KO_s106 + 1.05758 w_k1_MAT_s106
                                 + 1.15555 w_k1_MSFT_s106
                                 + 0.945536 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.991126 w_k1_ADM_s106
                                 + 0.990596 w_k1_AXP_s106
                                 + 0.982465 w_k1_DIS_s106
                                 + 0.981141 w_k1_FDX_s106
                                 + 1.17055 w_k1_IBM_s106
                                 + 0.946476 w_k1_INTC_s106
                                 + 1.18986 w_k1_KO_s106 + 1.25578 w_k1_MAT_s106
                                 + 1.06729 w_k1_MSFT_s106
                                 + 0.99869 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.30914 w_k1_ADM_s106
                                 + 1.43576 w_k1_AXP_s106
                                 + 1.41926 w_k1_DIS_s106
                                 + 1.18719 w_k1_FDX_s106
                                 + 1.42349 w_k1_IBM_s106
                                 + 1.30571 w_k1_INTC_s106
                                 + 1.44135 w_k1_KO_s106 + 1.45121 w_k1_MAT_s106
                                 + 1.19803 w_k1_MSFT_s106
                                 + 1.20038 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.917823 w_k1_ADM_s106
                                 + 0.944983 w_k1_AXP_s106
                                 + 0.896263 w_k1_DIS_s106
                                 + 0.833644 w_k1_FDX_s106
                                 + 1.25142 w_k1_IBM_s106
                                 + 0.996396 w_k1_INTC_s106
                                 + 1.18056 w_k1_KO_s106
                                 + 0.967987 w_k1_MAT_s106
                                 + 0.819278 w_k1_MSFT_s106
                                 + 0.921988 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.43208 w_k1_ADM_s106
                                 + 1.14666 w_k1_AXP_s106
                                 + 1.93509 w_k1_DIS_s106
                                 + 1.41926 w_k1_FDX_s106
                                 + 1.26843 w_k1_IBM_s106
                                 + 1.42285 w_k1_INTC_s106
                                 + 1.83784 w_k1_KO_s106 + 1.26971 w_k1_MAT_s106
                                 + 1.55294 w_k1_MSFT_s106
                                 + 1.24706 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.15818 w_k1_ADM_s106
                                 + 0.939416 w_k1_AXP_s106
                                 + 0.967732 w_k1_DIS_s106
                                 + 0.975631 w_k1_FDX_s106
                                 + 0.820761 w_k1_IBM_s106
                                 + 1.00305 w_k1_INTC_s106
                                 + 0.987511 w_k1_KO_s106
                                 + 0.56084 w_k1_MAT_s106
                                 + 0.962263 w_k1_MSFT_s106
                                 + 0.882659 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.940406 w_k1_ADM_s107
                                 + 0.801597 w_k1_AXP_s107
                                 + 0.828809 w_k1_DIS_s107
                                 + 0.911528 w_k1_FDX_s107
                                 + 0.734807 w_k1_IBM_s107
                                 + 0.813561 w_k1_INTC_s107
                                 + 0.856534 w_k1_KO_s107
                                 + 0.669623 w_k1_MAT_s107
                                 + 0.835493 w_k1_MSFT_s107
                                 + 0.955393 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.988129 w_k1_ADM_s107
                                 + 0.934807 w_k1_AXP_s107
                                 + 0.626039 w_k1_DIS_s107
                                 + 0.738313 w_k1_FDX_s107
                                 + 0.976175 w_k1_IBM_s107
                                 + 0.887344 w_k1_INTC_s107
                                 + 0.991151 w_k1_KO_s107
                                 + 1.18493 w_k1_MAT_s107
                                 + 0.798701 w_k1_MSFT_s107
                                 + 0.889846 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.13815 w_k1_ADM_s107
                                 + 1.1091 w_k1_AXP_s107 + 1.75441 w_k1_DIS_s107
                                 + 1.32638 w_k1_FDX_s107
                                 + 1.19811 w_k1_IBM_s107
                                 + 1.27222 w_k1_INTC_s107
                                 + 1.39167 w_k1_KO_s107
                                 + 0.970962 w_k1_MAT_s107
                                 + 1.34403 w_k1_MSFT_s107
                                 + 1.17753 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.835708 w_k1_ADM_s107
                                 + 1.28836 w_k1_AXP_s107
                                 + 1.30362 w_k1_DIS_s107
                                 + 1.18373 w_k1_FDX_s107
                                 + 1.48743 w_k1_IBM_s107
                                 + 1.20237 w_k1_INTC_s107
                                 + 1.17471 w_k1_KO_s107 + 1.56435 w_k1_MAT_s107
                                 + 1.04447 w_k1_MSFT_s107
                                 + 1.10541 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.764587 w_k1_ADM_s107
                                 + 0.781168 w_k1_AXP_s107
                                 + 0.950275 w_k1_DIS_s107
                                 + 0.910785 w_k1_FDX_s107
                                 + 1.20411 w_k1_IBM_s107
                                 + 0.976443 w_k1_INTC_s107
                                 + 1.10148 w_k1_KO_s107 + 1.0789 w_k1_MAT_s107
                                 + 0.924744 w_k1_MSFT_s107
                                 + 0.88231 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.894546 w_k1_ADM_s107
                                 + 1.04653 w_k1_AXP_s107 + 1.0316 w_k1_DIS_s107
                                 + 1.01359 w_k1_FDX_s107
                                 + 0.968147 w_k1_IBM_s107
                                 + 1.14725 w_k1_INTC_s107 + 1.0988 w_k1_KO_s107
                                 + 1.08681 w_k1_MAT_s107
                                 + 0.979683 w_k1_MSFT_s107
                                 + 1.01937 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.37685 w_k1_ADM_s107
                                 + 1.33174 w_k1_AXP_s107
                                 + 1.25231 w_k1_DIS_s107
                                 + 1.24959 w_k1_FDX_s107
                                 + 0.890531 w_k1_IBM_s107
                                 + 1.24959 w_k1_INTC_s107
                                 + 1.11637 w_k1_KO_s107
                                 + 0.842504 w_k1_MAT_s107
                                 + 1.08721 w_k1_MSFT_s107
                                 + 1.02282 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.00702 w_k1_ADM_s107
                                 + 1.33603 w_k1_AXP_s107
                                 + 0.758672 w_k1_DIS_s107
                                 + 0.935509 w_k1_FDX_s107
                                 + 0.904892 w_k1_IBM_s107
                                 + 1.06966 w_k1_INTC_s107
                                 + 0.882399 w_k1_KO_s107
                                 + 1.18552 w_k1_MAT_s107
                                 + 0.924973 w_k1_MSFT_s107
                                 + 1.01113 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.07814 w_k1_ADM_s110
                                 + 0.97685 w_k1_AXP_s110
                                 + 0.959568 w_k1_DIS_s110
                                 + 0.908198 w_k1_FDX_s110
                                 + 0.956756 w_k1_IBM_s110
                                 + 0.93045 w_k1_INTC_s110
                                 + 0.863027 w_k1_KO_s110
                                 + 1.15334 w_k1_MAT_s110
                                 + 0.943223 w_k1_MSFT_s110
                                 + 0.976995 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.81932 w_k1_ADM_s110
                                 + 0.733725 w_k1_AXP_s110
                                 + 0.638601 w_k1_DIS_s110
                                 + 0.63079 w_k1_FDX_s110
                                 + 0.714598 w_k1_IBM_s110
                                 + 0.71603 w_k1_INTC_s110
                                 + 0.67435 w_k1_KO_s110
                                 + 0.705755 w_k1_MAT_s110
                                 + 0.736468 w_k1_MSFT_s110
                                 + 0.73732 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.919552 w_k1_ADM_s110
                                 + 0.991241 w_k1_AXP_s110
                                 + 1.22153 w_k1_DIS_s110
                                 + 0.787099 w_k1_FDX_s110
                                 + 1.83533 w_k1_IBM_s110
                                 + 1.15466 w_k1_INTC_s110
                                 + 1.32543 w_k1_KO_s110 + 1.58606 w_k1_MAT_s110
                                 + 1.16153 w_k1_MSFT_s110
                                 + 1.08188 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.23602 w_k1_ADM_s110
                                 + 1.41988 w_k1_AXP_s110
                                 + 1.66601 w_k1_DIS_s110
                                 + 1.16447 w_k1_FDX_s110 + 1.5617 w_k1_IBM_s110
                                 + 1.19219 w_k1_INTC_s110
                                 + 1.31332 w_k1_KO_s110 + 1.5451 w_k1_MAT_s110
                                 + 1.36591 w_k1_MSFT_s110
                                 + 1.15935 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.02733 w_k1_ADM_s110
                                 + 0.739408 w_k1_AXP_s110
                                 + 0.765401 w_k1_DIS_s110
                                 + 0.839629 w_k1_FDX_s110
                                 + 0.585284 w_k1_IBM_s110
                                 + 0.96819 w_k1_INTC_s110
                                 + 0.765246 w_k1_KO_s110
                                 + 0.580452 w_k1_MAT_s110
                                 + 0.960373 w_k1_MSFT_s110
                                 + 0.730559 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.18464 w_k1_ADM_s110
                                 + 1.21227 w_k1_AXP_s110
                                 + 1.50723 w_k1_DIS_s110
                                 + 1.26356 w_k1_FDX_s110
                                 + 0.789712 w_k1_IBM_s110
                                 + 1.14349 w_k1_INTC_s110
                                 + 0.935987 w_k1_KO_s110
                                 + 1.03184 w_k1_MAT_s110
                                 + 1.24266 w_k1_MSFT_s110
                                 + 1.11166 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.955327 w_k1_ADM_s110
                                 + 0.994646 w_k1_AXP_s110
                                 + 0.932854 w_k1_DIS_s110
                                 + 0.786376 w_k1_FDX_s110
                                 + 1.18977 w_k1_IBM_s110
                                 + 0.87214 w_k1_INTC_s110
                                 + 1.10585 w_k1_KO_s110 + 1.15263 w_k1_MAT_s110
                                 + 0.906192 w_k1_MSFT_s110
                                 + 0.869535 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.10567 w_k1_ADM_s110
                                 + 0.997811 w_k1_AXP_s110
                                 + 1.0307 w_k1_DIS_s110
                                 + 0.984502 w_k1_FDX_s110
                                 + 0.746753 w_k1_IBM_s110
                                 + 0.972814 w_k1_INTC_s110
                                 + 0.868006 w_k1_KO_s110
                                 + 0.603029 w_k1_MAT_s110
                                 + 1.17692 w_k1_MSFT_s110
                                 + 0.953231 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.897887 w_k1_ADM_s111
                                 + 0.925159 w_k1_AXP_s111
                                 + 0.929442 w_k1_DIS_s111
                                 + 1.00016 w_k1_FDX_s111
                                 + 1.01719 w_k1_IBM_s111
                                 + 1.12938 w_k1_INTC_s111
                                 + 0.965133 w_k1_KO_s111
                                 + 0.922734 w_k1_MAT_s111
                                 + 0.990908 w_k1_MSFT_s111
                                 + 1.00015 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.857886 w_k1_ADM_s111
                                 + 0.964922 w_k1_AXP_s111
                                 + 0.769527 w_k1_DIS_s111
                                 + 0.811486 w_k1_FDX_s111
                                 + 1.01673 w_k1_IBM_s111
                                 + 0.919587 w_k1_INTC_s111
                                 + 0.804882 w_k1_KO_s111
                                 + 1.29193 w_k1_MAT_s111
                                 + 0.777475 w_k1_MSFT_s111
                                 + 0.851364 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.20541 w_k1_ADM_s111
                                 + 1.20347 w_k1_AXP_s111
                                 + 0.848231 w_k1_DIS_s111
                                 + 0.96347 w_k1_FDX_s111
                                 + 0.799765 w_k1_IBM_s111
                                 + 0.967561 w_k1_INTC_s111
                                 + 0.794085 w_k1_KO_s111 + 1.2182 w_k1_MAT_s111
                                 + 0.909337 w_k1_MSFT_s111
                                 + 0.920542 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.09342 w_k1_ADM_s111
                                 + 1.16327 w_k1_AXP_s111
                                 + 0.966533 w_k1_DIS_s111
                                 + 1.12871 w_k1_FDX_s111
                                 + 0.567469 w_k1_IBM_s111
                                 + 1.03298 w_k1_INTC_s111
                                 + 0.887807 w_k1_KO_s111
                                 + 0.853828 w_k1_MAT_s111
                                 + 1.06887 w_k1_MSFT_s111
                                 + 0.95935 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.21168 w_k1_ADM_s111
                                 + 1.35236 w_k1_AXP_s111
                                 + 1.57967 w_k1_DIS_s111
                                 + 1.30993 w_k1_FDX_s111
                                 + 1.18184 w_k1_IBM_s111
                                 + 1.45602 w_k1_INTC_s111
                                 + 1.26763 w_k1_KO_s111 + 1.26368 w_k1_MAT_s111
                                 + 1.31944 w_k1_MSFT_s111
                                 + 1.20173 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.30209 w_k1_ADM_s111
                                 + 1.40785 w_k1_AXP_s111
                                 + 0.929699 w_k1_DIS_s111
                                 + 1.01538 w_k1_FDX_s111
                                 + 1.18064 w_k1_IBM_s111
                                 + 1.04175 w_k1_INTC_s111
                                 + 1.08766 w_k1_KO_s111 + 1.58607 w_k1_MAT_s111
                                 + 0.977143 w_k1_MSFT_s111
                                 + 1.11344 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.998397 w_k1_ADM_s111
                                 + 1.50589 w_k1_AXP_s111
                                 + 1.16502 w_k1_DIS_s111
                                 + 0.982075 w_k1_FDX_s111
                                 + 1.6656 w_k1_IBM_s111 + 1.185 w_k1_INTC_s111
                                 + 1.08817 w_k1_KO_s111 + 2.04873 w_k1_MAT_s111
                                 + 0.940012 w_k1_MSFT_s111
                                 + 1.04784 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.729196 w_k1_ADM_s111
                                 + 0.778947 w_k1_AXP_s111
                                 + 0.640548 w_k1_DIS_s111
                                 + 0.80561 w_k1_FDX_s111
                                 + 0.626435 w_k1_IBM_s111
                                 + 1.00973 w_k1_INTC_s111
                                 + 0.669665 w_k1_KO_s111
                                 + 0.548442 w_k1_MAT_s111
                                 + 0.798666 w_k1_MSFT_s111
                                 + 0.757438 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.712163 w_k1_ADM_s112
                                 + 0.76892 w_k1_AXP_s112
                                 + 0.677328 w_k1_DIS_s112
                                 + 0.699027 w_k1_FDX_s112
                                 + 1.16121 w_k1_IBM_s112
                                 + 0.741359 w_k1_INTC_s112
                                 + 1.01257 w_k1_KO_s112 + 1.14397 w_k1_MAT_s112
                                 + 0.717298 w_k1_MSFT_s112
                                 + 0.915453 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.03265 w_k1_ADM_s112
                                 + 1.07603 w_k1_AXP_s112
                                 + 1.20308 w_k1_DIS_s112
                                 + 1.03686 w_k1_FDX_s112
                                 + 1.40271 w_k1_IBM_s112
                                 + 0.966227 w_k1_INTC_s112
                                 + 1.16022 w_k1_KO_s112 + 1.03233 w_k1_MAT_s112
                                 + 0.998931 w_k1_MSFT_s112
                                 + 1.02486 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.28344 w_k1_ADM_s112
                                 + 1.41532 w_k1_AXP_s112
                                 + 1.44902 w_k1_DIS_s112
                                 + 1.33733 w_k1_FDX_s112
                                 + 1.01185 w_k1_IBM_s112
                                 + 1.42443 w_k1_INTC_s112
                                 + 1.10898 w_k1_KO_s112
                                 + 0.850143 w_k1_MAT_s112
                                 + 1.25854 w_k1_MSFT_s112
                                 + 1.13814 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.02991 w_k1_ADM_s112
                                 + 1.07635 w_k1_AXP_s112
                                 + 1.14103 w_k1_DIS_s112
                                 + 1.12954 w_k1_FDX_s112
                                 + 0.827587 w_k1_IBM_s112
                                 + 1.04099 w_k1_INTC_s112
                                 + 0.980072 w_k1_KO_s112
                                 + 0.91897 w_k1_MAT_s112
                                 + 1.07066 w_k1_MSFT_s112
                                 + 0.988002 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.03517 w_k1_ADM_s112
                                 + 1.53778 w_k1_AXP_s112
                                 + 1.23827 w_k1_DIS_s112
                                 + 1.08439 w_k1_FDX_s112 + 1.356 w_k1_IBM_s112
                                 + 1.19787 w_k1_INTC_s112
                                 + 1.19858 w_k1_KO_s112 + 1.90737 w_k1_MAT_s112
                                 + 0.951719 w_k1_MSFT_s112
                                 + 1.03084 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.807181 w_k1_ADM_s112
                                 + 0.785149 w_k1_AXP_s112
                                 + 0.768745 w_k1_DIS_s112
                                 + 0.785927 w_k1_FDX_s112
                                 + 0.809886 w_k1_IBM_s112
                                 + 0.843781 w_k1_INTC_s112
                                 + 0.8759 w_k1_KO_s112 + 0.70355 w_k1_MAT_s112
                                 + 0.854767 w_k1_MSFT_s112
                                 + 0.801568 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.90748 w_k1_ADM_s112
                                 + 0.819316 w_k1_AXP_s112
                                 + 0.930789 w_k1_DIS_s112
                                 + 0.948186 w_k1_FDX_s112
                                 + 0.84322 w_k1_IBM_s112
                                 + 1.01494 w_k1_INTC_s112
                                 + 0.848237 w_k1_KO_s112
                                 + 0.799929 w_k1_MAT_s112
                                 + 0.951988 w_k1_MSFT_s112
                                 + 0.925023 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.27496 w_k1_ADM_s112
                                 + 1.54199 w_k1_AXP_s112
                                 + 1.81553 w_k1_DIS_s112
                                 + 1.37775 w_k1_FDX_s112
                                 + 1.66727 w_k1_IBM_s112
                                 + 1.44967 w_k1_INTC_s112
                                 + 1.79431 w_k1_KO_s112 + 1.79393 w_k1_MAT_s112
                                 + 1.4236 w_k1_MSFT_s112 + 1.24698 w_k1_PG_s112
                                 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.02668 w_k1_ADM_s113
                                 + 1.02136 w_k1_AXP_s113
                                 + 1.74596 w_k1_DIS_s113
                                 + 1.19641 w_k1_FDX_s113 + 1.1467 w_k1_IBM_s113
                                 + 1.32728 w_k1_INTC_s113
                                 + 1.23339 w_k1_KO_s113
                                 + 0.976226 w_k1_MAT_s113
                                 + 1.40227 w_k1_MSFT_s113
                                 + 1.09197 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.20065 w_k1_ADM_s113
                                 + 1.2665 w_k1_AXP_s113 + 1.28303 w_k1_DIS_s113
                                 + 1.29017 w_k1_FDX_s113
                                 + 0.864171 w_k1_IBM_s113
                                 + 1.30698 w_k1_INTC_s113
                                 + 1.08355 w_k1_KO_s113
                                 + 0.932163 w_k1_MAT_s113
                                 + 1.2284 w_k1_MSFT_s113 + 1.03789 w_k1_PG_s113
                                 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.750875 w_k1_ADM_s113
                                 + 0.719208 w_k1_AXP_s113
                                 + 0.73365 w_k1_DIS_s113
                                 + 0.841581 w_k1_FDX_s113
                                 + 0.963759 w_k1_IBM_s113
                                 + 0.854955 w_k1_INTC_s113
                                 + 1.00639 w_k1_KO_s113
                                 + 0.872326 w_k1_MAT_s113
                                 + 0.890741 w_k1_MSFT_s113
                                 + 0.939076 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.01463 w_k1_ADM_s113
                                 + 1.07518 w_k1_AXP_s113
                                 + 1.07448 w_k1_DIS_s113
                                 + 0.883119 w_k1_FDX_s113
                                 + 1.3662 w_k1_IBM_s113
                                 + 1.20987 w_k1_INTC_s113
                                 + 1.20993 w_k1_KO_s113 + 1.34644 w_k1_MAT_s113
                                 + 0.962718 w_k1_MSFT_s113
                                 + 0.898862 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.52292 w_k1_ADM_s113
                                 + 1.8897 w_k1_AXP_s113 + 1.44824 w_k1_DIS_s113
                                 + 1.43642 w_k1_FDX_s113
                                 + 1.07861 w_k1_IBM_s113
                                 + 1.20284 w_k1_INTC_s113 + 1.1387 w_k1_KO_s113
                                 + 1.17946 w_k1_MAT_s113
                                 + 1.07763 w_k1_MSFT_s113
                                 + 1.13574 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.29352 w_k1_ADM_s113
                                 + 1.31413 w_k1_AXP_s113
                                 + 1.20213 w_k1_DIS_s113
                                 + 1.15759 w_k1_FDX_s113 + 1.2864 w_k1_IBM_s113
                                 + 1.20837 w_k1_INTC_s113
                                 + 1.21193 w_k1_KO_s113 + 1.13539 w_k1_MAT_s113
                                 + 1.02397 w_k1_MSFT_s113
                                 + 1.09604 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.891612 w_k1_ADM_s113
                                 + 0.885702 w_k1_AXP_s113
                                 + 0.982323 w_k1_DIS_s113
                                 + 0.972871 w_k1_FDX_s113
                                 + 0.78569 w_k1_IBM_s113
                                 + 1.05434 w_k1_INTC_s113
                                 + 0.894487 w_k1_KO_s113
                                 + 0.680423 w_k1_MAT_s113
                                 + 0.905952 w_k1_MSFT_s113
                                 + 0.892005 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.786386 w_k1_ADM_s113
                                 + 0.921913 w_k1_AXP_s113
                                 + 0.746222 w_k1_DIS_s113
                                 + 0.934533 w_k1_FDX_s113
                                 + 0.649232 w_k1_IBM_s113
                                 + 1.04628 w_k1_INTC_s113
                                 + 0.74512 w_k1_KO_s113
                                 + 0.558272 w_k1_MAT_s113
                                 + 0.797233 w_k1_MSFT_s113
                                 + 0.856026 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.12059 w_k1_ADM_s114
                                 + 0.970094 w_k1_AXP_s114
                                 + 0.992646 w_k1_DIS_s114
                                 + 0.961646 w_k1_FDX_s114
                                 + 1.45031 w_k1_IBM_s114
                                 + 0.901845 w_k1_INTC_s114
                                 + 1.0801 w_k1_KO_s114 + 1.54589 w_k1_MAT_s114
                                 + 1.04594 w_k1_MSFT_s114
                                 + 1.13747 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.51768 w_k1_ADM_s114
                                 + 1.14158 w_k1_AXP_s114
                                 + 1.33924 w_k1_DIS_s114
                                 + 1.30417 w_k1_FDX_s114
                                 + 1.02067 w_k1_IBM_s114
                                 + 1.2015 w_k1_INTC_s114 + 1.13059 w_k1_KO_s114
                                 + 0.843925 w_k1_MAT_s114
                                 + 1.34632 w_k1_MSFT_s114
                                 + 1.12237 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.878639 w_k1_ADM_s114
                                 + 0.858844 w_k1_AXP_s114
                                 + 0.871111 w_k1_DIS_s114
                                 + 1.01728 w_k1_FDX_s114
                                 + 0.928602 w_k1_IBM_s114
                                 + 1.18455 w_k1_INTC_s114
                                 + 0.935873 w_k1_KO_s114
                                 + 0.957622 w_k1_MAT_s114
                                 + 1.1504 w_k1_MSFT_s114
                                 + 0.986637 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.26621 w_k1_ADM_s114
                                 + 1.43968 w_k1_AXP_s114
                                 + 1.66503 w_k1_DIS_s114
                                 + 1.34297 w_k1_FDX_s114
                                 + 1.38393 w_k1_IBM_s114
                                 + 1.46159 w_k1_INTC_s114
                                 + 1.29317 w_k1_KO_s114 + 1.39145 w_k1_MAT_s114
                                 + 1.3984 w_k1_MSFT_s114 + 1.14496 w_k1_PG_s114
                                 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.921017 w_k1_ADM_s114
                                 + 0.601664 w_k1_AXP_s114
                                 + 0.702104 w_k1_DIS_s114
                                 + 0.828579 w_k1_FDX_s114
                                 + 0.767962 w_k1_IBM_s114
                                 + 1.03914 w_k1_INTC_s114
                                 + 0.734543 w_k1_KO_s114
                                 + 0.497798 w_k1_MAT_s114
                                 + 0.91343 w_k1_MSFT_s114
                                 + 0.823728 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.18232 w_k1_ADM_s114
                                 + 1.13215 w_k1_AXP_s114
                                 + 0.992609 w_k1_DIS_s114
                                 + 1.16149 w_k1_FDX_s114
                                 + 0.840274 w_k1_IBM_s114
                                 + 1.16683 w_k1_INTC_s114
                                 + 0.934051 w_k1_KO_s114
                                 + 0.782901 w_k1_MAT_s114
                                 + 1.04165 w_k1_MSFT_s114
                                 + 1.00267 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.926195 w_k1_ADM_s114
                                 + 0.830227 w_k1_AXP_s114
                                 + 0.927138 w_k1_DIS_s114
                                 + 0.901662 w_k1_FDX_s114
                                 + 1.13116 w_k1_IBM_s114
                                 + 1.15774 w_k1_INTC_s114
                                 + 0.867895 w_k1_KO_s114
                                 + 0.755972 w_k1_MAT_s114
                                 + 0.986182 w_k1_MSFT_s114
                                 + 0.894033 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.21031 w_k1_ADM_s114
                                 + 0.652917 w_k1_AXP_s114
                                 + 1.06008 w_k1_DIS_s114
                                 + 0.96912 w_k1_FDX_s114
                                 + 0.904106 w_k1_IBM_s114
                                 + 0.857346 w_k1_INTC_s114
                                 + 0.874048 w_k1_KO_s114
                                 + 0.531605 w_k1_MAT_s114
                                 + 1.06726 w_k1_MSFT_s114
                                 + 0.878813 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.808408 w_k1_ADM_s115
                                 + 1.0657 w_k1_AXP_s115 + 0.88338 w_k1_DIS_s115
                                 + 0.883855 w_k1_FDX_s115
                                 + 1.35574 w_k1_IBM_s115
                                 + 1.02956 w_k1_INTC_s115
                                 + 1.24889 w_k1_KO_s115 + 1.43757 w_k1_MAT_s115
                                 + 0.803597 w_k1_MSFT_s115
                                 + 0.883989 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.881559 w_k1_ADM_s115
                                 + 0.720704 w_k1_AXP_s115
                                 + 0.65917 w_k1_DIS_s115
                                 + 0.811378 w_k1_FDX_s115
                                 + 0.70906 w_k1_IBM_s115
                                 + 0.880573 w_k1_INTC_s115
                                 + 0.849294 w_k1_KO_s115
                                 + 0.702452 w_k1_MAT_s115
                                 + 0.894095 w_k1_MSFT_s115
                                 + 0.792908 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.825906 w_k1_ADM_s115
                                 + 0.659926 w_k1_AXP_s115
                                 + 0.758935 w_k1_DIS_s115
                                 + 0.738432 w_k1_FDX_s115
                                 + 1.02747 w_k1_IBM_s115
                                 + 0.848571 w_k1_INTC_s115
                                 + 1.04635 w_k1_KO_s115 + 1.11383 w_k1_MAT_s115
                                 + 0.899866 w_k1_MSFT_s115
                                 + 0.807724 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.31313 w_k1_ADM_s115
                                 + 1.17203 w_k1_AXP_s115
                                 + 1.44921 w_k1_DIS_s115
                                 + 1.27683 w_k1_FDX_s115
                                 + 1.00196 w_k1_IBM_s115
                                 + 1.04798 w_k1_INTC_s115
                                 + 1.11966 w_k1_KO_s115
                                 + 0.944199 w_k1_MAT_s115
                                 + 1.27591 w_k1_MSFT_s115
                                 + 1.06326 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.22681 w_k1_ADM_s115
                                 + 1.38071 w_k1_AXP_s115
                                 + 1.39257 w_k1_DIS_s115 + 1.2194 w_k1_FDX_s115
                                 + 1.50032 w_k1_IBM_s115
                                 + 1.35391 w_k1_INTC_s115
                                 + 1.35883 w_k1_KO_s115 + 1.37711 w_k1_MAT_s115
                                 + 1.1958 w_k1_MSFT_s115 + 1.09901 w_k1_PG_s115
                                 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.894292 w_k1_ADM_s115
                                 + 1.11749 w_k1_AXP_s115 + 1.0495 w_k1_DIS_s115
                                 + 1.13787 w_k1_FDX_s115
                                 + 0.804869 w_k1_IBM_s115
                                 + 1.17862 w_k1_INTC_s115
                                 + 0.93817 w_k1_KO_s115
                                 + 0.870565 w_k1_MAT_s115
                                 + 1.0268 w_k1_MSFT_s115
                                 + 0.911517 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.24087 w_k1_ADM_s115
                                 + 1.75498 w_k1_AXP_s115
                                 + 1.42779 w_k1_DIS_s115
                                 + 1.47446 w_k1_FDX_s115
                                 + 1.23352 w_k1_IBM_s115
                                 + 1.37586 w_k1_INTC_s115
                                 + 1.48499 w_k1_KO_s115 + 1.53604 w_k1_MAT_s115
                                 + 1.33126 w_k1_MSFT_s115
                                 + 1.25185 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.904843 w_k1_ADM_s115
                                 + 1.01951 w_k1_AXP_s115
                                 + 0.965408 w_k1_DIS_s115
                                 + 0.966628 w_k1_FDX_s115
                                 + 1.08872 w_k1_IBM_s115
                                 + 1.05489 w_k1_INTC_s115 + 1.084 w_k1_KO_s115
                                 + 1.13463 w_k1_MAT_s115
                                 + 1.01613 w_k1_MSFT_s115
                                 + 1.05973 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.910819 w_k1_ADM_s116
                                 + 0.722869 w_k1_AXP_s116
                                 + 0.665553 w_k1_DIS_s116
                                 + 0.701166 w_k1_FDX_s116
                                 + 0.869518 w_k1_IBM_s116
                                 + 0.805686 w_k1_INTC_s116
                                 + 0.765841 w_k1_KO_s116
                                 + 1.11435 w_k1_MAT_s116
                                 + 0.759348 w_k1_MSFT_s116
                                 + 0.930782 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.02906 w_k1_ADM_s116
                                 + 0.972106 w_k1_AXP_s116
                                 + 0.936864 w_k1_DIS_s116
                                 + 1.06655 w_k1_FDX_s116
                                 + 0.780484 w_k1_IBM_s116
                                 + 1.17691 w_k1_INTC_s116
                                 + 1.01826 w_k1_KO_s116
                                 + 0.931653 w_k1_MAT_s116
                                 + 1.06091 w_k1_MSFT_s116
                                 + 0.983101 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.928803 w_k1_ADM_s116
                                 + 1.1962 w_k1_AXP_s116
                                 + 0.860967 w_k1_DIS_s116
                                 + 0.909639 w_k1_FDX_s116
                                 + 1.32715 w_k1_IBM_s116
                                 + 0.739692 w_k1_INTC_s116
                                 + 0.972262 w_k1_KO_s116
                                 + 1.85958 w_k1_MAT_s116
                                 + 0.856424 w_k1_MSFT_s116
                                 + 1.06446 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.106 w_k1_ADM_s116
                                 + 1.37644 w_k1_AXP_s116
                                 + 1.32933 w_k1_DIS_s116
                                 + 1.23088 w_k1_FDX_s116
                                 + 0.996658 w_k1_IBM_s116
                                 + 1.25391 w_k1_INTC_s116
                                 + 1.18787 w_k1_KO_s116 + 1.35658 w_k1_MAT_s116
                                 + 1.28626 w_k1_MSFT_s116
                                 + 1.20223 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.02411 w_k1_ADM_s116
                                 + 1.16165 w_k1_AXP_s116
                                 + 1.31314 w_k1_DIS_s116
                                 + 1.05644 w_k1_FDX_s116
                                 + 1.35721 w_k1_IBM_s116
                                 + 1.12075 w_k1_INTC_s116 + 1.2217 w_k1_KO_s116
                                 + 1.33331 w_k1_MAT_s116
                                 + 1.00291 w_k1_MSFT_s116
                                 + 1.24817 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.24346 w_k1_ADM_s116
                                 + 1.01608 w_k1_AXP_s116
                                 + 0.993478 w_k1_DIS_s116
                                 + 0.990332 w_k1_FDX_s116
                                 + 1.01185 w_k1_IBM_s116
                                 + 1.04358 w_k1_INTC_s116
                                 + 0.958753 w_k1_KO_s116
                                 + 1.02506 w_k1_MAT_s116
                                 + 0.9012 w_k1_MSFT_s116 + 1.02049 w_k1_PG_s116
                                 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.16861 w_k1_ADM_s116
                                 + 1.06917 w_k1_AXP_s116
                                 + 1.12849 w_k1_DIS_s116
                                 + 0.964507 w_k1_FDX_s116
                                 + 1.76327 w_k1_IBM_s116
                                 + 0.817508 w_k1_INTC_s116
                                 + 1.23584 w_k1_KO_s116 + 1.59971 w_k1_MAT_s116
                                 + 0.85104 w_k1_MSFT_s116 + 1.0095 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.26315 w_k1_ADM_s116
                                 + 1.28231 w_k1_AXP_s116
                                 + 1.84194 w_k1_DIS_s116 + 1.3764 w_k1_FDX_s116
                                 + 1.11442 w_k1_IBM_s116
                                 + 1.06236 w_k1_INTC_s116
                                 + 1.44335 w_k1_KO_s116 + 1.14503 w_k1_MAT_s116
                                 + 1.31443 w_k1_MSFT_s116 + 1.3001 w_k1_PG_s116
                                 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.946596 w_k1_ADM_s117
                                 + 1.35563 w_k1_AXP_s117
                                 + 1.25569 w_k1_DIS_s117
                                 + 0.970938 w_k1_FDX_s117
                                 + 1.77393 w_k1_IBM_s117
                                 + 1.01944 w_k1_INTC_s117
                                 + 1.26082 w_k1_KO_s117 + 2.11172 w_k1_MAT_s117
                                 + 1.10926 w_k1_MSFT_s117 + 1.0582 w_k1_PG_s117
                                 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.29234 w_k1_ADM_s117
                                 + 1.59202 w_k1_AXP_s117
                                 + 1.58814 w_k1_DIS_s117
                                 + 1.31564 w_k1_FDX_s117
                                 + 1.29149 w_k1_IBM_s117
                                 + 1.37476 w_k1_INTC_s117
                                 + 1.34766 w_k1_KO_s117 + 1.20722 w_k1_MAT_s117
                                 + 1.352 w_k1_MSFT_s117 + 1.29652 w_k1_PG_s117
                                 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.811433 w_k1_ADM_s117
                                 + 1.02617 w_k1_AXP_s117
                                 + 1.00405 w_k1_DIS_s117
                                 + 0.857486 w_k1_FDX_s117
                                 + 1.29663 w_k1_IBM_s117
                                 + 1.06387 w_k1_INTC_s117
                                 + 1.03586 w_k1_KO_s117 + 1.40867 w_k1_MAT_s117
                                 + 0.932233 w_k1_MSFT_s117
                                 + 0.915949 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.738749 w_k1_ADM_s117
                                 + 0.844081 w_k1_AXP_s117
                                 + 0.652506 w_k1_DIS_s117
                                 + 0.853197 w_k1_FDX_s117
                                 + 0.516705 w_k1_IBM_s117
                                 + 0.770755 w_k1_INTC_s117
                                 + 0.619751 w_k1_KO_s117
                                 + 0.678121 w_k1_MAT_s117
                                 + 0.82742 w_k1_MSFT_s117
                                 + 0.792898 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.143 w_k1_ADM_s117
                                 + 1.14193 w_k1_AXP_s117
                                 + 1.00885 w_k1_DIS_s117
                                 + 1.02907 w_k1_FDX_s117
                                 + 0.91755 w_k1_IBM_s117
                                 + 1.07588 w_k1_INTC_s117
                                 + 0.955599 w_k1_KO_s117
                                 + 0.834433 w_k1_MAT_s117
                                 + 0.986576 w_k1_MSFT_s117
                                 + 0.998553 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.935469 w_k1_ADM_s117
                                 + 0.87441 w_k1_AXP_s117
                                 + 0.811779 w_k1_DIS_s117
                                 + 0.829989 w_k1_FDX_s117
                                 + 0.827564 w_k1_IBM_s117
                                 + 0.904321 w_k1_INTC_s117
                                 + 0.91279 w_k1_KO_s117
                                 + 0.867974 w_k1_MAT_s117
                                 + 0.871524 w_k1_MSFT_s117
                                 + 0.868946 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.96082 w_k1_ADM_s117
                                 + 1.29351 w_k1_AXP_s117
                                 + 1.34817 w_k1_DIS_s117
                                 + 1.26606 w_k1_FDX_s117
                                 + 0.978225 w_k1_IBM_s117
                                 + 1.11585 w_k1_INTC_s117
                                 + 1.06224 w_k1_KO_s117 + 1.508 w_k1_MAT_s117
                                 + 1.18826 w_k1_MSFT_s117
                                 + 1.05316 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.10675 w_k1_ADM_s117
                                 + 1.22044 w_k1_AXP_s117
                                 + 1.25072 w_k1_DIS_s117
                                 + 1.31466 w_k1_FDX_s117
                                 + 0.593596 w_k1_IBM_s117
                                 + 1.31403 w_k1_INTC_s117
                                 + 0.902417 w_k1_KO_s117
                                 + 0.705439 w_k1_MAT_s117
                                 + 1.25447 w_k1_MSFT_s117
                                 + 1.00721 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.010295772739065 w_k1_ADM_s000
                                 + 0.00749283655876514 w_k1_ADM_s001
                                 + 0.00669747065189049 w_k1_ADM_s002
                                 + 0.011477457351222 w_k1_ADM_s003
                                 + 0.0109440843419608 w_k1_ADM_s004
                                 + 0.0132280839704244 w_k1_ADM_s005
                                 + 0.011264982488901 w_k1_ADM_s006
                                 + 0.0199805130987438 w_k1_ADM_s007
                                 + 0.0100362457462497 w_k1_ADM_s010
                                 + 0.0468147730665638 w_k1_ADM_s011
                                 + 0.013747165939317 w_k1_ADM_s012
                                 + 0.0152024296446892 w_k1_ADM_s013
                                 + 0.0166296242148507 w_k1_ADM_s014
                                 + 0.0515824245525261 w_k1_ADM_s015
                                 + 0.016287221784745 w_k1_ADM_s016
                                 + 0.0179880998858314 w_k1_ADM_s017
                                 + 0.0308357161896521 w_k1_ADM_s100
                                 + 0.0502666008201078 w_k1_ADM_s101
                                 + 0.0415593329348161 w_k1_ADM_s102
                                 + 0.0231645875961377 w_k1_ADM_s103
                                 + 0.025553412473526 w_k1_ADM_s104
                                 + 0.0783247857274393 w_k1_ADM_s105
                                 + 0.034050235529005 w_k1_ADM_s106
                                 + 0.0317289041442251 w_k1_ADM_s107
                                 + 0.0575393757709656 w_k1_ADM_s110
                                 + 0.0357973725693397 w_k1_ADM_s111
                                 + 0.0720562310768498 w_k1_ADM_s112
                                 + 0.0377102177064778 w_k1_ADM_s113
                                 + 0.0502623602319446 w_k1_ADM_s114
                                 + 0.0795534700305117 w_k1_ADM_s115
                                 + 0.0466841629836506 w_k1_ADM_s116
                                 + 0.0221440645081339 w_k1_ADM_s117
                                 + 0.0108657131094389 w_k1_AXP_s000
                                 + 0.00816078153503479 w_k1_AXP_s001
                                 + 0.00717093453546838 w_k1_AXP_s002
                                 + 0.0116190693344839 w_k1_AXP_s003
                                 + 0.00940619253296027 w_k1_AXP_s004
                                 + 0.0129574862171445 w_k1_AXP_s005
                                 + 0.0103721425190561 w_k1_AXP_s006
                                 + 0.0171340230160783 w_k1_AXP_s007
                                 + 0.0107879866499909 w_k1_AXP_s010
                                 + 0.0434307131034523 w_k1_AXP_s011
                                 + 0.0148094260561507 w_k1_AXP_s012
                                 + 0.014985889114226 w_k1_AXP_s013
                                 + 0.0195735546634543 w_k1_AXP_s014
                                 + 0.0547143793054503 w_k1_AXP_s015
                                 + 0.0164458599023854 w_k1_AXP_s016
                                 + 0.0219033478705989 w_k1_AXP_s017
                                 + 0.0375011047873217 w_k1_AXP_s100
                                 + 0.053817317662436 w_k1_AXP_s101
                                 + 0.0401326878138165 w_k1_AXP_s102
                                 + 0.0261296457067267 w_k1_AXP_s103
                                 + 0.0263635852650299 w_k1_AXP_s104
                                 + 0.0846629697466828 w_k1_AXP_s105
                                 + 0.0305728892059991 w_k1_AXP_s106
                                 + 0.0329136677685427 w_k1_AXP_s107
                                 + 0.0560159879435264 w_k1_AXP_s110
                                 + 0.039580973855804 w_k1_AXP_s111
                                 + 0.0766801844672542 w_k1_AXP_s112
                                 + 0.0394675356043387 w_k1_AXP_s113
                                 + 0.0418068194160645 w_k1_AXP_s114
                                 + 0.0864504032725655 w_k1_AXP_s115
                                 + 0.045250128699879 w_k1_AXP_s116
                                 + 0.0259372391790219 w_k1_AXP_s117
                                 + 0.0112114451991481 w_k1_DIS_s000
                                 + 0.00775528316231149 w_k1_DIS_s001
                                 + 0.00689736835504424 w_k1_DIS_s002
                                 + 0.0113804924576257 w_k1_DIS_s003
                                 + 0.0101208359577888 w_k1_DIS_s004
                                 + 0.0131756910501878 w_k1_DIS_s005
                                 + 0.0111006882990612 w_k1_DIS_s006
                                 + 0.0170911279986199 w_k1_DIS_s007
                                 + 0.0107841959801005 w_k1_DIS_s010
                                 + 0.044370864365499 w_k1_DIS_s011
                                 + 0.014455399352625 w_k1_DIS_s012
                                 + 0.0147315115706109 w_k1_DIS_s013
                                 + 0.0202896683785254 w_k1_DIS_s014
                                 + 0.0509460922824419 w_k1_DIS_s015
                                 + 0.018126178427065 w_k1_DIS_s016
                                 + 0.0208434845718278 w_k1_DIS_s017
                                 + 0.0365321398421221 w_k1_DIS_s100
                                 + 0.0606010874419595 w_k1_DIS_s101
                                 + 0.0419820525617594 w_k1_DIS_s102
                                 + 0.0251756742697006 w_k1_DIS_s103
                                 + 0.0265586193884338 w_k1_DIS_s104
                                 + 0.0905986139258985 w_k1_DIS_s105
                                 + 0.0347902907626317 w_k1_DIS_s106
                                 + 0.0334787336148136 w_k1_DIS_s107
                                 + 0.0595122970496484 w_k1_DIS_s110
                                 + 0.0352635349554071 w_k1_DIS_s111
                                 + 0.0807878872416304 w_k1_DIS_s112
                                 + 0.0398735593391631 w_k1_DIS_s113
                                 + 0.0468336801281309 w_k1_DIS_s114
                                 + 0.0841476420577873 w_k1_DIS_s115
                                 + 0.0456898053167497 w_k1_DIS_s116
                                 + 0.0246563486551972 w_k1_DIS_s117
                                 + 0.011004033870848 w_k1_FDX_s000
                                 + 0.00738345228870846 w_k1_FDX_s001
                                 + 0.00716527497216385 w_k1_FDX_s002
                                 + 0.0117266889468608 w_k1_FDX_s003
                                 + 0.0101769093735589 w_k1_FDX_s004
                                 + 0.0129272855874673 w_k1_FDX_s005
                                 + 0.0107551936347528 w_k1_FDX_s006
                                 + 0.0175629852562586 w_k1_FDX_s007
                                 + 0.0106344396437606 w_k1_FDX_s010
                                 + 0.0442904220035417 w_k1_FDX_s011
                                 + 0.0134047489867727 w_k1_FDX_s012
                                 + 0.0144174741948397 w_k1_FDX_s013
                                 + 0.0198087519834648 w_k1_FDX_s014
                                 + 0.0523240206659438 w_k1_FDX_s015
                                 + 0.0165525887929487 w_k1_FDX_s016
                                 + 0.020575590825674 w_k1_FDX_s017
                                 + 0.0379332811243629 w_k1_FDX_s100
                                 + 0.0583923829100031 w_k1_FDX_s101
                                 + 0.0416319227059766 w_k1_FDX_s102
                                 + 0.0255080560253585 w_k1_FDX_s103
                                 + 0.0274873082890257 w_k1_FDX_s104
                                 + 0.0886144327156081 w_k1_FDX_s105
                                 + 0.0316570217819888 w_k1_FDX_s106
                                 + 0.032976386067728 w_k1_FDX_s107
                                 + 0.0510912452138618 w_k1_FDX_s110
                                 + 0.0353639807825493 w_k1_FDX_s111
                                 + 0.0743350121636046 w_k1_FDX_s112
                                 + 0.0380623979644982 w_k1_FDX_s113
                                 + 0.0468865450534799 w_k1_FDX_s114
                                 + 0.0847160023509421 w_k1_FDX_s115
                                 + 0.0437463618615586 w_k1_FDX_s116
                                 + 0.0231880294793915 w_k1_FDX_s117
                                 + 0.0119646563359451 w_k1_IBM_s000
                                 + 0.00799076975321443 w_k1_IBM_s001
                                 + 0.00698447417682447 w_k1_IBM_s002
                                 + 0.0130954245077887 w_k1_IBM_s003
                                 + 0.0101170701016716 w_k1_IBM_s004
                                 + 0.0131784526120821 w_k1_IBM_s005
                                 + 0.0105857230355267 w_k1_IBM_s006
                                 + 0.0179171967499713 w_k1_IBM_s007
                                 + 0.0112055586732511 w_k1_IBM_s010
                                 + 0.0394247972687857 w_k1_IBM_s011
                                 + 0.0144188718854767 w_k1_IBM_s012
                                 + 0.0145476293280926 w_k1_IBM_s013
                                 + 0.0188282882698141 w_k1_IBM_s014
                                 + 0.0537195848296165 w_k1_IBM_s015
                                 + 0.0171429277840153 w_k1_IBM_s016
                                 + 0.0203665982780467 w_k1_IBM_s017
                                 + 0.0365483357821443 w_k1_IBM_s100
                                 + 0.0599760021613145 w_k1_IBM_s101
                                 + 0.0375274687510435 w_k1_IBM_s102
                                 + 0.0269536363120181 w_k1_IBM_s103
                                 + 0.0259291061196218 w_k1_IBM_s104
                                 + 0.0859076343766242 w_k1_IBM_s105
                                 + 0.0318978806445933 w_k1_IBM_s106
                                 + 0.0320082160338456 w_k1_IBM_s107
                                 + 0.0551820107591171 w_k1_IBM_s110
                                 + 0.0356154188588569 w_k1_IBM_s111
                                 + 0.0798519226645353 w_k1_IBM_s112
                                 + 0.0373752523485014 w_k1_IBM_s113
                                 + 0.0480952101720955 w_k1_IBM_s114
                                 + 0.0823963568143116 w_k1_IBM_s115
                                 + 0.0465516218253744 w_k1_IBM_s116
                                 + 0.0235477910524086 w_k1_IBM_s117
                                 + 0.0106135857788072 w_k1_INTC_s000
                                 + 0.0077287724161453 w_k1_INTC_s001
                                 + 0.00680660097188313 w_k1_INTC_s002
                                 + 0.0117771211133605 w_k1_INTC_s003
                                 + 0.00868666065928804 w_k1_INTC_s004
                                 + 0.0141279112991705 w_k1_INTC_s005
                                 + 0.0115591267749115 w_k1_INTC_s006
                                 + 0.0170087475857757 w_k1_INTC_s007
                                 + 0.0123581700299596 w_k1_INTC_s010
                                 + 0.0424936075238676 w_k1_INTC_s011
                                 + 0.0135281902456477 w_k1_INTC_s012
                                 + 0.014984395685374 w_k1_INTC_s013
                                 + 0.0208401349476938 w_k1_INTC_s014
                                 + 0.0596802719236239 w_k1_INTC_s015
                                 + 0.0160358997276568 w_k1_INTC_s016
                                 + 0.0208860188963597 w_k1_INTC_s017
                                 + 0.0384374506996493 w_k1_INTC_s100
                                 + 0.0649280147227414 w_k1_INTC_s101
                                 + 0.0420089671743314 w_k1_INTC_s102
                                 + 0.0289240473244802 w_k1_INTC_s103
                                 + 0.0259595797357974 w_k1_INTC_s104
                                 + 0.0719534429411216 w_k1_INTC_s105
                                 + 0.0327920766880125 w_k1_INTC_s106
                                 + 0.0333090339334795 w_k1_INTC_s107
                                 + 0.0533190109262998 w_k1_INTC_s110
                                 + 0.0389446718138702 w_k1_INTC_s111
                                 + 0.0759484357162031 w_k1_INTC_s112
                                 + 0.0409793561027674 w_k1_INTC_s113
                                 + 0.0511668060225845 w_k1_INTC_s114
                                 + 0.0868996802709203 w_k1_INTC_s115
                                 + 0.0449287033879485 w_k1_INTC_s116
                                 + 0.0240121170380814 w_k1_INTC_s117
                                 + 0.0111647714553544 w_k1_KO_s000
                                 + 0.00861797652850164 w_k1_KO_s001
                                 + 0.00644118939404092 w_k1_KO_s002
                                 + 0.0126316498266774 w_k1_KO_s003
                                 + 0.00967445107852445 w_k1_KO_s004
                                 + 0.0128897488972866 w_k1_KO_s005
                                 + 0.0102528169499757 w_k1_KO_s006
                                 + 0.0170359146406726 w_k1_KO_s007
                                 + 0.01104049555528 w_k1_KO_s010
                                 + 0.0405780887730923 w_k1_KO_s011
                                 + 0.0138854468166467 w_k1_KO_s012
                                 + 0.0138605077341931 w_k1_KO_s013
                                 + 0.0187360465932379 w_k1_KO_s014
                                 + 0.0552554679718762 w_k1_KO_s015
                                 + 0.0167103185763926 w_k1_KO_s016
                                 + 0.021506291457515 w_k1_KO_s017
                                 + 0.0321951462546245 w_k1_KO_s100
                                 + 0.0535759055890948 w_k1_KO_s101
                                 + 0.0405485295231768 w_k1_KO_s102
                                 + 0.0267617280250329 w_k1_KO_s103
                                 + 0.0258809781335211 w_k1_KO_s104
                                 + 0.0842263948346232 w_k1_KO_s105
                                 + 0.0363843522537445 w_k1_KO_s106
                                 + 0.0335944899147804 w_k1_KO_s107
                                 + 0.0529506862192394 w_k1_KO_s110
                                 + 0.0337029261825946 w_k1_KO_s111
                                 + 0.0782404385509474 w_k1_KO_s112
                                 + 0.038443583913885 w_k1_KO_s113
                                 + 0.043643882734581 w_k1_KO_s114
                                 + 0.0879431774914854 w_k1_KO_s115
                                 + 0.0456545358483595 w_k1_KO_s116
                                 + 0.0227614877769343 w_k1_KO_s117
                                 + 0.00954800434175113 w_k1_MAT_s000
                                 + 0.00738072361980168 w_k1_MAT_s001
                                 + 0.0077233522122109 w_k1_MAT_s002
                                 + 0.0168182082909054 w_k1_MAT_s003
                                 + 0.0119863434687297 w_k1_MAT_s004
                                 + 0.0118000243086525 w_k1_MAT_s005
                                 + 0.0100724121382406 w_k1_MAT_s006
                                 + 0.0155656382600135 w_k1_MAT_s007
                                 + 0.0100108415391533 w_k1_MAT_s010
                                 + 0.0412958780625077 w_k1_MAT_s011
                                 + 0.0141082252565219 w_k1_MAT_s012
                                 + 0.0128503246369624 w_k1_MAT_s013
                                 + 0.0177091405747277 w_k1_MAT_s014
                                 + 0.0481484911400465 w_k1_MAT_s015
                                 + 0.0183694456069824 w_k1_MAT_s016
                                 + 0.0229628565905936 w_k1_MAT_s017
                                 + 0.0403595005539692 w_k1_MAT_s100
                                 + 0.0596284408204263 w_k1_MAT_s101
                                 + 0.0329027418107772 w_k1_MAT_s102
                                 + 0.0234822830588783 w_k1_MAT_s103
                                 + 0.027992233260025 w_k1_MAT_s104
                                 + 0.0931803958398947 w_k1_MAT_s105
                                 + 0.028088532433661 w_k1_MAT_s106
                                 + 0.0318031027321579 w_k1_MAT_s107
                                 + 0.0568993907346813 w_k1_MAT_s110
                                 + 0.0415065565165997 w_k1_MAT_s111
                                 + 0.0744305630171422 w_k1_MAT_s112
                                 + 0.0347828651041949 w_k1_MAT_s113
                                 + 0.0401513314061397 w_k1_MAT_s114
                                 + 0.0864090660422874 w_k1_MAT_s115
                                 + 0.0524359425749673 w_k1_MAT_s116
                                 + 0.0265108788122301 w_k1_MAT_s117
                                 + 0.0108549814213175 w_k1_MSFT_s000
                                 + 0.00812405305657975 w_k1_MSFT_s001
                                 + 0.00676592952793719 w_k1_MSFT_s002
                                 + 0.0101336142514069 w_k1_MSFT_s003
                                 + 0.00861965655062097 w_k1_MSFT_s004
                                 + 0.0137765820133362 w_k1_MSFT_s005
                                 + 0.011307801805254 w_k1_MSFT_s006
                                 + 0.0165546521035848 w_k1_MSFT_s007
                                 + 0.0106991070138553 w_k1_MSFT_s010
                                 + 0.0431955576217374 w_k1_MSFT_s011
                                 + 0.0146208154529719 w_k1_MSFT_s012
                                 + 0.0139897913289527 w_k1_MSFT_s013
                                 + 0.0196983151437567 w_k1_MSFT_s014
                                 + 0.050439654734396 w_k1_MSFT_s015
                                 + 0.0162601871704451 w_k1_MSFT_s016
                                 + 0.0224290366487202 w_k1_MSFT_s017
                                 + 0.0375126585712698 w_k1_MSFT_s100
                                 + 0.0597435148323083 w_k1_MSFT_s101
                                 + 0.0399938022844475 w_k1_MSFT_s102
                                 + 0.0272015545335224 w_k1_MSFT_s103
                                 + 0.026495776565972 w_k1_MSFT_s104
                                 + 0.0846998926858418 w_k1_MSFT_s105
                                 + 0.032071073228334 w_k1_MSFT_s106
                                 + 0.0311519394194641 w_k1_MSFT_s107
                                 + 0.0572713481392982 w_k1_MSFT_s110
                                 + 0.0345604377884049 w_k1_MSFT_s111
                                 + 0.0735146745435014 w_k1_MSFT_s112
                                 + 0.0364697300479435 w_k1_MSFT_s113
                                 + 0.0497850137632513 w_k1_MSFT_s114
                                 + 0.0839640556502634 w_k1_MSFT_s115
                                 + 0.0426393558603069 w_k1_MSFT_s116
                                 + 0.023471866766454 w_k1_MSFT_s117
                                 + 0.0111565268503573 w_k1_PG_s000
                                 + 0.00811555929978632 w_k1_PG_s001
                                 + 0.00621648466324989 w_k1_PG_s002
                                 + 0.0113298255745343 w_k1_PG_s003
                                 + 0.00927788080427405 w_k1_PG_s004
                                 + 0.0130034639926593 w_k1_PG_s005
                                 + 0.0110810585048931 w_k1_PG_s006
                                 + 0.0178173624881653 w_k1_PG_s007
                                 + 0.0116747538929035 w_k1_PG_s010
                                 + 0.040086190585986 w_k1_PG_s011
                                 + 0.0135453579403191 w_k1_PG_s012
                                 + 0.012207347677279 w_k1_PG_s013
                                 + 0.0181540864662891 w_k1_PG_s014
                                 + 0.0503278281505345 w_k1_PG_s015
                                 + 0.0159339636560153 w_k1_PG_s016
                                 + 0.0204243210739511 w_k1_PG_s017
                                 + 0.0359478559127443 w_k1_PG_s100
                                 + 0.0574579796396549 w_k1_PG_s101
                                 + 0.039315720132011 w_k1_PG_s102
                                 + 0.0251706610860333 w_k1_PG_s103
                                 + 0.0265273710565589 w_k1_PG_s104
                                 + 0.0876312618494031 w_k1_PG_s105
                                 + 0.0303376928255835 w_k1_PG_s106
                                 + 0.0321304534105866 w_k1_PG_s107
                                 + 0.0519854482155332 w_k1_PG_s110
                                 + 0.0346589599239235 w_k1_PG_s111
                                 + 0.0716361835817578 w_k1_PG_s112
                                 + 0.0353775604292587 w_k1_PG_s113
                                 + 0.0448673141743355 w_k1_PG_s114
                                 + 0.0775720919183296 w_k1_PG_s115
                                 + 0.0467656616202243 w_k1_PG_s116
                                 + 0.0222620458130654 w_k1_PG_s117 >= 1.19
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
