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
 w_k1_ADM_s0_return_constraint:  1.18441 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.948828 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.17073 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.878446 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.36164 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.993706 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.15991 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.683547 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.16395 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.17939 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.17992 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.98857 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.882712 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.805027 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.917022 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.991552 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.78187 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.48062 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.20472 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.905572 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.12109 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.277 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.30374 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.591997 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.08022 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.923358 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.787097 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.18024 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.824241 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.810421 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.37854 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.932856 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.899813 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.0232 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.669886 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.08944 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.916478 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.4656 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.30887 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.915439 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.3021 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.926937 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.54633 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.96819 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.40805 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.627332 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.5016 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.17497 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.1732 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.907603 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.884204 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.805908 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.19803 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.865228 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.712982 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.42426 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.41429 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.801333 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.02265 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.99864 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.30642 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.614277 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.922358 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.00288 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.643499 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.52733 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.875025 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.957863 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.30178 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.892302 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.00381 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.2414 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.616998 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.13599 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.80247 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.79743 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.48407 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.946067 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.33307 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.815738 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.58975 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.911878 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.7113 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.6573 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.07946 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.19515 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.12361 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.93775 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.987853 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.712238 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.979541 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.906287 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.594229 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.70437 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.29278 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.671516 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.938517 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.12793 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.4644 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.606318 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.997783 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.984864 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.592082 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.60986 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.820826 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.01405 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.28172 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.870707 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.13862 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.25564 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.494772 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.32468 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.718586 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.90818 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.29551 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.960662 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.1543 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.87173 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.28118 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.954471 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.39438 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.830654 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.16764 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.2252 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.972491 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.05307 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.923069 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.805236 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.969201 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.993188 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.676847 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.51107 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.24268 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.912846 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.930478 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.20687 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.26866 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.675412 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.09882 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.02112 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.741499 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.52818 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.924893 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.869507 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.3135 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.925096 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.97381 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.01787 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.665085 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.20193 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.739013 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.51815 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.14928 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.01691 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.26273 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.834296 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.52367 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.897615 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.1381 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.525293 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.24824 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.896391 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.62712 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.734052 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.25894 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.948307 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.79108 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.716399 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.01501 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.20867 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.29056 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.813844 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.29556 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.884754 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.1751 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.29983 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.555887 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.998133 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.797289 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.808728 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.731901 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.39839 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.877933 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.784694 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.23094 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.8182 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.600578 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.17869 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.22982 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.34217 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.13805 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.943475 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.22348 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.00423 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.38875 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.884589 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.21436 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.891836 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.981308 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.12337 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.95956 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.939684 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.991781 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.883613 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.08293 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.83925 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.763666 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.31429 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.16003 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.831212 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.840158 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.04584 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.09114 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.682635 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.904882 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.862712 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.662521 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.05322 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.847733 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.807867 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.26636 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.00167 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.20326 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.05692 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.811925 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.27537 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.915935 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.25409 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.2638 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.993512 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.2035 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.903488 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.45588 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.897288 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.42946 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.729882 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.10734 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.14553 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.24679 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.920458 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.05749 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.798224 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.1627 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.762791 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.733828 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.1016 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.13117 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.733945 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.986834 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.896533 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.30254 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.801274 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.801387 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.972493 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.78971 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.11169 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.78592 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.11911 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.13287 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.816561 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.14417 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.214 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.591446 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.25338 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.882824 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.36899 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.14321 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.12452 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.78102 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.975901 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.42584 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.78933 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.14962 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.432194 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.36601 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.741449 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.50702 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.601201 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.842176 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.567208 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.55028 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.654356 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.85015 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.12427 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.3756 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.797177 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.26231 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.929835 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.07938 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.20921 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.533915 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.11039 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.811622 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.891037 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.812745 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.41563 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.01525 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.739024 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.21056 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.9827 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.535078 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.27762 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.955161 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.49117 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.22972 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.980497 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.13085 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.913659 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.33585 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.869085 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.43015 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.86451 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.111 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.19633 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.09061 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.10412 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.06969 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.902074 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.986601 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.907934 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.737893 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.43832 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.14629 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.864212 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.890023 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.19251 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.13658 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.682301 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.999515 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.936301 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.75994 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.28516 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.798377 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.79571 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.25124 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.995642 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.09877 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.1266 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.749593 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.32703 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.842628 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.46848 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.13965 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.97833 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.14048 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.949746 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.20208 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.96291 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.26922 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.812943 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.1351 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.05916 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.05533 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.07868 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.04435 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.799508 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.03171 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.891428 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.789455 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.12314 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.23175 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.902429 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.971246 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.06771 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.22722 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.897554 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.998861 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.04768 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.877846 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.36147 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.943738 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.09258 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.13877 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.866334 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.02499 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.10424 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.729222 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.13594 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.868592 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.30458 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.0710002682234956 z_k1_s0000
                                 - 0.0410001571206177 z_k1_s0001
                                 - 0.0700002064398698 z_k1_s0002
                                 - 0.134000517379593 z_k1_s0003
                                 - 0.151000854898819 z_k1_s0004
                                 - 0.160000381347968 z_k1_s0005
                                 - 0.110000460177398 z_k1_s0006
                                 - 0.0550001508884309 z_k1_s0007
                                 - 0.0810001852703787 z_k1_s0010
                                 - 0.0640000446581332 z_k1_s0011
                                 - 0.0370000000679975 z_k1_s0012
                                 - 0.0610001713147121 z_k1_s0013
                                 - 0.0669999180015543 z_k1_s0014
                                 - 0.110999948704021 z_k1_s0015
                                 - 0.0629999152104209 z_k1_s0016
                                 - 0.0309999958812975 z_k1_s0017
                                 - 0.0389998124642232 z_k1_s0020
                                 - 0.0259999847760093 z_k1_s0021
                                 - 0.037999901734264 z_k1_s0022
                                 - 0.0520001159518327 z_k1_s0023
                                 - 0.0409999999236766 z_k1_s0024
                                 - 0.0730000712786506 z_k1_s0025
                                 - 0.0339999660147993 z_k1_s0026
                                 - 0.0629998659804537 z_k1_s0027
                                 - 0.0889999505174399 z_k1_s0030
                                 - 0.0439998109923384 z_k1_s0031
                                 - 0.0739997538095934 z_k1_s0032
                                 - 0.0489998014618809 z_k1_s0033
                                 - 0.0469998202940118 z_k1_s0034
                                 - 0.202999102385195 z_k1_s0035
                                 - 0.0809995752475255 z_k1_s0036
                                 - 0.16399905656318 z_k1_s0037
                                 - 0.131000103034224 z_k1_s0040
                                 - 0.177999963985543 z_k1_s0041
                                 - 0.0889999819927713 z_k1_s0042
                                 - 0.28799973199921 z_k1_s0043
                                 - 0.120000345832721 z_k1_s0044
                                 - 0.116999513687412 z_k1_s0045
                                 - 0.0629999070623483 z_k1_s0046
                                 - 0.112000322777207 z_k1_s0047
                                 - 0.21599922448153 z_k1_s0050
                                 - 0.112999875066436 z_k1_s0051
                                 - 0.250999468262877 z_k1_s0052
                                 - 0.273999400462284 z_k1_s0053
                                 - 0.142999856022112 z_k1_s0054
                                 - 0.160000430429723 z_k1_s0055
                                 - 0.102999828214599 z_k1_s0056
                                 - 0.335999148304235 z_k1_s0057
                                 - 0.175999464003829 z_k1_s0060
                                 - 0.0479999528736484 z_k1_s0061
                                 - 0.0499999111850998 z_k1_s0062
                                 - 0.088000072501491 z_k1_s0063
                                 - 0.0799996944563631 z_k1_s0064
                                 - 0.0950000627914016 z_k1_s0065
                                 - 0.0769998931890165 z_k1_s0066
                                 - 0.0669998973320134 z_k1_s0067
                                 - 0.302999475009773 z_k1_s0070
                                 - 0.117000015580044 z_k1_s0071
                                 - 0.135999986702439 z_k1_s0072
                                 - 0.172999457698613 z_k1_s0073
                                 - 0.140999620945122 z_k1_s0074
                                 - 0.167000168807435 z_k1_s0075
                                 - 0.102999950900371 z_k1_s0076
                                 - 0.221000573829052 z_k1_s0077
                                 - 0.039000042465502 z_k1_s0100
                                 - 0.0670000021458458 z_k1_s0101
                                 - 0.0910001253861776 z_k1_s0102
                                 - 0.147999865781836 z_k1_s0103
                                 - 0.084000115741087 z_k1_s0104
                                 - 0.121000279436592 z_k1_s0105
                                 - 0.0960000195612158 z_k1_s0106
                                 - 0.142999971606797 z_k1_s0107
                                 - 0.140000232893004 z_k1_s0110
                                 - 0.151000293148944 z_k1_s0111
                                 - 0.10900024108107 z_k1_s0112
                                 - 0.210999731816397 z_k1_s0113
                                 - 0.254001035363926 z_k1_s0114
                                 - 0.104000213692006 z_k1_s0115
                                 - 0.37600063565551 z_k1_s0116
                                 - 0.43500060284594 z_k1_s0117
                                 - 0.089999859153415 z_k1_s0120
                                 - 0.178000101881279 z_k1_s0121
                                 - 0.0629999870074087 z_k1_s0122
                                 - 0.0859997703021319 z_k1_s0123
                                 - 0.0540000010920672 z_k1_s0124
                                 - 0.173000204817221 z_k1_s0125
                                 - 0.077000041186845 z_k1_s0126
                                 - 0.134999788730122 z_k1_s0127
                                 - 0.183000121849695 z_k1_s0130
                                 - 0.0540000242392436 z_k1_s0131
                                 - 0.182000561943812 z_k1_s0132
                                 - 0.134999805347878 z_k1_s0133
                                 - 0.113000298410123 z_k1_s0134
                                 - 0.122000421566772 z_k1_s0135
                                 - 0.164000315630515 z_k1_s0136
                                 - 0.0680000910272505 z_k1_s0137
                                 - 0.0739997702748477 z_k1_s0140
                                 - 0.141000269386557 z_k1_s0141
                                 - 0.201999819505751 z_k1_s0142
                                 - 0.333998665403535 z_k1_s0143
                                 - 0.129999585263953 z_k1_s0144
                                 - 0.248999450302571 z_k1_s0145
                                 - 0.131999672095442 z_k1_s0146
                                 - 0.11499969137564 z_k1_s0147
                                 - 0.288000233001427 z_k1_s0150
                                 - 0.247999539973962 z_k1_s0151
                                 - 0.214999265526074 z_k1_s0152
                                 - 0.369000546282886 z_k1_s0153
                                 - 0.245999703522436 z_k1_s0154
                                 - 0.157999962660633 z_k1_s0155
                                 - 0.225000429782173 z_k1_s0156
                                 - 0.23299977558828 z_k1_s0157
                                 - 0.10100008671125 z_k1_s0160
                                 - 0.179000600439235 z_k1_s0161
                                 - 0.124000166695065 z_k1_s0162
                                 - 0.0870000777037529 z_k1_s0163
                                 - 0.0730000686962562 z_k1_s0164
                                 - 0.0580000518025019 z_k1_s0165
                                 - 0.280999985235971 z_k1_s0166
                                 - 0.110999658573275 z_k1_s0167
                                 - 0.196999972428809 z_k1_s0170
                                 - 0.16399990800776 z_k1_s0171
                                 - 0.167000016900552 z_k1_s0172
                                 - 0.391000213764712 z_k1_s0173
                                 - 0.225000610961643 z_k1_s0174
                                 - 0.435000677459329 z_k1_s0175
                                 - 0.11099987323454 z_k1_s0176
                                 - 0.199000422823889 z_k1_s0177
                                 - 0.280999946401713 z_k1_s1000
                                 - 0.297998367266407 z_k1_s1001
                                 - 0.174999621424739 z_k1_s1002
                                 - 0.547998161158022 z_k1_s1003
                                 - 0.418999945093593 z_k1_s1004
                                 - 0.317000047929765 z_k1_s1005
                                 - 0.138999754196079 z_k1_s1006
                                 - 0.16599947889303 z_k1_s1007
                                 - 0.0749998902577753 z_k1_s1010
                                 - 0.0859998727702514 z_k1_s1011
                                 - 0.0679998824044143 z_k1_s1012
                                 - 0.146000014656073 z_k1_s1013
                                 - 0.0679998824044143 z_k1_s1014
                                 - 0.19899962668944 z_k1_s1015
                                 - 0.140999263333538 z_k1_s1016
                                 - 0.260999547105181 z_k1_s1017
                                 - 0.379999790333238 z_k1_s1020
                                 - 0.117999762393264 z_k1_s1021
                                 - 0.264999647535393 z_k1_s1022
                                 - 0.0819999097562009 z_k1_s1023
                                 - 0.156999092126002 z_k1_s1024
                                 - 0.0939998131353074 z_k1_s1025
                                 - 0.209000098432896 z_k1_s1026
                                 - 0.119999746289781 z_k1_s1027
                                 - 0.351999376687396 z_k1_s1030
                                 - 0.642000859117039 z_k1_s1031
                                 - 0.185999881972284 z_k1_s1032
                                 - 0.256000056572571 z_k1_s1033
                                 - 0.441999938544855 z_k1_s1034
                                 - 0.277999833115562 z_k1_s1035
                                 - 0.589998435288614 z_k1_s1036
                                 - 0.205999974029502 z_k1_s1037
                                 - 0.300999293413571 z_k1_s1040
                                 - 0.091999844505716 z_k1_s1041
                                 - 0.31599893831175 z_k1_s1042
                                 - 0.224000149004409 z_k1_s1043
                                 - 0.112999874962354 z_k1_s1044
                                 - 0.0869999189063908 z_k1_s1045
                                 - 0.108999881722975 z_k1_s1046
                                 - 0.0769999358079434 z_k1_s1047
                                 - 0.0539999883940375 z_k1_s1050
                                 - 0.211999578036195 z_k1_s1051
                                 - 0.0679998912851703 z_k1_s1052
                                 - 0.108000353187731 z_k1_s1053
                                 - 0.185999771824079 z_k1_s1054
                                 - 0.127000066804409 z_k1_s1055
                                 - 0.141999593080961 z_k1_s1056
                                 - 0.0439999905432898 z_k1_s1057
                                 - 0.381000996669853 z_k1_s1060
                                 - 0.290999757676941 z_k1_s1061
                                 - 0.113000064957718 z_k1_s1062
                                 - 0.430000702299296 z_k1_s1063
                                 - 0.0919999994593792 z_k1_s1064
                                 - 0.24699974821376 z_k1_s1065
                                 - 0.321000170674579 z_k1_s1066
                                 - 0.360999976004737 z_k1_s1067
                                 - 0.420998754287736 z_k1_s1070
                                 - 0.379000279463367 z_k1_s1071
                                 - 0.444000136286891 z_k1_s1072
                                 - 0.470998909459689 z_k1_s1073
                                 - 0.417001363114093 z_k1_s1074
                                 - 0.212999902252488 z_k1_s1075
                                 - 0.270000148128518 z_k1_s1076
                                 - 0.833999277228036 z_k1_s1077
                                 - 0.327998899998096 z_k1_s1100
                                 - 1.09099977701357 z_k1_s1101
                                 - 1.15499763050139 z_k1_s1102
                                 - 0.810996140539863 z_k1_s1103
                                 - 0.48699858283073 z_k1_s1104
                                 - 0.902000796384343 z_k1_s1105
                                 - 1.35199373732369 z_k1_s1106
                                 - 0.821994794548247 z_k1_s1107
                                 - 1.72100053988678 z_k1_s1110
                                 - 1.7269998480367 z_k1_s1111
                                 - 2.29399182702856 z_k1_s1112
                                 - 1.81999974636498 z_k1_s1113
                                 - 2.48899058591001 z_k1_s1114
                                 - 1.59300113668245 z_k1_s1115
                                 - 3.74300319491021 z_k1_s1116
                                 - 5.85699453343213 z_k1_s1117
                                 - 1.14099964669524 z_k1_s1120
                                 - 0.478000205301152 z_k1_s1121
                                 - 0.713999750809273 z_k1_s1122
                                 - 0.247000366762915 z_k1_s1123
                                 - 0.290000412065758 z_k1_s1124
                                 - 0.457000428206915 z_k1_s1125
                                 - 0.524002640147757 z_k1_s1126
                                 - 0.730001195234606 z_k1_s1127
                                 - 0.728999275669214 z_k1_s1130
                                 - 0.610999021623892 z_k1_s1131
                                 - 0.523999414074521 z_k1_s1132
                                 - 1.25500194572023 z_k1_s1133
                                 - 1.75099741767344 z_k1_s1134
                                 - 0.887998609714555 z_k1_s1135
                                 - 1.08199617901963 z_k1_s1136
                                 - 0.59599914057049 z_k1_s1137
                                 - 1.44400009329847 z_k1_s1140
                                 - 0.385000044710198 z_k1_s1141
                                 - 0.691001005876726 z_k1_s1142
                                 - 0.340000211162345 z_k1_s1143
                                 - 1.0349978240015 z_k1_s1144
                                 - 0.446000076115203 z_k1_s1145
                                 - 0.750999682339933 z_k1_s1146
                                 - 0.416000187083302 z_k1_s1147
                                 - 0.568000374022198 z_k1_s1150
                                 - 0.589000203906796 z_k1_s1151
                                 - 0.523999921978167 z_k1_s1152
                                 - 0.609000290654067 z_k1_s1153
                                 - 1.74900197024804 z_k1_s1154
                                 - 0.770997205906438 z_k1_s1155
                                 - 0.650999950423263 z_k1_s1156
                                 - 1.06900032684102 z_k1_s1157
                                 - 0.958001826169531 z_k1_s1160
                                 - 2.6109969002052 z_k1_s1161
                                 - 0.781999539790151 z_k1_s1162
                                 - 1.2859962070374 z_k1_s1163
                                 - 0.740999777731964 z_k1_s1164
                                 - 1.12399615033008 z_k1_s1165
                                 - 1.23000215073971 z_k1_s1166
                                 - 0.556999667452067 z_k1_s1167
                                 - 0.396000032249776 z_k1_s1170
                                 - 0.872999171443666 z_k1_s1171
                                 - 0.294999896913068 z_k1_s1172
                                 - 0.40000004244826 z_k1_s1173
                                 - 0.78699773042604 z_k1_s1174
                                 - 0.383000121279723 z_k1_s1175
                                 - 0.494999429437113 z_k1_s1176
                                 - 1.2580019856731 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.12551 w_k1_ADM_s000
                                 + 0.657763 w_k1_AXP_s000
                                 + 1.08098 w_k1_DIS_s000
                                 + 1.03947 w_k1_FDX_s000
                                 + 0.946638 w_k1_IBM_s000
                                 + 0.842621 w_k1_INTC_s000
                                 + 0.952958 w_k1_KO_s000
                                 + 0.60927 w_k1_MAT_s000
                                 + 0.930876 w_k1_MSFT_s000
                                 + 0.964179 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.68295 w_k1_ADM_s000
                                 + 1.80795 w_k1_AXP_s000
                                 + 2.08133 w_k1_DIS_s000
                                 + 1.57791 w_k1_FDX_s000 + 1.2825 w_k1_IBM_s000
                                 + 1.13008 w_k1_INTC_s000
                                 + 1.51157 w_k1_KO_s000 + 2.03682 w_k1_MAT_s000
                                 + 1.54299 w_k1_MSFT_s000
                                 + 1.36322 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.739224 w_k1_ADM_s000
                                 + 0.670243 w_k1_AXP_s000
                                 + 0.632767 w_k1_DIS_s000
                                 + 0.79757 w_k1_FDX_s000
                                 + 0.742212 w_k1_IBM_s000
                                 + 0.829027 w_k1_INTC_s000
                                 + 0.706155 w_k1_KO_s000
                                 + 0.472328 w_k1_MAT_s000
                                 + 0.760527 w_k1_MSFT_s000
                                 + 0.803142 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.0739 w_k1_ADM_s000
                                 + 0.760969 w_k1_AXP_s000
                                 + 0.809738 w_k1_DIS_s000
                                 + 0.904952 w_k1_FDX_s000
                                 + 0.788031 w_k1_IBM_s000
                                 + 1.04065 w_k1_INTC_s000
                                 + 0.871419 w_k1_KO_s000
                                 + 0.789245 w_k1_MAT_s000
                                 + 0.945855 w_k1_MSFT_s000
                                 + 1.03404 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.19094 w_k1_ADM_s000
                                 + 1.01863 w_k1_AXP_s000
                                 + 0.923346 w_k1_DIS_s000
                                 + 1.09109 w_k1_FDX_s000
                                 + 0.722718 w_k1_IBM_s000
                                 + 0.88258 w_k1_INTC_s000
                                 + 0.70163 w_k1_KO_s000
                                 + 0.535756 w_k1_MAT_s000
                                 + 0.868588 w_k1_MSFT_s000
                                 + 0.861593 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.968215 w_k1_ADM_s000
                                 + 1.11372 w_k1_AXP_s000
                                 + 1.57323 w_k1_DIS_s000
                                 + 1.29019 w_k1_FDX_s000
                                 + 1.12434 w_k1_IBM_s000
                                 + 1.2863 w_k1_INTC_s000 + 1.08692 w_k1_KO_s000
                                 + 1.19701 w_k1_MAT_s000
                                 + 1.22439 w_k1_MSFT_s000
                                 + 1.18496 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.15198 w_k1_ADM_s000
                                 + 1.16853 w_k1_AXP_s000
                                 + 0.958077 w_k1_DIS_s000
                                 + 0.983635 w_k1_FDX_s000
                                 + 1.33402 w_k1_IBM_s000
                                 + 0.899972 w_k1_INTC_s000
                                 + 1.14173 w_k1_KO_s000 + 1.07227 w_k1_MAT_s000
                                 + 0.773083 w_k1_MSFT_s000
                                 + 1.00589 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.7292 w_k1_ADM_s000
                                 + 1.54575 w_k1_AXP_s000
                                 + 1.22623 w_k1_DIS_s000
                                 + 1.36037 w_k1_FDX_s000
                                 + 1.24579 w_k1_IBM_s000
                                 + 1.09557 w_k1_INTC_s000
                                 + 1.03763 w_k1_KO_s000
                                 + 0.967171 w_k1_MAT_s000
                                 + 0.951391 w_k1_MSFT_s000
                                 + 1.1822 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.923791 w_k1_ADM_s001
                                 + 1.4082 w_k1_AXP_s001
                                 + 0.871636 w_k1_DIS_s001
                                 + 0.985136 w_k1_FDX_s001
                                 + 1.13137 w_k1_IBM_s001
                                 + 1.00862 w_k1_INTC_s001
                                 + 0.787961 w_k1_KO_s001
                                 + 1.28375 w_k1_MAT_s001
                                 + 1.0293 w_k1_MSFT_s001
                                 + 0.903171 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.932588 w_k1_ADM_s001
                                 + 0.817515 w_k1_AXP_s001
                                 + 0.720066 w_k1_DIS_s001
                                 + 0.862597 w_k1_FDX_s001
                                 + 0.771424 w_k1_IBM_s001
                                 + 0.904136 w_k1_INTC_s001
                                 + 0.756515 w_k1_KO_s001
                                 + 0.787026 w_k1_MAT_s001
                                 + 0.888789 w_k1_MSFT_s001
                                 + 0.91614 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.44323 w_k1_ADM_s001
                                 + 0.666981 w_k1_AXP_s001
                                 + 0.850621 w_k1_DIS_s001
                                 + 1.02926 w_k1_FDX_s001
                                 + 0.610605 w_k1_IBM_s001
                                 + 0.968627 w_k1_INTC_s001
                                 + 0.62707 w_k1_KO_s001
                                 + 0.444094 w_k1_MAT_s001
                                 + 1.06795 w_k1_MSFT_s001
                                 + 0.963359 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.34036 w_k1_ADM_s001
                                 + 1.3303 w_k1_AXP_s001 + 1.1206 w_k1_DIS_s001
                                 + 1.4079 w_k1_FDX_s001
                                 + 0.838766 w_k1_IBM_s001
                                 + 1.13367 w_k1_INTC_s001
                                 + 0.940559 w_k1_KO_s001
                                 + 0.583071 w_k1_MAT_s001
                                 + 1.19396 w_k1_MSFT_s001
                                 + 1.11278 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.06597 w_k1_ADM_s001
                                 + 0.825491 w_k1_AXP_s001
                                 + 0.652961 w_k1_DIS_s001
                                 + 0.722636 w_k1_FDX_s001 + 0.988 w_k1_IBM_s001
                                 + 0.730916 w_k1_INTC_s001
                                 + 0.721063 w_k1_KO_s001
                                 + 0.778423 w_k1_MAT_s001
                                 + 0.719617 w_k1_MSFT_s001
                                 + 0.705418 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.96141 w_k1_ADM_s001
                                 + 1.4168 w_k1_AXP_s001 + 1.4404 w_k1_DIS_s001
                                 + 1.26042 w_k1_FDX_s001
                                 + 1.43221 w_k1_IBM_s001
                                 + 0.959938 w_k1_INTC_s001
                                 + 1.18928 w_k1_KO_s001 + 1.29591 w_k1_MAT_s001
                                 + 1.08281 w_k1_MSFT_s001
                                 + 1.23807 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.23848 w_k1_ADM_s001
                                 + 0.979272 w_k1_AXP_s001
                                 + 1.10857 w_k1_DIS_s001
                                 + 0.883617 w_k1_FDX_s001
                                 + 1.14729 w_k1_IBM_s001
                                 + 0.665494 w_k1_INTC_s001
                                 + 0.89888 w_k1_KO_s001
                                 + 0.711741 w_k1_MAT_s001
                                 + 0.896813 w_k1_MSFT_s001
                                 + 0.92523 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.22625 w_k1_ADM_s001
                                 + 1.25044 w_k1_AXP_s001
                                 + 0.733635 w_k1_DIS_s001
                                 + 0.890497 w_k1_FDX_s001
                                 + 0.866854 w_k1_IBM_s001
                                 + 0.757835 w_k1_INTC_s001
                                 + 0.627864 w_k1_KO_s001
                                 + 0.952537 w_k1_MAT_s001
                                 + 0.697939 w_k1_MSFT_s001
                                 + 0.887404 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.15998 w_k1_ADM_s002
                                 + 0.936951 w_k1_AXP_s002
                                 + 0.809267 w_k1_DIS_s002
                                 + 1.00238 w_k1_FDX_s002
                                 + 0.791653 w_k1_IBM_s002
                                 + 0.932474 w_k1_INTC_s002
                                 + 0.733619 w_k1_KO_s002
                                 + 0.76208 w_k1_MAT_s002
                                 + 0.885777 w_k1_MSFT_s002
                                 + 0.966361 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.816328 w_k1_ADM_s002
                                 + 0.823882 w_k1_AXP_s002
                                 + 0.579772 w_k1_DIS_s002
                                 + 0.665883 w_k1_FDX_s002
                                 + 0.940007 w_k1_IBM_s002
                                 + 0.710924 w_k1_INTC_s002
                                 + 0.851151 w_k1_KO_s002
                                 + 1.00573 w_k1_MAT_s002
                                 + 0.582021 w_k1_MSFT_s002
                                 + 0.82205 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.07751 w_k1_ADM_s002
                                 + 1.78705 w_k1_AXP_s002
                                 + 1.47479 w_k1_DIS_s002
                                 + 1.32107 w_k1_FDX_s002
                                 + 1.22084 w_k1_IBM_s002
                                 + 1.63803 w_k1_INTC_s002 + 1.3612 w_k1_KO_s002
                                 + 1.5459 w_k1_MAT_s002
                                 + 1.27183 w_k1_MSFT_s002
                                 + 1.26704 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.174 w_k1_ADM_s002
                                 + 1.19752 w_k1_AXP_s002
                                 + 0.930027 w_k1_DIS_s002
                                 + 0.952416 w_k1_FDX_s002
                                 + 1.23386 w_k1_IBM_s002
                                 + 1.04914 w_k1_INTC_s002
                                 + 1.03417 w_k1_KO_s002 + 1.07995 w_k1_MAT_s002
                                 + 0.861264 w_k1_MSFT_s002
                                 + 0.999356 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.959338 w_k1_ADM_s002
                                 + 0.819771 w_k1_AXP_s002
                                 + 0.982259 w_k1_DIS_s002
                                 + 0.903329 w_k1_FDX_s002
                                 + 0.851724 w_k1_IBM_s002
                                 + 1.00905 w_k1_INTC_s002
                                 + 0.950737 w_k1_KO_s002
                                 + 0.887804 w_k1_MAT_s002
                                 + 0.942355 w_k1_MSFT_s002
                                 + 0.982873 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.20424 w_k1_ADM_s002
                                 + 1.0759 w_k1_AXP_s002 + 1.2901 w_k1_DIS_s002
                                 + 1.13443 w_k1_FDX_s002
                                 + 0.988041 w_k1_IBM_s002
                                 + 1.03283 w_k1_INTC_s002
                                 + 1.07003 w_k1_KO_s002 + 1.05504 w_k1_MAT_s002
                                 + 1.0622 w_k1_MSFT_s002 + 1.09908 w_k1_PG_s002
                                 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.97077 w_k1_ADM_s002
                                 + 1.26272 w_k1_AXP_s002 + 1.2128 w_k1_DIS_s002
                                 + 1.12663 w_k1_FDX_s002
                                 + 1.38556 w_k1_IBM_s002
                                 + 1.33893 w_k1_INTC_s002 + 1.2219 w_k1_KO_s002
                                 + 1.42925 w_k1_MAT_s002
                                 + 1.09737 w_k1_MSFT_s002
                                 + 1.18866 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.809751 w_k1_ADM_s002
                                 + 0.750317 w_k1_AXP_s002
                                 + 0.668124 w_k1_DIS_s002
                                 + 0.802997 w_k1_FDX_s002
                                 + 0.72208 w_k1_IBM_s002
                                 + 0.937439 w_k1_INTC_s002
                                 + 0.744764 w_k1_KO_s002
                                 + 0.689897 w_k1_MAT_s002
                                 + 0.778938 w_k1_MSFT_s002
                                 + 0.874272 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.10405 w_k1_ADM_s003
                                 + 1.17279 w_k1_AXP_s003
                                 + 1.16403 w_k1_DIS_s003
                                 + 1.27033 w_k1_FDX_s003
                                 + 0.788307 w_k1_IBM_s003
                                 + 0.973933 w_k1_INTC_s003
                                 + 0.908672 w_k1_KO_s003
                                 + 0.873668 w_k1_MAT_s003
                                 + 1.14871 w_k1_MSFT_s003
                                 + 1.23584 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.3506 w_k1_ADM_s003
                                 + 1.1492 w_k1_AXP_s003 + 1.23055 w_k1_DIS_s003
                                 + 1.12014 w_k1_FDX_s003
                                 + 1.32245 w_k1_IBM_s003
                                 + 1.2001 w_k1_INTC_s003 + 1.15853 w_k1_KO_s003
                                 + 0.983663 w_k1_MAT_s003
                                 + 1.22535 w_k1_MSFT_s003
                                 + 1.19739 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.15252 w_k1_ADM_s003
                                 + 0.956402 w_k1_AXP_s003
                                 + 1.08312 w_k1_DIS_s003 + 1.1518 w_k1_FDX_s003
                                 + 0.762904 w_k1_IBM_s003
                                 + 1.05149 w_k1_INTC_s003
                                 + 0.904243 w_k1_KO_s003
                                 + 0.984295 w_k1_MAT_s003
                                 + 1.04306 w_k1_MSFT_s003
                                 + 0.980443 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.585041 w_k1_ADM_s003
                                 + 0.675924 w_k1_AXP_s003
                                 + 0.637173 w_k1_DIS_s003
                                 + 0.767411 w_k1_FDX_s003
                                 + 1.24172 w_k1_IBM_s003
                                 + 1.18065 w_k1_INTC_s003
                                 + 0.832736 w_k1_KO_s003
                                 + 0.979356 w_k1_MAT_s003
                                 + 0.791971 w_k1_MSFT_s003
                                 + 0.896185 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.19379 w_k1_ADM_s003
                                 + 1.14071 w_k1_AXP_s003
                                 + 1.59634 w_k1_DIS_s003
                                 + 1.17184 w_k1_FDX_s003
                                 + 1.49577 w_k1_IBM_s003
                                 + 1.1136 w_k1_INTC_s003 + 1.22414 w_k1_KO_s003
                                 + 1.02939 w_k1_MAT_s003
                                 + 1.15873 w_k1_MSFT_s003 + 1.2171 w_k1_PG_s003
                                 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.98318 w_k1_ADM_s003
                                 + 0.925439 w_k1_AXP_s003
                                 + 0.857176 w_k1_DIS_s003
                                 + 0.977348 w_k1_FDX_s003
                                 + 0.841508 w_k1_IBM_s003
                                 + 1.05568 w_k1_INTC_s003
                                 + 0.819904 w_k1_KO_s003
                                 + 0.624732 w_k1_MAT_s003
                                 + 1.00076 w_k1_MSFT_s003
                                 + 0.951122 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.11856 w_k1_ADM_s003
                                 + 0.95934 w_k1_AXP_s003
                                 + 0.998848 w_k1_DIS_s003
                                 + 0.935508 w_k1_FDX_s003
                                 + 1.34532 w_k1_IBM_s003
                                 + 0.925366 w_k1_INTC_s003
                                 + 1.01031 w_k1_KO_s003 + 1.02941 w_k1_MAT_s003
                                 + 1.00987 w_k1_MSFT_s003
                                 + 0.993557 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.818898 w_k1_ADM_s003
                                 + 1.23299 w_k1_AXP_s003
                                 + 0.797563 w_k1_DIS_s003
                                 + 0.971438 w_k1_FDX_s003
                                 + 0.996951 w_k1_IBM_s003
                                 + 1.16922 w_k1_INTC_s003
                                 + 0.782357 w_k1_KO_s003
                                 + 1.08935 w_k1_MAT_s003
                                 + 0.892115 w_k1_MSFT_s003
                                 + 0.940967 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.28661 w_k1_ADM_s004
                                 + 1.36438 w_k1_AXP_s004
                                 + 1.42516 w_k1_DIS_s004
                                 + 1.48761 w_k1_FDX_s004
                                 + 0.953188 w_k1_IBM_s004
                                 + 1.46509 w_k1_INTC_s004
                                 + 1.30347 w_k1_KO_s004 + 1.11976 w_k1_MAT_s004
                                 + 1.08435 w_k1_MSFT_s004
                                 + 1.34258 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.01401 w_k1_ADM_s004
                                 + 1.09456 w_k1_AXP_s004
                                 + 1.07405 w_k1_DIS_s004
                                 + 1.15715 w_k1_FDX_s004
                                 + 1.07003 w_k1_IBM_s004
                                 + 1.06779 w_k1_INTC_s004
                                 + 1.08141 w_k1_KO_s004
                                 + 0.933479 w_k1_MAT_s004
                                 + 0.932827 w_k1_MSFT_s004
                                 + 1.15683 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.810461 w_k1_ADM_s004
                                 + 0.803026 w_k1_AXP_s004
                                 + 0.643297 w_k1_DIS_s004
                                 + 0.701466 w_k1_FDX_s004
                                 + 0.995342 w_k1_IBM_s004
                                 + 0.726577 w_k1_INTC_s004
                                 + 0.88647 w_k1_KO_s004
                                 + 0.931821 w_k1_MAT_s004
                                 + 0.640838 w_k1_MSFT_s004
                                 + 1.02717 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.882676 w_k1_ADM_s004
                                 + 0.971474 w_k1_AXP_s004
                                 + 0.842803 w_k1_DIS_s004
                                 + 1.06304 w_k1_FDX_s004
                                 + 0.812995 w_k1_IBM_s004
                                 + 1.09556 w_k1_INTC_s004
                                 + 0.938613 w_k1_KO_s004
                                 + 0.725859 w_k1_MAT_s004
                                 + 1.01569 w_k1_MSFT_s004
                                 + 1.05538 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.970776 w_k1_ADM_s004
                                 + 1.46686 w_k1_AXP_s004
                                 + 1.64492 w_k1_DIS_s004 + 1.303 w_k1_FDX_s004
                                 + 1.63142 w_k1_IBM_s004
                                 + 1.25947 w_k1_INTC_s004
                                 + 1.49409 w_k1_KO_s004 + 1.23926 w_k1_MAT_s004
                                 + 1.34934 w_k1_MSFT_s004
                                 + 1.17451 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.599756 w_k1_ADM_s004
                                 + 0.764132 w_k1_AXP_s004
                                 + 0.675003 w_k1_DIS_s004
                                 + 0.934529 w_k1_FDX_s004
                                 + 0.569487 w_k1_IBM_s004
                                 + 0.957424 w_k1_INTC_s004
                                 + 0.705172 w_k1_KO_s004
                                 + 0.508239 w_k1_MAT_s004
                                 + 0.800457 w_k1_MSFT_s004
                                 + 0.84677 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.17949 w_k1_ADM_s004
                                 + 1.21461 w_k1_AXP_s004
                                 + 0.880058 w_k1_DIS_s004
                                 + 0.965577 w_k1_FDX_s004
                                 + 1.59113 w_k1_IBM_s004
                                 + 0.929427 w_k1_INTC_s004
                                 + 1.27101 w_k1_KO_s004 + 1.69714 w_k1_MAT_s004
                                 + 0.953848 w_k1_MSFT_s004
                                 + 1.11468 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.746147 w_k1_ADM_s004
                                 + 0.943487 w_k1_AXP_s004
                                 + 1.10734 w_k1_DIS_s004
                                 + 0.99948 w_k1_FDX_s004
                                 + 1.28601 w_k1_IBM_s004
                                 + 0.769477 w_k1_INTC_s004
                                 + 1.15284 w_k1_KO_s004 + 1.02858 w_k1_MAT_s004
                                 + 0.813012 w_k1_MSFT_s004
                                 + 0.914338 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.94612 w_k1_ADM_s005
                                 + 0.774958 w_k1_AXP_s005
                                 + 0.683201 w_k1_DIS_s005
                                 + 0.78472 w_k1_FDX_s005
                                 + 0.716363 w_k1_IBM_s005
                                 + 0.919335 w_k1_INTC_s005
                                 + 0.778077 w_k1_KO_s005
                                 + 0.885351 w_k1_MAT_s005
                                 + 0.853718 w_k1_MSFT_s005
                                 + 0.809386 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.07461 w_k1_ADM_s005
                                 + 1.03384 w_k1_AXP_s005
                                 + 1.57896 w_k1_DIS_s005
                                 + 1.28283 w_k1_FDX_s005
                                 + 0.919987 w_k1_IBM_s005
                                 + 1.25033 w_k1_INTC_s005
                                 + 1.12196 w_k1_KO_s005 + 1.03654 w_k1_MAT_s005
                                 + 1.25836 w_k1_MSFT_s005
                                 + 1.05109 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.22987 w_k1_ADM_s005
                                 + 0.995817 w_k1_AXP_s005
                                 + 1.08716 w_k1_DIS_s005
                                 + 1.11761 w_k1_FDX_s005
                                 + 0.82155 w_k1_IBM_s005
                                 + 1.20973 w_k1_INTC_s005
                                 + 0.874923 w_k1_KO_s005
                                 + 0.684319 w_k1_MAT_s005
                                 + 1.07388 w_k1_MSFT_s005
                                 + 0.88637 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.990229 w_k1_ADM_s005
                                 + 1.15733 w_k1_AXP_s005
                                 + 0.802924 w_k1_DIS_s005
                                 + 1.02628 w_k1_FDX_s005
                                 + 0.876788 w_k1_IBM_s005
                                 + 0.978139 w_k1_INTC_s005
                                 + 0.726988 w_k1_KO_s005
                                 + 0.899883 w_k1_MAT_s005
                                 + 0.896344 w_k1_MSFT_s005
                                 + 0.858408 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.49635 w_k1_ADM_s005
                                 + 1.56881 w_k1_AXP_s005
                                 + 1.75866 w_k1_DIS_s005
                                 + 1.35415 w_k1_FDX_s005
                                 + 1.59611 w_k1_IBM_s005
                                 + 1.37533 w_k1_INTC_s005
                                 + 1.33417 w_k1_KO_s005 + 1.83247 w_k1_MAT_s005
                                 + 1.27503 w_k1_MSFT_s005
                                 + 1.14452 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.35141 w_k1_ADM_s005
                                 + 1.6298 w_k1_AXP_s005 + 1.25874 w_k1_DIS_s005
                                 + 1.13673 w_k1_FDX_s005
                                 + 1.15035 w_k1_IBM_s005
                                 + 1.01416 w_k1_INTC_s005
                                 + 1.05862 w_k1_KO_s005 + 1.40462 w_k1_MAT_s005
                                 + 0.93816 w_k1_MSFT_s005
                                 + 1.05199 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.954454 w_k1_ADM_s005
                                 + 1.15195 w_k1_AXP_s005
                                 + 1.36592 w_k1_DIS_s005
                                 + 1.01357 w_k1_FDX_s005
                                 + 1.39829 w_k1_IBM_s005
                                 + 1.07944 w_k1_INTC_s005
                                 + 1.30067 w_k1_KO_s005 + 1.45899 w_k1_MAT_s005
                                 + 0.966907 w_k1_MSFT_s005
                                 + 1.11051 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.842716 w_k1_ADM_s005
                                 + 0.967397 w_k1_AXP_s005
                                 + 0.923829 w_k1_DIS_s005
                                 + 0.84591 w_k1_FDX_s005
                                 + 1.23178 w_k1_IBM_s005
                                 + 1.13579 w_k1_INTC_s005
                                 + 1.04212 w_k1_KO_s005 + 1.57845 w_k1_MAT_s005
                                 + 0.837387 w_k1_MSFT_s005
                                 + 0.887461 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.08901 w_k1_ADM_s006
                                 + 0.828181 w_k1_AXP_s006
                                 + 0.867227 w_k1_DIS_s006
                                 + 0.873865 w_k1_FDX_s006
                                 + 0.696964 w_k1_IBM_s006
                                 + 0.844489 w_k1_INTC_s006
                                 + 0.903463 w_k1_KO_s006
                                 + 0.768404 w_k1_MAT_s006
                                 + 0.783355 w_k1_MSFT_s006
                                 + 0.886553 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.02546 w_k1_ADM_s006
                                 + 1.41139 w_k1_AXP_s006
                                 + 1.66387 w_k1_DIS_s006
                                 + 1.27523 w_k1_FDX_s006
                                 + 1.53968 w_k1_IBM_s006
                                 + 1.30677 w_k1_INTC_s006
                                 + 1.57839 w_k1_KO_s006 + 1.84694 w_k1_MAT_s006
                                 + 1.09316 w_k1_MSFT_s006
                                 + 1.18422 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.06582 w_k1_ADM_s006
                                 + 1.14249 w_k1_AXP_s006
                                 + 0.98241 w_k1_DIS_s006
                                 + 1.02761 w_k1_FDX_s006
                                 + 1.20213 w_k1_IBM_s006
                                 + 1.01235 w_k1_INTC_s006
                                 + 1.29326 w_k1_KO_s006 + 1.52577 w_k1_MAT_s006
                                 + 1.17301 w_k1_MSFT_s006
                                 + 1.19021 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.07291 w_k1_ADM_s006
                                 + 1.3249 w_k1_AXP_s006 + 1.4138 w_k1_DIS_s006
                                 + 1.18058 w_k1_FDX_s006
                                 + 1.05725 w_k1_IBM_s006
                                 + 1.10232 w_k1_INTC_s006
                                 + 1.13464 w_k1_KO_s006 + 1.31239 w_k1_MAT_s006
                                 + 1.1898 w_k1_MSFT_s006 + 1.00041 w_k1_PG_s006
                                 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.02979 w_k1_ADM_s006
                                 + 1.06422 w_k1_AXP_s006
                                 + 1.17381 w_k1_DIS_s006
                                 + 1.04631 w_k1_FDX_s006
                                 + 1.08563 w_k1_IBM_s006
                                 + 0.930865 w_k1_INTC_s006
                                 + 1.08115 w_k1_KO_s006 + 1.40865 w_k1_MAT_s006
                                 + 1.00598 w_k1_MSFT_s006
                                 + 1.03811 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.791056 w_k1_ADM_s006
                                 + 0.74235 w_k1_AXP_s006
                                 + 0.781856 w_k1_DIS_s006
                                 + 0.787203 w_k1_FDX_s006
                                 + 0.93544 w_k1_IBM_s006
                                 + 0.854118 w_k1_INTC_s006
                                 + 0.953783 w_k1_KO_s006
                                 + 0.840642 w_k1_MAT_s006
                                 + 0.842229 w_k1_MSFT_s006
                                 + 0.867991 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.10081 w_k1_ADM_s006
                                 + 0.971921 w_k1_AXP_s006
                                 + 0.929645 w_k1_DIS_s006
                                 + 0.827137 w_k1_FDX_s006
                                 + 1.38017 w_k1_IBM_s006
                                 + 1.02785 w_k1_INTC_s006
                                 + 1.08745 w_k1_KO_s006 + 1.09193 w_k1_MAT_s006
                                 + 0.872522 w_k1_MSFT_s006
                                 + 0.988607 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.12763 w_k1_ADM_s006
                                 + 1.22819 w_k1_AXP_s006
                                 + 1.23613 w_k1_DIS_s006
                                 + 1.29885 w_k1_FDX_s006
                                 + 0.804966 w_k1_IBM_s006
                                 + 1.17531 w_k1_INTC_s006
                                 + 0.989287 w_k1_KO_s006
                                 + 0.718514 w_k1_MAT_s006
                                 + 1.13298 w_k1_MSFT_s006
                                 + 1.04091 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.982388 w_k1_ADM_s007
                                 + 1.02698 w_k1_AXP_s007
                                 + 1.14512 w_k1_DIS_s007
                                 + 1.05095 w_k1_FDX_s007
                                 + 1.03002 w_k1_IBM_s007
                                 + 1.09768 w_k1_INTC_s007
                                 + 1.14461 w_k1_KO_s007
                                 + 0.793221 w_k1_MAT_s007
                                 + 0.974419 w_k1_MSFT_s007
                                 + 1.16556 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.01299 w_k1_ADM_s007
                                 + 0.747886 w_k1_AXP_s007
                                 + 0.650295 w_k1_DIS_s007
                                 + 0.736372 w_k1_FDX_s007
                                 + 1.11238 w_k1_IBM_s007
                                 + 0.692143 w_k1_INTC_s007
                                 + 0.838954 w_k1_KO_s007
                                 + 1.23902 w_k1_MAT_s007
                                 + 0.695764 w_k1_MSFT_s007
                                 + 1.01188 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.697571 w_k1_ADM_s007
                                 + 0.546488 w_k1_AXP_s007
                                 + 0.601511 w_k1_DIS_s007
                                 + 0.7134 w_k1_FDX_s007
                                 + 0.741283 w_k1_IBM_s007
                                 + 0.758664 w_k1_INTC_s007
                                 + 0.778031 w_k1_KO_s007
                                 + 0.722276 w_k1_MAT_s007
                                 + 0.736582 w_k1_MSFT_s007
                                 + 0.737303 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.02438 w_k1_ADM_s007
                                 + 1.01884 w_k1_AXP_s007
                                 + 1.04987 w_k1_DIS_s007
                                 + 0.916138 w_k1_FDX_s007
                                 + 1.55726 w_k1_IBM_s007
                                 + 1.09677 w_k1_INTC_s007
                                 + 1.32148 w_k1_KO_s007 + 1.66901 w_k1_MAT_s007
                                 + 0.957337 w_k1_MSFT_s007
                                 + 1.41858 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.1678 w_k1_ADM_s007
                                 + 1.41 w_k1_AXP_s007 + 0.9741 w_k1_DIS_s007
                                 + 1.05395 w_k1_FDX_s007 + 1.3336 w_k1_IBM_s007
                                 + 0.893755 w_k1_INTC_s007
                                 + 1.20491 w_k1_KO_s007 + 2.61143 w_k1_MAT_s007
                                 + 0.933008 w_k1_MSFT_s007
                                 + 1.28345 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.43961 w_k1_ADM_s007
                                 + 1.18195 w_k1_AXP_s007
                                 + 1.43675 w_k1_DIS_s007 + 1.3228 w_k1_FDX_s007
                                 + 0.940277 w_k1_IBM_s007
                                 + 1.16815 w_k1_INTC_s007
                                 + 1.17272 w_k1_KO_s007 + 1.11621 w_k1_MAT_s007
                                 + 1.1324 w_k1_MSFT_s007 + 1.21495 w_k1_PG_s007
                                 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.09605 w_k1_ADM_s007
                                 + 1.21817 w_k1_AXP_s007
                                 + 0.906821 w_k1_DIS_s007
                                 + 0.961131 w_k1_FDX_s007
                                 + 0.932707 w_k1_IBM_s007
                                 + 0.920918 w_k1_INTC_s007
                                 + 0.961396 w_k1_KO_s007
                                 + 1.51829 w_k1_MAT_s007
                                 + 0.830858 w_k1_MSFT_s007
                                 + 1.15311 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.08411 w_k1_ADM_s007
                                 + 0.872713 w_k1_AXP_s007
                                 + 0.979654 w_k1_DIS_s007
                                 + 1.10182 w_k1_FDX_s007
                                 + 0.692467 w_k1_IBM_s007
                                 + 0.923801 w_k1_INTC_s007
                                 + 0.77433 w_k1_KO_s007
                                 + 0.833463 w_k1_MAT_s007
                                 + 0.955824 w_k1_MSFT_s007
                                 + 1.00826 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.829898 w_k1_ADM_s010
                                 + 0.670053 w_k1_AXP_s010
                                 + 1.42203 w_k1_DIS_s010
                                 + 1.01046 w_k1_FDX_s010
                                 + 0.877836 w_k1_IBM_s010
                                 + 0.788055 w_k1_INTC_s010
                                 + 0.933219 w_k1_KO_s010
                                 + 0.749045 w_k1_MAT_s010
                                 + 0.986118 w_k1_MSFT_s010
                                 + 0.974052 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.32974 w_k1_ADM_s010
                                 + 1.01829 w_k1_AXP_s010
                                 + 1.12521 w_k1_DIS_s010
                                 + 1.18664 w_k1_FDX_s010
                                 + 0.731973 w_k1_IBM_s010
                                 + 0.834655 w_k1_INTC_s010
                                 + 0.830219 w_k1_KO_s010
                                 + 0.818937 w_k1_MAT_s010
                                 + 1.01476 w_k1_MSFT_s010
                                 + 1.08131 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.13692 w_k1_ADM_s010
                                 + 1.22106 w_k1_AXP_s010
                                 + 1.48698 w_k1_DIS_s010
                                 + 1.15378 w_k1_FDX_s010
                                 + 1.59032 w_k1_IBM_s010
                                 + 1.11262 w_k1_INTC_s010
                                 + 1.51378 w_k1_KO_s010 + 2.06724 w_k1_MAT_s010
                                 + 1.19808 w_k1_MSFT_s010
                                 + 1.28519 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.837048 w_k1_ADM_s010
                                 + 0.683193 w_k1_AXP_s010
                                 + 0.734391 w_k1_DIS_s010
                                 + 0.743253 w_k1_FDX_s010
                                 + 0.797615 w_k1_IBM_s010
                                 + 0.63198 w_k1_INTC_s010
                                 + 0.774566 w_k1_KO_s010
                                 + 0.730045 w_k1_MAT_s010
                                 + 0.721783 w_k1_MSFT_s010
                                 + 0.865253 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.6391 w_k1_ADM_s010
                                 + 1.29893 w_k1_AXP_s010
                                 + 1.43667 w_k1_DIS_s010
                                 + 1.21083 w_k1_FDX_s010
                                 + 1.20233 w_k1_IBM_s010
                                 + 1.27234 w_k1_INTC_s010
                                 + 1.14953 w_k1_KO_s010 + 1.28016 w_k1_MAT_s010
                                 + 1.15912 w_k1_MSFT_s010
                                 + 1.13958 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.53581 w_k1_ADM_s010
                                 + 1.19725 w_k1_AXP_s010
                                 + 1.14042 w_k1_DIS_s010
                                 + 1.14858 w_k1_FDX_s010 + 0.9493 w_k1_IBM_s010
                                 + 0.914643 w_k1_INTC_s010
                                 + 1.02944 w_k1_KO_s010 + 1.33785 w_k1_MAT_s010
                                 + 1.01475 w_k1_MSFT_s010 + 1.0604 w_k1_PG_s010
                                 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.26984 w_k1_ADM_s010
                                 + 0.808846 w_k1_AXP_s010
                                 + 1.25381 w_k1_DIS_s010
                                 + 0.975202 w_k1_FDX_s010
                                 + 1.00248 w_k1_IBM_s010
                                 + 0.763635 w_k1_INTC_s010
                                 + 0.914785 w_k1_KO_s010
                                 + 0.765178 w_k1_MAT_s010
                                 + 0.94184 w_k1_MSFT_s010
                                 + 0.995317 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.879332 w_k1_ADM_s010
                                 + 0.821986 w_k1_AXP_s010
                                 + 0.973379 w_k1_DIS_s010
                                 + 0.813941 w_k1_FDX_s010
                                 + 1.30417 w_k1_IBM_s010
                                 + 1.04508 w_k1_INTC_s010
                                 + 1.13312 w_k1_KO_s010 + 1.14251 w_k1_MAT_s010
                                 + 0.896357 w_k1_MSFT_s010
                                 + 1.11796 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.47697 w_k1_ADM_s011
                                 + 1.40764 w_k1_AXP_s011
                                 + 1.67394 w_k1_DIS_s011
                                 + 1.29409 w_k1_FDX_s011
                                 + 1.08782 w_k1_IBM_s011
                                 + 1.57816 w_k1_INTC_s011
                                 + 1.51252 w_k1_KO_s011
                                 + 0.939893 w_k1_MAT_s011
                                 + 1.35743 w_k1_MSFT_s011 + 1.1816 w_k1_PG_s011
                                 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.934289 w_k1_ADM_s011
                                 + 0.948337 w_k1_AXP_s011
                                 + 0.958921 w_k1_DIS_s011
                                 + 0.913631 w_k1_FDX_s011
                                 + 1.50313 w_k1_IBM_s011
                                 + 1.0958 w_k1_INTC_s011 + 1.28571 w_k1_KO_s011
                                 + 1.22638 w_k1_MAT_s011
                                 + 0.976323 w_k1_MSFT_s011
                                 + 1.07923 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.45936 w_k1_ADM_s011
                                 + 1.5366 w_k1_AXP_s011 + 1.40075 w_k1_DIS_s011
                                 + 1.43155 w_k1_FDX_s011
                                 + 1.14711 w_k1_IBM_s011
                                 + 1.46169 w_k1_INTC_s011
                                 + 1.20054 w_k1_KO_s011 + 1.3941 w_k1_MAT_s011
                                 + 1.48217 w_k1_MSFT_s011
                                 + 1.26668 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.12076 w_k1_ADM_s011
                                 + 0.87403 w_k1_AXP_s011
                                 + 0.92459 w_k1_DIS_s011
                                 + 1.10796 w_k1_FDX_s011
                                 + 0.553249 w_k1_IBM_s011
                                 + 1.16988 w_k1_INTC_s011
                                 + 0.906422 w_k1_KO_s011
                                 + 0.562209 w_k1_MAT_s011
                                 + 1.0694 w_k1_MSFT_s011 + 1.03176 w_k1_PG_s011
                                 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.12446 w_k1_ADM_s011
                                 + 1.12842 w_k1_AXP_s011
                                 + 1.29874 w_k1_DIS_s011
                                 + 1.01154 w_k1_FDX_s011
                                 + 1.35016 w_k1_IBM_s011
                                 + 1.07624 w_k1_INTC_s011
                                 + 1.33684 w_k1_KO_s011 + 1.54892 w_k1_MAT_s011
                                 + 1.29524 w_k1_MSFT_s011
                                 + 1.22048 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.997181 w_k1_ADM_s011
                                 + 0.880592 w_k1_AXP_s011
                                 + 1.35122 w_k1_DIS_s011
                                 + 1.06987 w_k1_FDX_s011 + 1.0062 w_k1_IBM_s011
                                 + 0.977825 w_k1_INTC_s011
                                 + 1.16099 w_k1_KO_s011 + 1.24446 w_k1_MAT_s011
                                 + 1.23151 w_k1_MSFT_s011
                                 + 1.11476 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.01868 w_k1_ADM_s011
                                 + 0.932602 w_k1_AXP_s011
                                 + 0.954796 w_k1_DIS_s011
                                 + 0.952513 w_k1_FDX_s011
                                 + 0.919041 w_k1_IBM_s011
                                 + 1.10221 w_k1_INTC_s011
                                 + 0.939053 w_k1_KO_s011
                                 + 0.834451 w_k1_MAT_s011
                                 + 0.969278 w_k1_MSFT_s011
                                 + 0.975375 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.875953 w_k1_ADM_s011
                                 + 0.599225 w_k1_AXP_s011
                                 + 0.704982 w_k1_DIS_s011
                                 + 0.740963 w_k1_FDX_s011
                                 + 0.779517 w_k1_IBM_s011
                                 + 0.940011 w_k1_INTC_s011
                                 + 0.89897 w_k1_KO_s011 + 0.40565 w_k1_MAT_s011
                                 + 0.876517 w_k1_MSFT_s011
                                 + 0.859077 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.783085 w_k1_ADM_s012
                                 + 0.651026 w_k1_AXP_s012
                                 + 0.534311 w_k1_DIS_s012
                                 + 0.770432 w_k1_FDX_s012
                                 + 0.57542 w_k1_IBM_s012
                                 + 0.714033 w_k1_INTC_s012
                                 + 0.659245 w_k1_KO_s012
                                 + 0.498008 w_k1_MAT_s012
                                 + 0.68592 w_k1_MSFT_s012
                                 + 0.813007 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.848789 w_k1_ADM_s012
                                 + 0.701736 w_k1_AXP_s012
                                 + 0.810803 w_k1_DIS_s012
                                 + 0.921753 w_k1_FDX_s012
                                 + 0.766387 w_k1_IBM_s012
                                 + 1.02473 w_k1_INTC_s012
                                 + 0.877293 w_k1_KO_s012
                                 + 0.581583 w_k1_MAT_s012
                                 + 0.872231 w_k1_MSFT_s012
                                 + 0.886116 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.791548 w_k1_ADM_s012
                                 + 0.723002 w_k1_AXP_s012
                                 + 0.958599 w_k1_DIS_s012
                                 + 0.865325 w_k1_FDX_s012
                                 + 1.43174 w_k1_IBM_s012
                                 + 0.948088 w_k1_INTC_s012
                                 + 1.24765 w_k1_KO_s012 + 1.32758 w_k1_MAT_s012
                                 + 0.894233 w_k1_MSFT_s012
                                 + 1.05788 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.25406 w_k1_ADM_s012
                                 + 1.1498 w_k1_AXP_s012 + 1.52737 w_k1_DIS_s012
                                 + 1.34716 w_k1_FDX_s012
                                 + 1.11262 w_k1_IBM_s012
                                 + 1.10562 w_k1_INTC_s012
                                 + 1.26432 w_k1_KO_s012 + 1.0664 w_k1_MAT_s012
                                 + 1.33468 w_k1_MSFT_s012
                                 + 1.25379 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.15867 w_k1_ADM_s012
                                 + 1.2388 w_k1_AXP_s012 + 1.13217 w_k1_DIS_s012
                                 + 1.13543 w_k1_FDX_s012
                                 + 1.50116 w_k1_IBM_s012
                                 + 1.11137 w_k1_INTC_s012
                                 + 1.46069 w_k1_KO_s012 + 1.71717 w_k1_MAT_s012
                                 + 1.10813 w_k1_MSFT_s012
                                 + 1.09896 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.865787 w_k1_ADM_s012
                                 + 0.92639 w_k1_AXP_s012
                                 + 0.932849 w_k1_DIS_s012
                                 + 0.964661 w_k1_FDX_s012
                                 + 1.02777 w_k1_IBM_s012
                                 + 0.892574 w_k1_INTC_s012
                                 + 0.928084 w_k1_KO_s012
                                 + 1.02054 w_k1_MAT_s012
                                 + 0.82253 w_k1_MSFT_s012
                                 + 0.913605 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.40861 w_k1_ADM_s012
                                 + 1.14333 w_k1_AXP_s012
                                 + 1.59221 w_k1_DIS_s012
                                 + 1.33721 w_k1_FDX_s012
                                 + 1.62236 w_k1_IBM_s012
                                 + 1.06537 w_k1_INTC_s012
                                 + 1.58059 w_k1_KO_s012 + 1.31011 w_k1_MAT_s012
                                 + 1.22816 w_k1_MSFT_s012 + 1.3425 w_k1_PG_s012
                                 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.03021 w_k1_ADM_s012
                                 + 0.929524 w_k1_AXP_s012
                                 + 1.13538 w_k1_DIS_s012
                                 + 1.17071 w_k1_FDX_s012
                                 + 0.884924 w_k1_IBM_s012
                                 + 1.06431 w_k1_INTC_s012
                                 + 1.02629 w_k1_KO_s012
                                 + 0.926736 w_k1_MAT_s012
                                 + 1.04637 w_k1_MSFT_s012
                                 + 1.11781 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.16803 w_k1_ADM_s013
                                 + 1.18562 w_k1_AXP_s013
                                 + 1.12132 w_k1_DIS_s013
                                 + 0.986906 w_k1_FDX_s013
                                 + 1.36416 w_k1_IBM_s013
                                 + 0.841866 w_k1_INTC_s013
                                 + 1.36764 w_k1_KO_s013 + 1.65681 w_k1_MAT_s013
                                 + 1.01531 w_k1_MSFT_s013
                                 + 1.19629 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.955904 w_k1_ADM_s013
                                 + 0.884164 w_k1_AXP_s013
                                 + 0.668652 w_k1_DIS_s013
                                 + 1.04822 w_k1_FDX_s013
                                 + 0.414138 w_k1_IBM_s013
                                 + 0.808246 w_k1_INTC_s013
                                 + 0.744742 w_k1_KO_s013
                                 + 0.641755 w_k1_MAT_s013
                                 + 0.881627 w_k1_MSFT_s013
                                 + 0.98777 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.27129 w_k1_ADM_s013
                                 + 1.19715 w_k1_AXP_s013
                                 + 1.30219 w_k1_DIS_s013
                                 + 1.27499 w_k1_FDX_s013
                                 + 0.755064 w_k1_IBM_s013
                                 + 1.15792 w_k1_INTC_s013
                                 + 1.19786 w_k1_KO_s013
                                 + 0.665883 w_k1_MAT_s013
                                 + 1.1369 w_k1_MSFT_s013 + 1.19493 w_k1_PG_s013
                                 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.957849 w_k1_ADM_s013
                                 + 0.883231 w_k1_AXP_s013
                                 + 0.965836 w_k1_DIS_s013
                                 + 1.00752 w_k1_FDX_s013
                                 + 0.804246 w_k1_IBM_s013
                                 + 0.818262 w_k1_INTC_s013
                                 + 1.03613 w_k1_KO_s013 + 1.05127 w_k1_MAT_s013
                                 + 0.899069 w_k1_MSFT_s013
                                 + 1.06561 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.96201 w_k1_ADM_s013
                                 + 1.38864 w_k1_AXP_s013
                                 + 1.78828 w_k1_DIS_s013
                                 + 1.20602 w_k1_FDX_s013
                                 + 1.50039 w_k1_IBM_s013
                                 + 1.03004 w_k1_INTC_s013
                                 + 1.56261 w_k1_KO_s013 + 1.89956 w_k1_MAT_s013
                                 + 1.17035 w_k1_MSFT_s013
                                 + 1.48895 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.767805 w_k1_ADM_s013
                                 + 0.685154 w_k1_AXP_s013
                                 + 0.662421 w_k1_DIS_s013
                                 + 0.771673 w_k1_FDX_s013
                                 + 0.595825 w_k1_IBM_s013
                                 + 0.719792 w_k1_INTC_s013
                                 + 0.833722 w_k1_KO_s013
                                 + 0.683054 w_k1_MAT_s013
                                 + 0.771229 w_k1_MSFT_s013
                                 + 0.943992 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.829139 w_k1_ADM_s013
                                 + 0.82284 w_k1_AXP_s013
                                 + 0.747776 w_k1_DIS_s013
                                 + 0.850865 w_k1_FDX_s013
                                 + 0.875532 w_k1_IBM_s013
                                 + 0.832012 w_k1_INTC_s013
                                 + 0.886831 w_k1_KO_s013
                                 + 0.809226 w_k1_MAT_s013
                                 + 0.800538 w_k1_MSFT_s013
                                 + 0.954036 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.05212 w_k1_ADM_s013
                                 + 1.35017 w_k1_AXP_s013
                                 + 0.967217 w_k1_DIS_s013
                                 + 1.10434 w_k1_FDX_s013
                                 + 1.07849 w_k1_IBM_s013
                                 + 0.982796 w_k1_INTC_s013
                                 + 1.05676 w_k1_KO_s013 + 1.50983 w_k1_MAT_s013
                                 + 0.973287 w_k1_MSFT_s013
                                 + 1.20891 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.40353 w_k1_ADM_s014
                                 + 1.28018 w_k1_AXP_s014
                                 + 1.33043 w_k1_DIS_s014
                                 + 1.21735 w_k1_FDX_s014
                                 + 0.869549 w_k1_IBM_s014
                                 + 1.00275 w_k1_INTC_s014
                                 + 1.01213 w_k1_KO_s014
                                 + 0.918567 w_k1_MAT_s014
                                 + 1.03035 w_k1_MSFT_s014
                                 + 1.01017 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.22769 w_k1_ADM_s014
                                 + 1.45851 w_k1_AXP_s014
                                 + 1.70926 w_k1_DIS_s014
                                 + 1.30147 w_k1_FDX_s014
                                 + 1.56537 w_k1_IBM_s014
                                 + 1.13773 w_k1_INTC_s014
                                 + 1.42754 w_k1_KO_s014 + 1.58749 w_k1_MAT_s014
                                 + 1.31728 w_k1_MSFT_s014
                                 + 1.15223 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.16002 w_k1_ADM_s014
                                 + 0.995939 w_k1_AXP_s014
                                 + 1.33879 w_k1_DIS_s014
                                 + 0.995365 w_k1_FDX_s014
                                 + 1.29502 w_k1_IBM_s014
                                 + 0.943684 w_k1_INTC_s014
                                 + 1.19015 w_k1_KO_s014
                                 + 0.935469 w_k1_MAT_s014
                                 + 1.00708 w_k1_MSFT_s014
                                 + 0.943553 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.718485 w_k1_ADM_s014
                                 + 0.892425 w_k1_AXP_s014
                                 + 0.894245 w_k1_DIS_s014
                                 + 0.846141 w_k1_FDX_s014
                                 + 1.07282 w_k1_IBM_s014
                                 + 0.894277 w_k1_INTC_s014
                                 + 0.878933 w_k1_KO_s014
                                 + 0.928444 w_k1_MAT_s014
                                 + 0.838134 w_k1_MSFT_s014
                                 + 0.804776 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.81644 w_k1_ADM_s014
                                 + 0.652312 w_k1_AXP_s014
                                 + 0.684611 w_k1_DIS_s014
                                 + 0.737507 w_k1_FDX_s014
                                 + 0.637087 w_k1_IBM_s014
                                 + 0.687625 w_k1_INTC_s014
                                 + 0.632225 w_k1_KO_s014
                                 + 0.556567 w_k1_MAT_s014
                                 + 0.801681 w_k1_MSFT_s014
                                 + 0.69821 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.989891 w_k1_ADM_s014
                                 + 1.00397 w_k1_AXP_s014
                                 + 0.814429 w_k1_DIS_s014
                                 + 0.99397 w_k1_FDX_s014
                                 + 0.783434 w_k1_IBM_s014
                                 + 0.84367 w_k1_INTC_s014
                                 + 0.730304 w_k1_KO_s014
                                 + 0.649767 w_k1_MAT_s014
                                 + 0.805985 w_k1_MSFT_s014
                                 + 0.828286 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.298 w_k1_ADM_s014
                                 + 1.09836 w_k1_AXP_s014
                                 + 1.21501 w_k1_DIS_s014 + 1.1154 w_k1_FDX_s014
                                 + 0.916017 w_k1_IBM_s014
                                 + 0.961055 w_k1_INTC_s014
                                 + 1.05233 w_k1_KO_s014
                                 + 0.703262 w_k1_MAT_s014
                                 + 1.0312 w_k1_MSFT_s014 + 0.86908 w_k1_PG_s014
                                 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.12691 w_k1_ADM_s014
                                 + 1.05245 w_k1_AXP_s014
                                 + 1.04226 w_k1_DIS_s014
                                 + 0.882604 w_k1_FDX_s014
                                 + 1.6893 w_k1_IBM_s014
                                 + 0.891192 w_k1_INTC_s014
                                 + 1.02738 w_k1_KO_s014 + 1.22512 w_k1_MAT_s014
                                 + 0.879932 w_k1_MSFT_s014
                                 + 0.993184 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.05991 w_k1_ADM_s015
                                 + 1.27358 w_k1_AXP_s015
                                 + 0.998336 w_k1_DIS_s015
                                 + 0.903081 w_k1_FDX_s015
                                 + 1.34566 w_k1_IBM_s015
                                 + 1.10113 w_k1_INTC_s015
                                 + 1.07247 w_k1_KO_s015 + 1.18802 w_k1_MAT_s015
                                 + 0.921127 w_k1_MSFT_s015
                                 + 0.942885 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.984821 w_k1_ADM_s015
                                 + 1.42478 w_k1_AXP_s015
                                 + 1.61543 w_k1_DIS_s015
                                 + 1.23302 w_k1_FDX_s015
                                 + 1.74522 w_k1_IBM_s015
                                 + 1.15652 w_k1_INTC_s015
                                 + 1.30336 w_k1_KO_s015 + 1.93531 w_k1_MAT_s015
                                 + 0.997491 w_k1_MSFT_s015
                                 + 1.16163 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.0359 w_k1_ADM_s015
                                 + 0.786878 w_k1_AXP_s015
                                 + 0.929069 w_k1_DIS_s015
                                 + 0.933516 w_k1_FDX_s015
                                 + 0.69182 w_k1_IBM_s015 + 1.131 w_k1_INTC_s015
                                 + 0.895746 w_k1_KO_s015
                                 + 0.55857 w_k1_MAT_s015
                                 + 0.913764 w_k1_MSFT_s015
                                 + 0.973796 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.935366 w_k1_ADM_s015
                                 + 0.714143 w_k1_AXP_s015
                                 + 0.773149 w_k1_DIS_s015
                                 + 0.70558 w_k1_FDX_s015
                                 + 0.767689 w_k1_IBM_s015
                                 + 0.903742 w_k1_INTC_s015
                                 + 0.888481 w_k1_KO_s015
                                 + 0.746804 w_k1_MAT_s015
                                 + 0.870626 w_k1_MSFT_s015
                                 + 0.774982 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.866166 w_k1_ADM_s015
                                 + 0.787436 w_k1_AXP_s015
                                 + 0.892502 w_k1_DIS_s015
                                 + 0.822946 w_k1_FDX_s015
                                 + 1.08824 w_k1_IBM_s015
                                 + 1.03215 w_k1_INTC_s015
                                 + 0.998872 w_k1_KO_s015
                                 + 0.77278 w_k1_MAT_s015
                                 + 0.776734 w_k1_MSFT_s015
                                 + 0.830126 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.24567 w_k1_ADM_s015
                                 + 1.16887 w_k1_AXP_s015
                                 + 1.09187 w_k1_DIS_s015
                                 + 1.09011 w_k1_FDX_s015
                                 + 0.707574 w_k1_IBM_s015
                                 + 1.12358 w_k1_INTC_s015
                                 + 1.04644 w_k1_KO_s015
                                 + 0.693663 w_k1_MAT_s015
                                 + 1.04327 w_k1_MSFT_s015
                                 + 0.930517 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.941526 w_k1_ADM_s015
                                 + 1.04191 w_k1_AXP_s015
                                 + 1.12418 w_k1_DIS_s015
                                 + 1.03278 w_k1_FDX_s015
                                 + 0.953892 w_k1_IBM_s015
                                 + 1.02645 w_k1_INTC_s015
                                 + 1.12222 w_k1_KO_s015
                                 + 0.839804 w_k1_MAT_s015
                                 + 1.10784 w_k1_MSFT_s015
                                 + 0.958108 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.17464 w_k1_ADM_s015
                                 + 1.26518 w_k1_AXP_s015
                                 + 1.59869 w_k1_DIS_s015
                                 + 1.23534 w_k1_FDX_s015
                                 + 0.955471 w_k1_IBM_s015
                                 + 1.17602 w_k1_INTC_s015
                                 + 1.30857 w_k1_KO_s015 + 1.37726 w_k1_MAT_s015
                                 + 1.22112 w_k1_MSFT_s015
                                 + 1.09674 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.11058 w_k1_ADM_s016
                                 + 1.14935 w_k1_AXP_s016
                                 + 1.09254 w_k1_DIS_s016
                                 + 1.26132 w_k1_FDX_s016
                                 + 0.91537 w_k1_IBM_s016
                                 + 1.5951 w_k1_INTC_s016 + 1.06759 w_k1_KO_s016
                                 + 1.10047 w_k1_MAT_s016
                                 + 1.31803 w_k1_MSFT_s016
                                 + 1.15667 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.829208 w_k1_ADM_s016
                                 + 0.845509 w_k1_AXP_s016
                                 + 0.906821 w_k1_DIS_s016
                                 + 0.921344 w_k1_FDX_s016
                                 + 1.20748 w_k1_IBM_s016
                                 + 1.18488 w_k1_INTC_s016
                                 + 1.02488 w_k1_KO_s016
                                 + 0.936235 w_k1_MAT_s016
                                 + 1.05905 w_k1_MSFT_s016
                                 + 1.08756 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.89093 w_k1_ADM_s016
                                 + 0.832292 w_k1_AXP_s016
                                 + 1.03875 w_k1_DIS_s016
                                 + 1.02977 w_k1_FDX_s016
                                 + 0.75671 w_k1_IBM_s016
                                 + 1.21008 w_k1_INTC_s016
                                 + 0.915124 w_k1_KO_s016
                                 + 0.612501 w_k1_MAT_s016
                                 + 1.09574 w_k1_MSFT_s016
                                 + 0.908572 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.855717 w_k1_ADM_s016
                                 + 1.38001 w_k1_AXP_s016
                                 + 1.05996 w_k1_DIS_s016
                                 + 1.16714 w_k1_FDX_s016
                                 + 1.16822 w_k1_IBM_s016
                                 + 1.42495 w_k1_INTC_s016 + 1.0796 w_k1_KO_s016
                                 + 1.30592 w_k1_MAT_s016
                                 + 1.17123 w_k1_MSFT_s016
                                 + 1.04482 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.24692 w_k1_ADM_s016
                                 + 1.2229 w_k1_AXP_s016 + 1.54501 w_k1_DIS_s016
                                 + 1.02405 w_k1_FDX_s016
                                 + 1.60674 w_k1_IBM_s016
                                 + 1.40571 w_k1_INTC_s016
                                 + 1.64275 w_k1_KO_s016 + 1.00483 w_k1_MAT_s016
                                 + 1.27401 w_k1_MSFT_s016
                                 + 1.07632 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.20266 w_k1_ADM_s016
                                 + 1.16668 w_k1_AXP_s016
                                 + 1.58068 w_k1_DIS_s016
                                 + 1.37996 w_k1_FDX_s016
                                 + 1.24189 w_k1_IBM_s016
                                 + 1.49694 w_k1_INTC_s016
                                 + 1.31394 w_k1_KO_s016
                                 + 0.919266 w_k1_MAT_s016
                                 + 1.40694 w_k1_MSFT_s016
                                 + 1.23615 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.663642 w_k1_ADM_s016
                                 + 0.795206 w_k1_AXP_s016
                                 + 0.829763 w_k1_DIS_s016
                                 + 0.855233 w_k1_FDX_s016
                                 + 0.875349 w_k1_IBM_s016
                                 + 0.920073 w_k1_INTC_s016
                                 + 0.78042 w_k1_KO_s016
                                 + 0.826231 w_k1_MAT_s016
                                 + 0.896761 w_k1_MSFT_s016
                                 + 0.850178 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.943492 w_k1_ADM_s016
                                 + 0.881609 w_k1_AXP_s016
                                 + 1.26093 w_k1_DIS_s016
                                 + 1.06893 w_k1_FDX_s016
                                 + 0.987127 w_k1_IBM_s016
                                 + 1.29957 w_k1_INTC_s016 + 1.1172 w_k1_KO_s016
                                 + 0.756949 w_k1_MAT_s016
                                 + 1.2379 w_k1_MSFT_s016 + 1.0781 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.718023 w_k1_ADM_s017
                                 + 0.727425 w_k1_AXP_s017
                                 + 0.654105 w_k1_DIS_s017
                                 + 0.682102 w_k1_FDX_s017
                                 + 1.0332 w_k1_IBM_s017
                                 + 0.764736 w_k1_INTC_s017
                                 + 0.922248 w_k1_KO_s017
                                 + 0.856264 w_k1_MAT_s017
                                 + 0.751046 w_k1_MSFT_s017
                                 + 0.910165 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.03518 w_k1_ADM_s017
                                 + 0.812944 w_k1_AXP_s017
                                 + 1.34913 w_k1_DIS_s017
                                 + 0.996137 w_k1_FDX_s017
                                 + 0.940605 w_k1_IBM_s017
                                 + 0.996125 w_k1_INTC_s017
                                 + 1.29539 w_k1_KO_s017
                                 + 0.443143 w_k1_MAT_s017
                                 + 1.05329 w_k1_MSFT_s017
                                 + 0.969427 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.02945 w_k1_ADM_s017
                                 + 1.04655 w_k1_AXP_s017
                                 + 0.857262 w_k1_DIS_s017
                                 + 0.850632 w_k1_FDX_s017
                                 + 1.17516 w_k1_IBM_s017
                                 + 0.832299 w_k1_INTC_s017
                                 + 0.974283 w_k1_KO_s017
                                 + 0.90553 w_k1_MAT_s017
                                 + 0.776664 w_k1_MSFT_s017
                                 + 0.974346 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.06949 w_k1_ADM_s017
                                 + 1.19333 w_k1_AXP_s017
                                 + 1.28491 w_k1_DIS_s017
                                 + 1.19343 w_k1_FDX_s017
                                 + 0.808593 w_k1_IBM_s017
                                 + 1.14221 w_k1_INTC_s017
                                 + 1.06911 w_k1_KO_s017
                                 + 0.815725 w_k1_MAT_s017
                                 + 1.01515 w_k1_MSFT_s017
                                 + 1.04806 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.715823 w_k1_ADM_s017
                                 + 0.707434 w_k1_AXP_s017
                                 + 0.765626 w_k1_DIS_s017
                                 + 0.807689 w_k1_FDX_s017
                                 + 0.721472 w_k1_IBM_s017
                                 + 0.973192 w_k1_INTC_s017
                                 + 0.825704 w_k1_KO_s017
                                 + 0.666483 w_k1_MAT_s017
                                 + 0.935541 w_k1_MSFT_s017
                                 + 0.909891 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.910925 w_k1_ADM_s017
                                 + 0.943175 w_k1_AXP_s017
                                 + 1.24041 w_k1_DIS_s017
                                 + 0.956345 w_k1_FDX_s017
                                 + 1.31122 w_k1_IBM_s017
                                 + 0.859777 w_k1_INTC_s017
                                 + 1.43209 w_k1_KO_s017 + 1.02824 w_k1_MAT_s017
                                 + 0.986783 w_k1_MSFT_s017
                                 + 1.05152 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.01623 w_k1_ADM_s017
                                 + 0.841496 w_k1_AXP_s017
                                 + 0.859578 w_k1_DIS_s017
                                 + 1.02015 w_k1_FDX_s017 + 0.7201 w_k1_IBM_s017
                                 + 1.08025 w_k1_INTC_s017
                                 + 0.906015 w_k1_KO_s017
                                 + 0.324804 w_k1_MAT_s017
                                 + 0.942615 w_k1_MSFT_s017
                                 + 0.849035 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.15626 w_k1_ADM_s017
                                 + 1.70356 w_k1_AXP_s017
                                 + 1.42049 w_k1_DIS_s017
                                 + 1.27775 w_k1_FDX_s017
                                 + 1.16907 w_k1_IBM_s017
                                 + 1.28916 w_k1_INTC_s017
                                 + 1.37759 w_k1_KO_s017 + 1.41115 w_k1_MAT_s017
                                 + 1.17217 w_k1_MSFT_s017
                                 + 1.17918 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.673628 w_k1_ADM_s100
                                 + 0.760103 w_k1_AXP_s100
                                 + 0.829195 w_k1_DIS_s100
                                 + 0.947551 w_k1_FDX_s100
                                 + 0.61331 w_k1_IBM_s100
                                 + 0.804638 w_k1_INTC_s100
                                 + 0.673442 w_k1_KO_s100
                                 + 0.748742 w_k1_MAT_s100
                                 + 0.936019 w_k1_MSFT_s100
                                 + 0.862114 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.728204 w_k1_ADM_s100
                                 + 0.820223 w_k1_AXP_s100
                                 + 0.788266 w_k1_DIS_s100
                                 + 0.948317 w_k1_FDX_s100
                                 + 0.746781 w_k1_IBM_s100
                                 + 0.824306 w_k1_INTC_s100
                                 + 0.690573 w_k1_KO_s100
                                 + 0.957006 w_k1_MAT_s100
                                 + 0.865308 w_k1_MSFT_s100
                                 + 0.818964 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.995188 w_k1_ADM_s100
                                 + 1.19551 w_k1_AXP_s100
                                 + 0.714442 w_k1_DIS_s100
                                 + 0.969583 w_k1_FDX_s100
                                 + 1.00753 w_k1_IBM_s100
                                 + 0.872267 w_k1_INTC_s100
                                 + 0.902342 w_k1_KO_s100
                                 + 1.25849 w_k1_MAT_s100
                                 + 0.89531 w_k1_MSFT_s100
                                 + 0.985173 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.11871 w_k1_ADM_s100
                                 + 1.41266 w_k1_AXP_s100
                                 + 1.51682 w_k1_DIS_s100
                                 + 1.40373 w_k1_FDX_s100
                                 + 1.37537 w_k1_IBM_s100
                                 + 1.15586 w_k1_INTC_s100
                                 + 1.22507 w_k1_KO_s100 + 1.59702 w_k1_MAT_s100
                                 + 1.30981 w_k1_MSFT_s100
                                 + 1.26944 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.920942 w_k1_ADM_s100
                                 + 0.989298 w_k1_AXP_s100
                                 + 1.07973 w_k1_DIS_s100
                                 + 1.02122 w_k1_FDX_s100
                                 + 1.38997 w_k1_IBM_s100
                                 + 1.15513 w_k1_INTC_s100
                                 + 1.20599 w_k1_KO_s100 + 1.42944 w_k1_MAT_s100
                                 + 1.09255 w_k1_MSFT_s100
                                 + 1.04084 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.832124 w_k1_ADM_s100
                                 + 0.97678 w_k1_AXP_s100 + 1.193 w_k1_DIS_s100
                                 + 1.13927 w_k1_FDX_s100
                                 + 0.818171 w_k1_IBM_s100
                                 + 1.09593 w_k1_INTC_s100
                                 + 1.00485 w_k1_KO_s100 + 1.0919 w_k1_MAT_s100
                                 + 1.06683 w_k1_MSFT_s100
                                 + 1.04461 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.735645 w_k1_ADM_s100
                                 + 0.900037 w_k1_AXP_s100
                                 + 1.01155 w_k1_DIS_s100
                                 + 0.906495 w_k1_FDX_s100
                                 + 0.921348 w_k1_IBM_s100
                                 + 1.19146 w_k1_INTC_s100
                                 + 0.947429 w_k1_KO_s100
                                 + 0.91401 w_k1_MAT_s100
                                 + 1.10281 w_k1_MSFT_s100
                                 + 1.02388 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.730525 w_k1_ADM_s100
                                 + 0.67192 w_k1_AXP_s100
                                 + 0.695267 w_k1_DIS_s100
                                 + 0.769162 w_k1_FDX_s100
                                 + 1.06097 w_k1_IBM_s100
                                 + 0.86975 w_k1_INTC_s100
                                 + 0.798321 w_k1_KO_s100
                                 + 1.01831 w_k1_MAT_s100
                                 + 0.942536 w_k1_MSFT_s100
                                 + 0.834017 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.07562 w_k1_ADM_s101
                                 + 0.664791 w_k1_AXP_s101
                                 + 0.5813 w_k1_DIS_s101
                                 + 0.675345 w_k1_FDX_s101
                                 + 0.766698 w_k1_IBM_s101
                                 + 0.524626 w_k1_INTC_s101
                                 + 0.794893 w_k1_KO_s101
                                 + 0.601924 w_k1_MAT_s101
                                 + 0.603989 w_k1_MSFT_s101
                                 + 0.72858 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.55718 w_k1_ADM_s101
                                 + 1.65734 w_k1_AXP_s101 + 1.284 w_k1_DIS_s101
                                 + 1.26023 w_k1_FDX_s101
                                 + 0.958967 w_k1_IBM_s101
                                 + 1.1843 w_k1_INTC_s101 + 1.18355 w_k1_KO_s101
                                 + 0.88776 w_k1_MAT_s101
                                 + 1.20229 w_k1_MSFT_s101
                                 + 1.17518 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.26583 w_k1_ADM_s101
                                 + 1.66285 w_k1_AXP_s101
                                 + 1.27413 w_k1_DIS_s101
                                 + 1.26723 w_k1_FDX_s101
                                 + 0.771404 w_k1_IBM_s101
                                 + 1.24132 w_k1_INTC_s101
                                 + 1.03312 w_k1_KO_s101
                                 + 0.781084 w_k1_MAT_s101
                                 + 1.03048 w_k1_MSFT_s101
                                 + 1.05782 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.16956 w_k1_ADM_s101
                                 + 0.938809 w_k1_AXP_s101
                                 + 0.927665 w_k1_DIS_s101
                                 + 0.818881 w_k1_FDX_s101
                                 + 0.948679 w_k1_IBM_s101
                                 + 1.01647 w_k1_INTC_s101
                                 + 1.13711 w_k1_KO_s101
                                 + 0.762305 w_k1_MAT_s101
                                 + 0.993223 w_k1_MSFT_s101
                                 + 0.899075 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.22483 w_k1_ADM_s101
                                 + 0.956942 w_k1_AXP_s101
                                 + 0.729035 w_k1_DIS_s101
                                 + 0.767104 w_k1_FDX_s101
                                 + 1.22314 w_k1_IBM_s101
                                 + 0.857986 w_k1_INTC_s101
                                 + 1.09156 w_k1_KO_s101
                                 + 0.971726 w_k1_MAT_s101
                                 + 0.827428 w_k1_MSFT_s101
                                 + 0.94533 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.23559 w_k1_ADM_s101
                                 + 1.53475 w_k1_AXP_s101
                                 + 1.32759 w_k1_DIS_s101
                                 + 1.03144 w_k1_FDX_s101
                                 + 1.62912 w_k1_IBM_s101
                                 + 1.22353 w_k1_INTC_s101
                                 + 1.45718 w_k1_KO_s101 + 1.55473 w_k1_MAT_s101
                                 + 1.03486 w_k1_MSFT_s101
                                 + 1.00724 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.02135 w_k1_ADM_s101
                                 + 1.30056 w_k1_AXP_s101
                                 + 1.01406 w_k1_DIS_s101 + 1.0288 w_k1_FDX_s101
                                 + 1.08184 w_k1_IBM_s101
                                 + 1.09165 w_k1_INTC_s101
                                 + 1.10906 w_k1_KO_s101
                                 + 0.917634 w_k1_MAT_s101
                                 + 1.03736 w_k1_MSFT_s101
                                 + 0.962451 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.929963 w_k1_ADM_s101
                                 + 1.05589 w_k1_AXP_s101
                                 + 0.790959 w_k1_DIS_s101
                                 + 0.876766 w_k1_FDX_s101
                                 + 0.671947 w_k1_IBM_s101
                                 + 0.998885 w_k1_INTC_s101
                                 + 0.79579 w_k1_KO_s101
                                 + 0.630638 w_k1_MAT_s101
                                 + 0.816481 w_k1_MSFT_s101
                                 + 0.869311 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.0825 w_k1_ADM_s102
                                 + 1.02026 w_k1_AXP_s102
                                 + 1.15731 w_k1_DIS_s102
                                 + 1.03712 w_k1_FDX_s102
                                 + 1.23333 w_k1_IBM_s102
                                 + 0.948467 w_k1_INTC_s102
                                 + 1.13031 w_k1_KO_s102 + 1.21707 w_k1_MAT_s102
                                 + 1.00081 w_k1_MSFT_s102
                                 + 1.04877 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.857207 w_k1_ADM_s102
                                 + 1.12732 w_k1_AXP_s102
                                 + 0.699253 w_k1_DIS_s102
                                 + 0.981111 w_k1_FDX_s102
                                 + 0.748248 w_k1_IBM_s102
                                 + 1.10514 w_k1_INTC_s102
                                 + 0.71055 w_k1_KO_s102
                                 + 0.794492 w_k1_MAT_s102
                                 + 0.837466 w_k1_MSFT_s102
                                 + 0.912758 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.20242 w_k1_ADM_s102
                                 + 1.35605 w_k1_AXP_s102
                                 + 1.23521 w_k1_DIS_s102
                                 + 1.23909 w_k1_FDX_s102
                                 + 1.16741 w_k1_IBM_s102
                                 + 1.10237 w_k1_INTC_s102
                                 + 1.10639 w_k1_KO_s102 + 1.64446 w_k1_MAT_s102
                                 + 1.06673 w_k1_MSFT_s102
                                 + 1.11558 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.814932 w_k1_ADM_s102
                                 + 0.716025 w_k1_AXP_s102
                                 + 0.729985 w_k1_DIS_s102
                                 + 0.907581 w_k1_FDX_s102
                                 + 0.840439 w_k1_IBM_s102
                                 + 1.09904 w_k1_INTC_s102
                                 + 0.766621 w_k1_KO_s102
                                 + 0.647222 w_k1_MAT_s102
                                 + 0.824357 w_k1_MSFT_s102
                                 + 0.838759 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.873717 w_k1_ADM_s102
                                 + 0.892738 w_k1_AXP_s102
                                 + 1.1195 w_k1_DIS_s102 + 1.01833 w_k1_FDX_s102
                                 + 0.911011 w_k1_IBM_s102
                                 + 1.30898 w_k1_INTC_s102
                                 + 1.02071 w_k1_KO_s102
                                 + 0.637823 w_k1_MAT_s102
                                 + 1.04184 w_k1_MSFT_s102
                                 + 1.01302 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.677848 w_k1_ADM_s102
                                 + 0.783547 w_k1_AXP_s102
                                 + 0.543623 w_k1_DIS_s102
                                 + 0.746909 w_k1_FDX_s102
                                 + 0.57195 w_k1_IBM_s102
                                 + 0.830389 w_k1_INTC_s102
                                 + 0.547229 w_k1_KO_s102
                                 + 0.497122 w_k1_MAT_s102
                                 + 0.702471 w_k1_MSFT_s102
                                 + 0.770958 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.583243 w_k1_ADM_s102
                                 + 0.808822 w_k1_AXP_s102
                                 + 0.681329 w_k1_DIS_s102
                                 + 0.777592 w_k1_FDX_s102
                                 + 1.09015 w_k1_IBM_s102
                                 + 0.92361 w_k1_INTC_s102
                                 + 0.828674 w_k1_KO_s102
                                 + 1.13457 w_k1_MAT_s102
                                 + 0.716466 w_k1_MSFT_s102
                                 + 0.793277 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.722865 w_k1_ADM_s102
                                 + 0.579315 w_k1_AXP_s102
                                 + 0.677108 w_k1_DIS_s102
                                 + 0.780024 w_k1_FDX_s102
                                 + 0.811705 w_k1_IBM_s102
                                 + 1.01618 w_k1_INTC_s102
                                 + 0.805196 w_k1_KO_s102
                                 + 0.699491 w_k1_MAT_s102
                                 + 0.830767 w_k1_MSFT_s102
                                 + 0.788899 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.21877 w_k1_ADM_s103
                                 + 1.77753 w_k1_AXP_s103
                                 + 1.47788 w_k1_DIS_s103
                                 + 1.25688 w_k1_FDX_s103
                                 + 1.49745 w_k1_IBM_s103
                                 + 1.24286 w_k1_INTC_s103
                                 + 1.28856 w_k1_KO_s103 + 1.47819 w_k1_MAT_s103
                                 + 1.2002 w_k1_MSFT_s103 + 1.31756 w_k1_PG_s103
                                 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.23868 w_k1_ADM_s103
                                 + 1.0968 w_k1_AXP_s103 + 1.12096 w_k1_DIS_s103
                                 + 1.01309 w_k1_FDX_s103
                                 + 0.993327 w_k1_IBM_s103
                                 + 1.02338 w_k1_INTC_s103
                                 + 1.01541 w_k1_KO_s103
                                 + 0.768357 w_k1_MAT_s103
                                 + 0.972916 w_k1_MSFT_s103
                                 + 1.01033 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.891077 w_k1_ADM_s103
                                 + 0.796874 w_k1_AXP_s103
                                 + 0.781034 w_k1_DIS_s103
                                 + 0.860051 w_k1_FDX_s103
                                 + 0.717882 w_k1_IBM_s103
                                 + 0.785005 w_k1_INTC_s103
                                 + 0.822098 w_k1_KO_s103
                                 + 0.747108 w_k1_MAT_s103
                                 + 0.902094 w_k1_MSFT_s103
                                 + 0.905199 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.871988 w_k1_ADM_s103
                                 + 0.859874 w_k1_AXP_s103
                                 + 0.61372 w_k1_DIS_s103
                                 + 0.775116 w_k1_FDX_s103
                                 + 0.714635 w_k1_IBM_s103
                                 + 1.04821 w_k1_INTC_s103
                                 + 0.733633 w_k1_KO_s103 + 0.6427 w_k1_MAT_s103
                                 + 0.80434 w_k1_MSFT_s103
                                 + 0.763723 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.05085 w_k1_ADM_s103
                                 + 1.13108 w_k1_AXP_s103
                                 + 1.07693 w_k1_DIS_s103 + 1.0935 w_k1_FDX_s103
                                 + 0.742328 w_k1_IBM_s103
                                 + 1.18295 w_k1_INTC_s103
                                 + 0.951077 w_k1_KO_s103
                                 + 0.783743 w_k1_MAT_s103
                                 + 1.1743 w_k1_MSFT_s103 + 1.01177 w_k1_PG_s103
                                 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.34624 w_k1_ADM_s103
                                 + 1.2562 w_k1_AXP_s103 + 1.51845 w_k1_DIS_s103
                                 + 1.35753 w_k1_FDX_s103
                                 + 0.896978 w_k1_IBM_s103
                                 + 1.49775 w_k1_INTC_s103
                                 + 1.07947 w_k1_KO_s103
                                 + 0.723936 w_k1_MAT_s103
                                 + 1.2441 w_k1_MSFT_s103 + 1.0676 w_k1_PG_s103
                                 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.928184 w_k1_ADM_s103
                                 + 0.952768 w_k1_AXP_s103
                                 + 0.888187 w_k1_DIS_s103
                                 + 0.879225 w_k1_FDX_s103
                                 + 1.22706 w_k1_IBM_s103
                                 + 1.00054 w_k1_INTC_s103
                                 + 1.03833 w_k1_KO_s103 + 1.12504 w_k1_MAT_s103
                                 + 0.874028 w_k1_MSFT_s103
                                 + 0.991113 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.759112 w_k1_ADM_s103
                                 + 1.01313 w_k1_AXP_s103
                                 + 0.711603 w_k1_DIS_s103
                                 + 0.7884 w_k1_FDX_s103
                                 + 0.875275 w_k1_IBM_s103
                                 + 1.01911 w_k1_INTC_s103
                                 + 0.855105 w_k1_KO_s103
                                 + 0.985852 w_k1_MAT_s103
                                 + 0.79668 w_k1_MSFT_s103
                                 + 0.873358 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.8375 w_k1_ADM_s104
                                 + 0.81369 w_k1_AXP_s104
                                 + 0.781546 w_k1_DIS_s104
                                 + 0.869758 w_k1_FDX_s104
                                 + 0.718843 w_k1_IBM_s104
                                 + 1.12387 w_k1_INTC_s104
                                 + 0.81736 w_k1_KO_s104
                                 + 0.576851 w_k1_MAT_s104
                                 + 0.883073 w_k1_MSFT_s104
                                 + 0.85386 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.07767 w_k1_ADM_s104
                                 + 1.22644 w_k1_AXP_s104
                                 + 1.21035 w_k1_DIS_s104
                                 + 1.19951 w_k1_FDX_s104
                                 + 1.20553 w_k1_IBM_s104 + 1.227 w_k1_INTC_s104
                                 + 1.03817 w_k1_KO_s104 + 1.10063 w_k1_MAT_s104
                                 + 1.11179 w_k1_MSFT_s104 + 1.1789 w_k1_PG_s104
                                 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.683951 w_k1_ADM_s104
                                 + 0.820949 w_k1_AXP_s104
                                 + 0.968763 w_k1_DIS_s104
                                 + 0.834596 w_k1_FDX_s104
                                 + 1.06197 w_k1_IBM_s104
                                 + 0.886472 w_k1_INTC_s104
                                 + 0.819116 w_k1_KO_s104
                                 + 0.899643 w_k1_MAT_s104
                                 + 0.878525 w_k1_MSFT_s104
                                 + 0.871921 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.05403 w_k1_ADM_s104
                                 + 0.97218 w_k1_AXP_s104
                                 + 1.28906 w_k1_DIS_s104
                                 + 1.00651 w_k1_FDX_s104
                                 + 0.941437 w_k1_IBM_s104
                                 + 1.30863 w_k1_INTC_s104
                                 + 1.01352 w_k1_KO_s104
                                 + 0.797783 w_k1_MAT_s104
                                 + 0.995397 w_k1_MSFT_s104
                                 + 0.959407 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.30136 w_k1_ADM_s104
                                 + 1.21142 w_k1_AXP_s104
                                 + 1.55831 w_k1_DIS_s104
                                 + 1.37633 w_k1_FDX_s104
                                 + 1.03734 w_k1_IBM_s104
                                 + 1.44018 w_k1_INTC_s104
                                 + 1.12017 w_k1_KO_s104
                                 + 0.961157 w_k1_MAT_s104
                                 + 1.28774 w_k1_MSFT_s104
                                 + 1.17944 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.20923 w_k1_ADM_s104
                                 + 1.28534 w_k1_AXP_s104
                                 + 1.09287 w_k1_DIS_s104 + 1.0098 w_k1_FDX_s104
                                 + 1.46165 w_k1_IBM_s104
                                 + 1.05611 w_k1_INTC_s104
                                 + 1.04656 w_k1_KO_s104 + 1.53087 w_k1_MAT_s104
                                 + 0.839809 w_k1_MSFT_s104
                                 + 1.04004 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.2285 w_k1_ADM_s104
                                 + 1.51988 w_k1_AXP_s104
                                 + 1.78689 w_k1_DIS_s104
                                 + 1.28965 w_k1_FDX_s104
                                 + 1.61437 w_k1_IBM_s104
                                 + 1.69496 w_k1_INTC_s104
                                 + 1.51876 w_k1_KO_s104 + 1.84003 w_k1_MAT_s104
                                 + 1.39391 w_k1_MSFT_s104
                                 + 1.31581 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.764881 w_k1_ADM_s104
                                 + 1.17937 w_k1_AXP_s104 + 1.3368 w_k1_DIS_s104
                                 + 1.02611 w_k1_FDX_s104
                                 + 1.27545 w_k1_IBM_s104
                                 + 1.30494 w_k1_INTC_s104
                                 + 1.22814 w_k1_KO_s104 + 1.4864 w_k1_MAT_s104
                                 + 1.19232 w_k1_MSFT_s104
                                 + 1.04749 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.666915 w_k1_ADM_s105
                                 + 0.659333 w_k1_AXP_s105
                                 + 0.574353 w_k1_DIS_s105
                                 + 0.761344 w_k1_FDX_s105
                                 + 0.59155 w_k1_IBM_s105
                                 + 0.938713 w_k1_INTC_s105
                                 + 0.640407 w_k1_KO_s105
                                 + 0.642496 w_k1_MAT_s105
                                 + 0.804808 w_k1_MSFT_s105
                                 + 0.664835 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.12902 w_k1_ADM_s105
                                 + 0.912648 w_k1_AXP_s105
                                 + 1.02848 w_k1_DIS_s105
                                 + 1.03337 w_k1_FDX_s105
                                 + 0.823104 w_k1_IBM_s105
                                 + 1.07704 w_k1_INTC_s105
                                 + 0.867819 w_k1_KO_s105
                                 + 0.720464 w_k1_MAT_s105
                                 + 1.02275 w_k1_MSFT_s105
                                 + 0.777928 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.73601 w_k1_ADM_s105
                                 + 1.60614 w_k1_AXP_s105
                                 + 1.86229 w_k1_DIS_s105
                                 + 1.34826 w_k1_FDX_s105
                                 + 1.76869 w_k1_IBM_s105
                                 + 1.60867 w_k1_INTC_s105
                                 + 1.61973 w_k1_KO_s105 + 1.53633 w_k1_MAT_s105
                                 + 1.44265 w_k1_MSFT_s105
                                 + 1.32933 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.56125 w_k1_ADM_s105
                                 + 1.48475 w_k1_AXP_s105
                                 + 1.38012 w_k1_DIS_s105
                                 + 1.30604 w_k1_FDX_s105
                                 + 1.26221 w_k1_IBM_s105
                                 + 1.53543 w_k1_INTC_s105
                                 + 1.21672 w_k1_KO_s105
                                 + 0.901618 w_k1_MAT_s105
                                 + 1.18167 w_k1_MSFT_s105
                                 + 1.15753 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.815874 w_k1_ADM_s105
                                 + 0.888117 w_k1_AXP_s105
                                 + 0.75543 w_k1_DIS_s105
                                 + 0.848902 w_k1_FDX_s105
                                 + 1.06573 w_k1_IBM_s105
                                 + 0.915688 w_k1_INTC_s105
                                 + 0.886638 w_k1_KO_s105
                                 + 1.54827 w_k1_MAT_s105
                                 + 0.826641 w_k1_MSFT_s105
                                 + 0.77212 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.743434 w_k1_ADM_s105
                                 + 0.889724 w_k1_AXP_s105
                                 + 0.759892 w_k1_DIS_s105
                                 + 0.943523 w_k1_FDX_s105
                                 + 0.667047 w_k1_IBM_s105
                                 + 0.96171 w_k1_INTC_s105
                                 + 0.681343 w_k1_KO_s105
                                 + 0.871578 w_k1_MAT_s105
                                 + 0.812455 w_k1_MSFT_s105
                                 + 0.796667 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.32579 w_k1_ADM_s105
                                 + 1.04758 w_k1_AXP_s105
                                 + 1.20648 w_k1_DIS_s105
                                 + 1.00522 w_k1_FDX_s105
                                 + 1.34592 w_k1_IBM_s105
                                 + 1.06893 w_k1_INTC_s105
                                 + 1.08956 w_k1_KO_s105 + 1.31102 w_k1_MAT_s105
                                 + 0.930863 w_k1_MSFT_s105
                                 + 1.03245 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.78844 w_k1_ADM_s105
                                 + 0.801027 w_k1_AXP_s105
                                 + 0.50522 w_k1_DIS_s105
                                 + 0.689051 w_k1_FDX_s105
                                 + 0.759621 w_k1_IBM_s105
                                 + 0.68383 w_k1_INTC_s105
                                 + 0.636091 w_k1_KO_s105
                                 + 0.820854 w_k1_MAT_s105
                                 + 0.545703 w_k1_MSFT_s105
                                 + 0.634348 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.34494 w_k1_ADM_s106
                                 + 1.19385 w_k1_AXP_s106
                                 + 1.36006 w_k1_DIS_s106
                                 + 1.28323 w_k1_FDX_s106
                                 + 0.920351 w_k1_IBM_s106
                                 + 0.999472 w_k1_INTC_s106
                                 + 1.22903 w_k1_KO_s106
                                 + 0.847183 w_k1_MAT_s106
                                 + 1.31287 w_k1_MSFT_s106
                                 + 1.15876 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.824522 w_k1_ADM_s106
                                 + 0.689238 w_k1_AXP_s106
                                 + 0.604739 w_k1_DIS_s106
                                 + 0.729156 w_k1_FDX_s106
                                 + 0.88048 w_k1_IBM_s106
                                 + 0.845698 w_k1_INTC_s106
                                 + 0.799885 w_k1_KO_s106
                                 + 0.679915 w_k1_MAT_s106
                                 + 0.943094 w_k1_MSFT_s106
                                 + 0.832732 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.23127 w_k1_ADM_s106
                                 + 1.29004 w_k1_AXP_s106
                                 + 1.03419 w_k1_DIS_s106
                                 + 1.15562 w_k1_FDX_s106
                                 + 1.15592 w_k1_IBM_s106
                                 + 0.992329 w_k1_INTC_s106
                                 + 1.04254 w_k1_KO_s106
                                 + 0.890491 w_k1_MAT_s106
                                 + 0.998972 w_k1_MSFT_s106
                                 + 0.992509 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.924684 w_k1_ADM_s106
                                 + 0.74928 w_k1_AXP_s106
                                 + 0.825307 w_k1_DIS_s106
                                 + 0.910114 w_k1_FDX_s106
                                 + 0.697675 w_k1_IBM_s106
                                 + 0.876432 w_k1_INTC_s106
                                 + 0.832784 w_k1_KO_s106 + 0.6591 w_k1_MAT_s106
                                 + 0.96334 w_k1_MSFT_s106
                                 + 0.940644 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.26512 w_k1_ADM_s106
                                 + 1.24246 w_k1_AXP_s106
                                 + 1.19074 w_k1_DIS_s106 + 1.0609 w_k1_FDX_s106
                                 + 1.64154 w_k1_IBM_s106
                                 + 0.792758 w_k1_INTC_s106
                                 + 1.21149 w_k1_KO_s106 + 1.90272 w_k1_MAT_s106
                                 + 1.11437 w_k1_MSFT_s106
                                 + 1.07282 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.4008 w_k1_ADM_s106
                                 + 1.42873 w_k1_AXP_s106
                                 + 1.74777 w_k1_DIS_s106
                                 + 1.38169 w_k1_FDX_s106 + 1.3934 w_k1_IBM_s106
                                 + 1.33844 w_k1_INTC_s106
                                 + 1.62561 w_k1_KO_s106 + 1.57454 w_k1_MAT_s106
                                 + 1.28753 w_k1_MSFT_s106
                                 + 1.11473 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.03606 w_k1_ADM_s106
                                 + 0.796819 w_k1_AXP_s106
                                 + 1.11483 w_k1_DIS_s106
                                 + 1.00299 w_k1_FDX_s106
                                 + 1.15794 w_k1_IBM_s106
                                 + 1.05288 w_k1_INTC_s106
                                 + 1.09279 w_k1_KO_s106
                                 + 0.748502 w_k1_MAT_s106
                                 + 1.18973 w_k1_MSFT_s106
                                 + 0.998033 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.98912 w_k1_ADM_s106
                                 + 1.11038 w_k1_AXP_s106
                                 + 0.881494 w_k1_DIS_s106
                                 + 0.954979 w_k1_FDX_s106
                                 + 1.0626 w_k1_IBM_s106
                                 + 1.01423 w_k1_INTC_s106
                                 + 1.02888 w_k1_KO_s106 + 1.23342 w_k1_MAT_s106
                                 + 1.00793 w_k1_MSFT_s106
                                 + 0.922654 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.05275 w_k1_ADM_s107
                                 + 0.924553 w_k1_AXP_s107
                                 + 0.801156 w_k1_DIS_s107
                                 + 0.915452 w_k1_FDX_s107
                                 + 1.22776 w_k1_IBM_s107
                                 + 0.941373 w_k1_INTC_s107
                                 + 1.04523 w_k1_KO_s107 + 1.25472 w_k1_MAT_s107
                                 + 0.976292 w_k1_MSFT_s107
                                 + 0.844249 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.54677 w_k1_ADM_s107
                                 + 1.26681 w_k1_AXP_s107
                                 + 1.56163 w_k1_DIS_s107
                                 + 1.28118 w_k1_FDX_s107
                                 + 1.76385 w_k1_IBM_s107
                                 + 1.47355 w_k1_INTC_s107
                                 + 1.47738 w_k1_KO_s107 + 1.41226 w_k1_MAT_s107
                                 + 1.29029 w_k1_MSFT_s107
                                 + 1.11784 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.17144 w_k1_ADM_s107
                                 + 0.954596 w_k1_AXP_s107
                                 + 1.20465 w_k1_DIS_s107 + 1.2198 w_k1_FDX_s107
                                 + 0.845708 w_k1_IBM_s107
                                 + 1.3135 w_k1_INTC_s107
                                 + 0.982912 w_k1_KO_s107
                                 + 1.00848 w_k1_MAT_s107
                                 + 1.38622 w_k1_MSFT_s107
                                 + 0.970556 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.19706 w_k1_ADM_s107
                                 + 1.07009 w_k1_AXP_s107 + 1.2708 w_k1_DIS_s107
                                 + 1.05518 w_k1_FDX_s107
                                 + 1.19222 w_k1_IBM_s107
                                 + 1.27455 w_k1_INTC_s107
                                 + 1.16921 w_k1_KO_s107 + 1.28309 w_k1_MAT_s107
                                 + 1.36867 w_k1_MSFT_s107 + 1.0148 w_k1_PG_s107
                                 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.888492 w_k1_ADM_s107
                                 + 0.774998 w_k1_AXP_s107
                                 + 0.927618 w_k1_DIS_s107
                                 + 0.919361 w_k1_FDX_s107
                                 + 0.617185 w_k1_IBM_s107
                                 + 1.04537 w_k1_INTC_s107
                                 + 0.742966 w_k1_KO_s107
                                 + 0.481955 w_k1_MAT_s107
                                 + 1.1569 w_k1_MSFT_s107
                                 + 0.852935 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.841927 w_k1_ADM_s107
                                 + 1.16338 w_k1_AXP_s107
                                 + 0.710764 w_k1_DIS_s107
                                 + 1.01493 w_k1_FDX_s107
                                 + 0.666275 w_k1_IBM_s107
                                 + 1.05788 w_k1_INTC_s107
                                 + 0.727123 w_k1_KO_s107
                                 + 0.854537 w_k1_MAT_s107
                                 + 1.0995 w_k1_MSFT_s107 + 0.86961 w_k1_PG_s107
                                 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.60324 w_k1_ADM_s107
                                 + 1.35965 w_k1_AXP_s107
                                 + 1.46718 w_k1_DIS_s107
                                 + 1.39871 w_k1_FDX_s107 + 1.0237 w_k1_IBM_s107
                                 + 1.21831 w_k1_INTC_s107
                                 + 1.16418 w_k1_KO_s107 + 1.17319 w_k1_MAT_s107
                                 + 1.22608 w_k1_MSFT_s107
                                 + 1.02623 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.884327 w_k1_ADM_s107
                                 + 0.761156 w_k1_AXP_s107
                                 + 0.743875 w_k1_DIS_s107
                                 + 0.810617 w_k1_FDX_s107
                                 + 0.828131 w_k1_IBM_s107
                                 + 0.872601 w_k1_INTC_s107
                                 + 0.79859 w_k1_KO_s107 + 0.74848 w_k1_MAT_s107
                                 + 0.920247 w_k1_MSFT_s107
                                 + 0.804056 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.778759 w_k1_ADM_s110
                                 + 0.741622 w_k1_AXP_s110
                                 + 0.589004 w_k1_DIS_s110
                                 + 0.727488 w_k1_FDX_s110
                                 + 0.591087 w_k1_IBM_s110
                                 + 0.886401 w_k1_INTC_s110
                                 + 0.623853 w_k1_KO_s110
                                 + 0.367778 w_k1_MAT_s110
                                 + 0.803529 w_k1_MSFT_s110
                                 + 0.739365 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.862712 w_k1_ADM_s110
                                 + 1.10591 w_k1_AXP_s110
                                 + 0.806545 w_k1_DIS_s110
                                 + 0.888629 w_k1_FDX_s110
                                 + 1.04761 w_k1_IBM_s110
                                 + 0.798233 w_k1_INTC_s110
                                 + 0.887669 w_k1_KO_s110 + 1.3393 w_k1_MAT_s110
                                 + 0.934372 w_k1_MSFT_s110
                                 + 0.945645 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.20162 w_k1_ADM_s110
                                 + 1.33369 w_k1_AXP_s110
                                 + 1.58537 w_k1_DIS_s110
                                 + 1.34041 w_k1_FDX_s110
                                 + 1.03061 w_k1_IBM_s110
                                 + 1.11299 w_k1_INTC_s110
                                 + 1.15662 w_k1_KO_s110 + 1.21308 w_k1_MAT_s110
                                 + 1.21642 w_k1_MSFT_s110
                                 + 1.11696 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.739542 w_k1_ADM_s110
                                 + 0.912604 w_k1_AXP_s110
                                 + 0.977537 w_k1_DIS_s110
                                 + 1.02558 w_k1_FDX_s110
                                 + 0.826387 w_k1_IBM_s110
                                 + 1.06373 w_k1_INTC_s110
                                 + 0.902171 w_k1_KO_s110
                                 + 0.797638 w_k1_MAT_s110
                                 + 1.09065 w_k1_MSFT_s110
                                 + 0.956449 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.92162 w_k1_ADM_s110
                                 + 0.73477 w_k1_AXP_s110
                                 + 0.887068 w_k1_DIS_s110
                                 + 0.966827 w_k1_FDX_s110
                                 + 0.614627 w_k1_IBM_s110
                                 + 0.806488 w_k1_INTC_s110
                                 + 0.696596 w_k1_KO_s110
                                 + 0.474922 w_k1_MAT_s110
                                 + 0.934744 w_k1_MSFT_s110
                                 + 0.857143 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.705496 w_k1_ADM_s110
                                 + 0.832032 w_k1_AXP_s110
                                 + 0.695214 w_k1_DIS_s110
                                 + 0.769253 w_k1_FDX_s110
                                 + 0.972645 w_k1_IBM_s110
                                 + 0.92157 w_k1_INTC_s110
                                 + 0.893722 w_k1_KO_s110
                                 + 0.74516 w_k1_MAT_s110
                                 + 0.924564 w_k1_MSFT_s110
                                 + 0.815532 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.924778 w_k1_ADM_s110
                                 + 1.23251 w_k1_AXP_s110
                                 + 1.25783 w_k1_DIS_s110
                                 + 1.07377 w_k1_FDX_s110
                                 + 0.990352 w_k1_IBM_s110
                                 + 1.10571 w_k1_INTC_s110
                                 + 1.08614 w_k1_KO_s110 + 0.91699 w_k1_MAT_s110
                                 + 1.0956 w_k1_MSFT_s110 + 1.02392 w_k1_PG_s110
                                 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.10666 w_k1_ADM_s110
                                 + 1.21769 w_k1_AXP_s110
                                 + 1.06607 w_k1_DIS_s110
                                 + 1.11653 w_k1_FDX_s110 + 1.1868 w_k1_IBM_s110
                                 + 1.06059 w_k1_INTC_s110
                                 + 0.953853 w_k1_KO_s110
                                 + 1.18369 w_k1_MAT_s110
                                 + 1.15345 w_k1_MSFT_s110
                                 + 1.13331 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.16781 w_k1_ADM_s111
                                 + 1.19679 w_k1_AXP_s111
                                 + 1.28147 w_k1_DIS_s111
                                 + 1.35676 w_k1_FDX_s111
                                 + 0.703771 w_k1_IBM_s111
                                 + 1.13627 w_k1_INTC_s111
                                 + 0.988276 w_k1_KO_s111
                                 + 0.780015 w_k1_MAT_s111
                                 + 1.13691 w_k1_MSFT_s111
                                 + 1.20989 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.978111 w_k1_ADM_s111
                                 + 0.97137 w_k1_AXP_s111
                                 + 1.14462 w_k1_DIS_s111
                                 + 1.08276 w_k1_FDX_s111
                                 + 1.12801 w_k1_IBM_s111
                                 + 1.07998 w_k1_INTC_s111
                                 + 1.29566 w_k1_KO_s111 + 1.33417 w_k1_MAT_s111
                                 + 1.1346 w_k1_MSFT_s111 + 1.15174 w_k1_PG_s111
                                 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.41345 w_k1_ADM_s111
                                 + 1.32975 w_k1_AXP_s111
                                 + 1.48413 w_k1_DIS_s111
                                 + 1.37276 w_k1_FDX_s111
                                 + 1.15063 w_k1_IBM_s111
                                 + 1.10898 w_k1_INTC_s111
                                 + 1.20073 w_k1_KO_s111 + 1.44521 w_k1_MAT_s111
                                 + 1.41824 w_k1_MSFT_s111
                                 + 1.29242 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.12597 w_k1_ADM_s111
                                 + 1.48961 w_k1_AXP_s111
                                 + 1.38596 w_k1_DIS_s111
                                 + 1.20666 w_k1_FDX_s111
                                 + 1.61449 w_k1_IBM_s111
                                 + 1.41931 w_k1_INTC_s111
                                 + 1.57812 w_k1_KO_s111 + 1.60986 w_k1_MAT_s111
                                 + 1.15675 w_k1_MSFT_s111
                                 + 1.15919 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.04676 w_k1_ADM_s111
                                 + 0.995193 w_k1_AXP_s111
                                 + 0.890903 w_k1_DIS_s111
                                 + 0.913265 w_k1_FDX_s111
                                 + 1.30351 w_k1_IBM_s111
                                 + 0.933043 w_k1_INTC_s111
                                 + 1.09481 w_k1_KO_s111 + 1.38465 w_k1_MAT_s111
                                 + 0.993105 w_k1_MSFT_s111
                                 + 1.0042 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.797377 w_k1_ADM_s111
                                 + 0.653899 w_k1_AXP_s111
                                 + 0.747271 w_k1_DIS_s111
                                 + 0.769384 w_k1_FDX_s111
                                 + 1.04583 w_k1_IBM_s111
                                 + 0.847151 w_k1_INTC_s111
                                 + 0.973277 w_k1_KO_s111
                                 + 0.837985 w_k1_MAT_s111
                                 + 0.787855 w_k1_MSFT_s111
                                 + 0.877462 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.842635 w_k1_ADM_s111
                                 + 0.819988 w_k1_AXP_s111
                                 + 0.645469 w_k1_DIS_s111
                                 + 0.781946 w_k1_FDX_s111
                                 + 0.88292 w_k1_IBM_s111
                                 + 0.9568 w_k1_INTC_s111 + 0.83937 w_k1_KO_s111
                                 + 1.05127 w_k1_MAT_s111
                                 + 0.851844 w_k1_MSFT_s111
                                 + 0.864157 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.963441 w_k1_ADM_s111
                                 + 0.826509 w_k1_AXP_s111
                                 + 0.955106 w_k1_DIS_s111
                                 + 1.03956 w_k1_FDX_s111
                                 + 0.794111 w_k1_IBM_s111
                                 + 1.05566 w_k1_INTC_s111
                                 + 0.966786 w_k1_KO_s111
                                 + 0.794568 w_k1_MAT_s111
                                 + 0.985042 w_k1_MSFT_s111
                                 + 1.07049 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.28889 w_k1_ADM_s112
                                 + 1.24203 w_k1_AXP_s112
                                 + 1.15709 w_k1_DIS_s112
                                 + 1.19805 w_k1_FDX_s112
                                 + 0.970412 w_k1_IBM_s112
                                 + 1.0107 w_k1_INTC_s112
                                 + 0.987586 w_k1_KO_s112
                                 + 1.07912 w_k1_MAT_s112
                                 + 1.10979 w_k1_MSFT_s112 + 1.0168 w_k1_PG_s112
                                 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.13897 w_k1_ADM_s112
                                 + 0.817532 w_k1_AXP_s112
                                 + 1.06052 w_k1_DIS_s112
                                 + 1.05271 w_k1_FDX_s112
                                 + 0.925994 w_k1_IBM_s112
                                 + 1.08029 w_k1_INTC_s112
                                 + 1.14917 w_k1_KO_s112
                                 + 0.712853 w_k1_MAT_s112
                                 + 1.08122 w_k1_MSFT_s112
                                 + 1.08469 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.10608 w_k1_ADM_s112
                                 + 0.722704 w_k1_AXP_s112
                                 + 0.881812 w_k1_DIS_s112
                                 + 0.937571 w_k1_FDX_s112
                                 + 0.74731 w_k1_IBM_s112
                                 + 0.889417 w_k1_INTC_s112
                                 + 0.974132 w_k1_KO_s112
                                 + 0.762391 w_k1_MAT_s112
                                 + 0.915501 w_k1_MSFT_s112
                                 + 0.924356 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.00041 w_k1_ADM_s112
                                 + 0.934671 w_k1_AXP_s112
                                 + 0.743193 w_k1_DIS_s112
                                 + 0.833424 w_k1_FDX_s112
                                 + 1.00064 w_k1_IBM_s112
                                 + 0.786579 w_k1_INTC_s112
                                 + 0.902457 w_k1_KO_s112
                                 + 0.987481 w_k1_MAT_s112
                                 + 0.846235 w_k1_MSFT_s112
                                 + 0.903059 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.40235 w_k1_ADM_s112
                                 + 1.36569 w_k1_AXP_s112
                                 + 1.97433 w_k1_DIS_s112
                                 + 1.42608 w_k1_FDX_s112
                                 + 1.32748 w_k1_IBM_s112
                                 + 1.31963 w_k1_INTC_s112
                                 + 1.48977 w_k1_KO_s112 + 1.08156 w_k1_MAT_s112
                                 + 1.43047 w_k1_MSFT_s112
                                 + 1.26406 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.1895 w_k1_ADM_s112
                                 + 1.26062 w_k1_AXP_s112
                                 + 1.39493 w_k1_DIS_s112
                                 + 1.04404 w_k1_FDX_s112
                                 + 1.43428 w_k1_IBM_s112
                                 + 1.15358 w_k1_INTC_s112
                                 + 1.31534 w_k1_KO_s112 + 1.1393 w_k1_MAT_s112
                                 + 1.15544 w_k1_MSFT_s112
                                 + 1.15587 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.06367 w_k1_ADM_s112
                                 + 0.832679 w_k1_AXP_s112
                                 + 0.668993 w_k1_DIS_s112
                                 + 0.781714 w_k1_FDX_s112
                                 + 0.75406 w_k1_IBM_s112
                                 + 0.767338 w_k1_INTC_s112
                                 + 0.70055 w_k1_KO_s112
                                 + 0.666057 w_k1_MAT_s112
                                 + 0.800744 w_k1_MSFT_s112
                                 + 0.859308 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.11051 w_k1_ADM_s112
                                 + 1.25929 w_k1_AXP_s112
                                 + 0.88205 w_k1_DIS_s112
                                 + 1.00135 w_k1_FDX_s112
                                 + 1.20862 w_k1_IBM_s112
                                 + 0.932271 w_k1_INTC_s112
                                 + 1.1291 w_k1_KO_s112 + 1.38345 w_k1_MAT_s112
                                 + 0.999161 w_k1_MSFT_s112
                                 + 1.06035 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.05401 w_k1_ADM_s113
                                 + 0.97839 w_k1_AXP_s113
                                 + 1.32439 w_k1_DIS_s113
                                 + 1.05027 w_k1_FDX_s113
                                 + 1.13372 w_k1_IBM_s113
                                 + 1.3515 w_k1_INTC_s113 + 1.44148 w_k1_KO_s113
                                 + 1.09951 w_k1_MAT_s113
                                 + 1.15488 w_k1_MSFT_s113
                                 + 1.08018 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.00548 w_k1_ADM_s113
                                 + 1.35118 w_k1_AXP_s113
                                 + 1.08501 w_k1_DIS_s113
                                 + 1.15803 w_k1_FDX_s113
                                 + 1.09763 w_k1_IBM_s113
                                 + 1.1329 w_k1_INTC_s113 + 1.22693 w_k1_KO_s113
                                 + 1.15922 w_k1_MAT_s113
                                 + 1.10194 w_k1_MSFT_s113
                                 + 1.13561 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.30071 w_k1_ADM_s113
                                 + 1.35379 w_k1_AXP_s113
                                 + 1.57764 w_k1_DIS_s113
                                 + 1.31403 w_k1_FDX_s113
                                 + 0.900613 w_k1_IBM_s113
                                 + 1.25984 w_k1_INTC_s113
                                 + 1.47835 w_k1_KO_s113
                                 + 0.807656 w_k1_MAT_s113
                                 + 1.19921 w_k1_MSFT_s113
                                 + 1.16587 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.774838 w_k1_ADM_s113
                                 + 0.769193 w_k1_AXP_s113
                                 + 0.732404 w_k1_DIS_s113
                                 + 0.769947 w_k1_FDX_s113
                                 + 0.935102 w_k1_IBM_s113
                                 + 1.06858 w_k1_INTC_s113
                                 + 1.10149 w_k1_KO_s113 + 0.72938 w_k1_MAT_s113
                                 + 0.84058 w_k1_MSFT_s113
                                 + 0.916294 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.891389 w_k1_ADM_s113
                                 + 0.828629 w_k1_AXP_s113
                                 + 0.91027 w_k1_DIS_s113
                                 + 0.995084 w_k1_FDX_s113
                                 + 0.652676 w_k1_IBM_s113
                                 + 1.02459 w_k1_INTC_s113
                                 + 0.965793 w_k1_KO_s113
                                 + 0.717332 w_k1_MAT_s113
                                 + 1.07187 w_k1_MSFT_s113
                                 + 0.907578 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.845922 w_k1_ADM_s113
                                 + 1.13963 w_k1_AXP_s113
                                 + 1.05437 w_k1_DIS_s113
                                 + 0.965247 w_k1_FDX_s113
                                 + 1.46355 w_k1_IBM_s113
                                 + 1.05549 w_k1_INTC_s113
                                 + 1.34749 w_k1_KO_s113 + 1.58232 w_k1_MAT_s113
                                 + 1.00074 w_k1_MSFT_s113
                                 + 1.13943 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.06458 w_k1_ADM_s113
                                 + 1.06684 w_k1_AXP_s113
                                 + 0.871632 w_k1_DIS_s113
                                 + 0.971843 w_k1_FDX_s113
                                 + 0.892578 w_k1_IBM_s113
                                 + 1.26563 w_k1_INTC_s113
                                 + 1.00956 w_k1_KO_s113
                                 + 0.761048 w_k1_MAT_s113
                                 + 0.971773 w_k1_MSFT_s113
                                 + 1.04562 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.17001 w_k1_ADM_s113
                                 + 1.76443 w_k1_AXP_s113
                                 + 1.77964 w_k1_DIS_s113
                                 + 1.44891 w_k1_FDX_s113
                                 + 1.51516 w_k1_IBM_s113
                                 + 1.70538 w_k1_INTC_s113
                                 + 1.67871 w_k1_KO_s113 + 1.54923 w_k1_MAT_s113
                                 + 1.49248 w_k1_MSFT_s113
                                 + 1.28242 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.873162 w_k1_ADM_s114
                                 + 0.742693 w_k1_AXP_s114
                                 + 0.679544 w_k1_DIS_s114
                                 + 0.746658 w_k1_FDX_s114
                                 + 0.995515 w_k1_IBM_s114
                                 + 0.66444 w_k1_INTC_s114
                                 + 0.806341 w_k1_KO_s114
                                 + 0.924935 w_k1_MAT_s114
                                 + 0.780076 w_k1_MSFT_s114
                                 + 0.903886 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.028 w_k1_ADM_s114
                                 + 0.968835 w_k1_AXP_s114
                                 + 1.25175 w_k1_DIS_s114
                                 + 1.06993 w_k1_FDX_s114 + 1.1568 w_k1_IBM_s114
                                 + 1.04448 w_k1_INTC_s114
                                 + 1.14897 w_k1_KO_s114 + 1.02815 w_k1_MAT_s114
                                 + 1.13544 w_k1_MSFT_s114
                                 + 1.08788 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.20289 w_k1_ADM_s114
                                 + 1.32195 w_k1_AXP_s114
                                 + 1.22214 w_k1_DIS_s114
                                 + 1.09357 w_k1_FDX_s114
                                 + 1.54303 w_k1_IBM_s114
                                 + 1.05586 w_k1_INTC_s114 + 1.2981 w_k1_KO_s114
                                 + 1.92804 w_k1_MAT_s114
                                 + 1.2359 w_k1_MSFT_s114 + 1.12342 w_k1_PG_s114
                                 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.2078 w_k1_ADM_s114
                                 + 1.00874 w_k1_AXP_s114
                                 + 1.35071 w_k1_DIS_s114
                                 + 1.20595 w_k1_FDX_s114
                                 + 0.970142 w_k1_IBM_s114
                                 + 1.28074 w_k1_INTC_s114
                                 + 1.24395 w_k1_KO_s114
                                 + 0.814244 w_k1_MAT_s114
                                 + 1.37921 w_k1_MSFT_s114
                                 + 1.08178 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.03523 w_k1_ADM_s114
                                 + 0.993145 w_k1_AXP_s114
                                 + 1.03602 w_k1_DIS_s114
                                 + 1.05175 w_k1_FDX_s114
                                 + 0.735112 w_k1_IBM_s114
                                 + 1.05183 w_k1_INTC_s114
                                 + 0.949616 w_k1_KO_s114
                                 + 0.849912 w_k1_MAT_s114
                                 + 0.995977 w_k1_MSFT_s114
                                 + 0.990893 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.938542 w_k1_ADM_s114
                                 + 0.814812 w_k1_AXP_s114
                                 + 0.861555 w_k1_DIS_s114
                                 + 1.0236 w_k1_FDX_s114
                                 + 0.491343 w_k1_IBM_s114
                                 + 1.02815 w_k1_INTC_s114
                                 + 0.659222 w_k1_KO_s114
                                 + 0.403588 w_k1_MAT_s114
                                 + 0.950244 w_k1_MSFT_s114
                                 + 0.815096 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.25353 w_k1_ADM_s114
                                 + 1.38706 w_k1_AXP_s114
                                 + 1.64335 w_k1_DIS_s114
                                 + 1.44152 w_k1_FDX_s114
                                 + 0.917891 w_k1_IBM_s114
                                 + 1.19445 w_k1_INTC_s114
                                 + 1.16616 w_k1_KO_s114 + 1.25573 w_k1_MAT_s114
                                 + 1.34624 w_k1_MSFT_s114
                                 + 1.23685 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.22365 w_k1_ADM_s114
                                 + 1.1335 w_k1_AXP_s114 + 1.02644 w_k1_DIS_s114
                                 + 0.988906 w_k1_FDX_s114
                                 + 1.10736 w_k1_IBM_s114
                                 + 0.880246 w_k1_INTC_s114
                                 + 1.03275 w_k1_KO_s114 + 1.16646 w_k1_MAT_s114
                                 + 0.98616 w_k1_MSFT_s114
                                 + 0.934079 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.922039 w_k1_ADM_s115
                                 + 0.645345 w_k1_AXP_s115
                                 + 0.723971 w_k1_DIS_s115
                                 + 0.806688 w_k1_FDX_s115
                                 + 0.608852 w_k1_IBM_s115
                                 + 0.757228 w_k1_INTC_s115
                                 + 0.686239 w_k1_KO_s115
                                 + 0.552787 w_k1_MAT_s115
                                 + 0.786998 w_k1_MSFT_s115
                                 + 0.82211 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.32108 w_k1_ADM_s115
                                 + 1.29547 w_k1_AXP_s115
                                 + 1.78057 w_k1_DIS_s115
                                 + 1.19738 w_k1_FDX_s115
                                 + 1.19342 w_k1_IBM_s115
                                 + 1.20578 w_k1_INTC_s115
                                 + 1.30145 w_k1_KO_s115 + 1.15285 w_k1_MAT_s115
                                 + 1.20392 w_k1_MSFT_s115
                                 + 1.21584 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.26078 w_k1_ADM_s115
                                 + 1.55182 w_k1_AXP_s115 + 1.3927 w_k1_DIS_s115
                                 + 1.24559 w_k1_FDX_s115 + 1.3451 w_k1_IBM_s115
                                 + 1.14145 w_k1_INTC_s115
                                 + 1.24136 w_k1_KO_s115 + 1.29583 w_k1_MAT_s115
                                 + 0.994934 w_k1_MSFT_s115
                                 + 1.25943 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.18778 w_k1_ADM_s115
                                 + 1.3755 w_k1_AXP_s115 + 1.10304 w_k1_DIS_s115
                                 + 1.01478 w_k1_FDX_s115
                                 + 1.46523 w_k1_IBM_s115
                                 + 0.868141 w_k1_INTC_s115
                                 + 1.18708 w_k1_KO_s115 + 1.4654 w_k1_MAT_s115
                                 + 0.980573 w_k1_MSFT_s115
                                 + 1.04062 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.895842 w_k1_ADM_s115
                                 + 1.00103 w_k1_AXP_s115
                                 + 0.818579 w_k1_DIS_s115
                                 + 0.914975 w_k1_FDX_s115
                                 + 0.886122 w_k1_IBM_s115
                                 + 1.02242 w_k1_INTC_s115
                                 + 0.850159 w_k1_KO_s115
                                 + 0.874623 w_k1_MAT_s115
                                 + 0.882164 w_k1_MSFT_s115
                                 + 0.876168 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.29255 w_k1_ADM_s115
                                 + 1.23613 w_k1_AXP_s115
                                 + 1.22312 w_k1_DIS_s115
                                 + 1.23647 w_k1_FDX_s115 + 0.8791 w_k1_IBM_s115
                                 + 1.23018 w_k1_INTC_s115
                                 + 1.09222 w_k1_KO_s115
                                 + 0.762282 w_k1_MAT_s115
                                 + 1.14146 w_k1_MSFT_s115 + 1.088 w_k1_PG_s115
                                 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.925875 w_k1_ADM_s115
                                 + 0.866354 w_k1_AXP_s115
                                 + 0.641473 w_k1_DIS_s115
                                 + 0.737298 w_k1_FDX_s115
                                 + 1.00717 w_k1_IBM_s115
                                 + 0.70596 w_k1_INTC_s115
                                 + 0.857667 w_k1_KO_s115
                                 + 0.989893 w_k1_MAT_s115
                                 + 0.686355 w_k1_MSFT_s115
                                 + 0.881247 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.01211 w_k1_ADM_s115
                                 + 0.900008 w_k1_AXP_s115
                                 + 1.09151 w_k1_DIS_s115
                                 + 1.02094 w_k1_FDX_s115
                                 + 0.978759 w_k1_IBM_s115
                                 + 1.06293 w_k1_INTC_s115
                                 + 1.03646 w_k1_KO_s115
                                 + 0.861161 w_k1_MAT_s115
                                 + 1.06564 w_k1_MSFT_s115
                                 + 1.00231 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.42115 w_k1_ADM_s116
                                 + 1.61571 w_k1_AXP_s116 + 1.7185 w_k1_DIS_s116
                                 + 1.44138 w_k1_FDX_s116
                                 + 1.26109 w_k1_IBM_s116
                                 + 1.45562 w_k1_INTC_s116
                                 + 1.50476 w_k1_KO_s116 + 1.94811 w_k1_MAT_s116
                                 + 1.38712 w_k1_MSFT_s116
                                 + 1.37899 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.821957 w_k1_ADM_s116
                                 + 0.879341 w_k1_AXP_s116
                                 + 0.872953 w_k1_DIS_s116
                                 + 1.01114 w_k1_FDX_s116
                                 + 0.764319 w_k1_IBM_s116
                                 + 0.90947 w_k1_INTC_s116
                                 + 0.916029 w_k1_KO_s116
                                 + 0.837057 w_k1_MAT_s116
                                 + 0.917022 w_k1_MSFT_s116
                                 + 0.959753 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.13015 w_k1_ADM_s116
                                 + 1.17896 w_k1_AXP_s116
                                 + 1.31029 w_k1_DIS_s116
                                 + 1.13661 w_k1_FDX_s116
                                 + 1.21716 w_k1_IBM_s116
                                 + 1.05568 w_k1_INTC_s116
                                 + 1.27365 w_k1_KO_s116 + 1.4763 w_k1_MAT_s116
                                 + 1.20909 w_k1_MSFT_s116
                                 + 1.12897 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.945622 w_k1_ADM_s116
                                 + 1.10353 w_k1_AXP_s116
                                 + 1.22856 w_k1_DIS_s116
                                 + 1.09805 w_k1_FDX_s116
                                 + 1.04066 w_k1_IBM_s116
                                 + 1.13984 w_k1_INTC_s116
                                 + 1.26816 w_k1_KO_s116 + 1.14513 w_k1_MAT_s116
                                 + 1.114 w_k1_MSFT_s116 + 1.15839 w_k1_PG_s116
                                 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.03395 w_k1_ADM_s116
                                 + 1.43948 w_k1_AXP_s116 + 1.3682 w_k1_DIS_s116
                                 + 1.29669 w_k1_FDX_s116
                                 + 0.801816 w_k1_IBM_s116
                                 + 1.06093 w_k1_INTC_s116
                                 + 1.16972 w_k1_KO_s116 + 1.37484 w_k1_MAT_s116
                                 + 1.15668 w_k1_MSFT_s116
                                 + 1.15322 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.586513 w_k1_ADM_s116
                                 + 0.632025 w_k1_AXP_s116
                                 + 0.609241 w_k1_DIS_s116
                                 + 0.659798 w_k1_FDX_s116
                                 + 0.798819 w_k1_IBM_s116
                                 + 0.845471 w_k1_INTC_s116
                                 + 0.799534 w_k1_KO_s116
                                 + 0.763287 w_k1_MAT_s116
                                 + 0.863997 w_k1_MSFT_s116
                                 + 0.798087 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.822015 w_k1_ADM_s116
                                 + 0.902219 w_k1_AXP_s116
                                 + 0.932828 w_k1_DIS_s116
                                 + 0.855631 w_k1_FDX_s116
                                 + 1.21898 w_k1_IBM_s116
                                 + 0.974194 w_k1_INTC_s116
                                 + 1.08963 w_k1_KO_s116 + 1.28696 w_k1_MAT_s116
                                 + 1.03228 w_k1_MSFT_s116
                                 + 1.00862 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.952673 w_k1_ADM_s116
                                 + 1.15042 w_k1_AXP_s116
                                 + 1.35545 w_k1_DIS_s116
                                 + 1.08285 w_k1_FDX_s116
                                 + 1.78376 w_k1_IBM_s116
                                 + 1.21834 w_k1_INTC_s116
                                 + 1.55871 w_k1_KO_s116 + 2.22485 w_k1_MAT_s116
                                 + 1.07977 w_k1_MSFT_s116
                                 + 1.24063 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.11674 w_k1_ADM_s117
                                 + 0.994776 w_k1_AXP_s117
                                 + 1.60028 w_k1_DIS_s117
                                 + 1.26063 w_k1_FDX_s117
                                 + 1.06968 w_k1_IBM_s117
                                 + 1.82231 w_k1_INTC_s117
                                 + 1.42469 w_k1_KO_s117 + 1.08823 w_k1_MAT_s117
                                 + 1.37487 w_k1_MSFT_s117
                                 + 1.21856 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.713256 w_k1_ADM_s117
                                 + 0.719017 w_k1_AXP_s117
                                 + 0.690069 w_k1_DIS_s117
                                 + 0.742225 w_k1_FDX_s117
                                 + 0.815723 w_k1_IBM_s117
                                 + 0.977435 w_k1_INTC_s117
                                 + 0.752177 w_k1_KO_s117
                                 + 0.957638 w_k1_MAT_s117
                                 + 0.828143 w_k1_MSFT_s117
                                 + 0.836133 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.04794 w_k1_ADM_s117
                                 + 1.31836 w_k1_AXP_s117
                                 + 0.897865 w_k1_DIS_s117
                                 + 1.00569 w_k1_FDX_s117
                                 + 1.13437 w_k1_IBM_s117
                                 + 1.19889 w_k1_INTC_s117
                                 + 1.17076 w_k1_KO_s117 + 1.46112 w_k1_MAT_s117
                                 + 1.00248 w_k1_MSFT_s117
                                 + 1.12834 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.55581 w_k1_ADM_s117
                                 + 1.5693 w_k1_AXP_s117 + 1.68947 w_k1_DIS_s117
                                 + 1.34706 w_k1_FDX_s117
                                 + 1.50455 w_k1_IBM_s117
                                 + 1.47044 w_k1_INTC_s117
                                 + 1.46196 w_k1_KO_s117 + 1.35311 w_k1_MAT_s117
                                 + 1.26441 w_k1_MSFT_s117
                                 + 1.18378 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.01249 w_k1_ADM_s117
                                 + 0.904191 w_k1_AXP_s117
                                 + 0.967058 w_k1_DIS_s117
                                 + 1.03156 w_k1_FDX_s117
                                 + 0.610607 w_k1_IBM_s117
                                 + 1.16585 w_k1_INTC_s117
                                 + 0.914934 w_k1_KO_s117
                                 + 0.836158 w_k1_MAT_s117
                                 + 1.08789 w_k1_MSFT_s117
                                 + 0.946025 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.890011 w_k1_ADM_s117
                                 + 0.906217 w_k1_AXP_s117
                                 + 0.89264 w_k1_DIS_s117
                                 + 0.831488 w_k1_FDX_s117
                                 + 1.46587 w_k1_IBM_s117
                                 + 1.1852 w_k1_INTC_s117 + 1.15609 w_k1_KO_s117
                                 + 1.47855 w_k1_MAT_s117
                                 + 0.972454 w_k1_MSFT_s117
                                 + 0.954415 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.33474 w_k1_ADM_s117
                                 + 1.08512 w_k1_AXP_s117
                                 + 1.24541 w_k1_DIS_s117 + 1.3789 w_k1_FDX_s117
                                 + 0.793357 w_k1_IBM_s117
                                 + 1.45252 w_k1_INTC_s117
                                 + 1.16467 w_k1_KO_s117 + 1.05855 w_k1_MAT_s117
                                 + 1.38074 w_k1_MSFT_s117
                                 + 1.12566 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.07763 w_k1_ADM_s117
                                 + 0.947593 w_k1_AXP_s117
                                 + 0.991797 w_k1_DIS_s117
                                 + 1.00266 w_k1_FDX_s117
                                 + 1.04422 w_k1_IBM_s117
                                 + 1.32588 w_k1_INTC_s117
                                 + 1.01874 w_k1_KO_s117
                                 + 0.969337 w_k1_MAT_s117
                                 + 1.0634 w_k1_MSFT_s117 + 1.02841 w_k1_PG_s117
                                 >= 1
 k1_return_constraint:           0.00901134032925077 w_k1_ADM_s000
                                 + 0.00674848807439185 w_k1_ADM_s001
                                 + 0.00379719743528726 w_k1_ADM_s002
                                 + 0.00752233586464937 w_k1_ADM_s003
                                 + 0.0101992233317125 w_k1_ADM_s004
                                 + 0.0171747532291125 w_k1_ADM_s005
                                 + 0.00706440872771133 w_k1_ADM_s006
                                 + 0.0144582622387439 w_k1_ADM_s007
                                 + 0.00919968639047751 w_k1_ADM_s010
                                 + 0.0189679016002562 w_k1_ADM_s011
                                 + 0.00839169123798988 w_k1_ADM_s012
                                 + 0.0103595665866846 w_k1_ADM_s013
                                 + 0.0140481143287293 w_k1_ADM_s014
                                 + 0.0201278478978693 w_k1_ADM_s015
                                 + 0.00897510590222973 w_k1_ADM_s016
                                 + 0.0180152015651874 w_k1_ADM_s017
                                 + 0.0206668081985889 w_k1_ADM_s100
                                 + 0.0118732049626147 w_k1_ADM_s101
                                 + 0.0130749704412578 w_k1_ADM_s102
                                 + 0.0315594338935398 w_k1_ADM_s103
                                 + 0.0124852161939229 w_k1_ADM_s104
                                 + 0.0103115074709518 w_k1_ADM_s105
                                 + 0.0244114356175467 w_k1_ADM_s106
                                 + 0.038336020351902 w_k1_ADM_s107
                                 + 0.0642943306702468 w_k1_ADM_s110
                                 + 0.216631594716755 w_k1_ADM_s111
                                 + 0.0537021701252779 w_k1_ADM_s112
                                 + 0.0719790814209841 w_k1_ADM_s113
                                 + 0.0583896637193414 w_k1_ADM_s114
                                 + 0.0693391423452749 w_k1_ADM_s115
                                 + 0.0857454839598314 w_k1_ADM_s116
                                 + 0.051504251023415 w_k1_ADM_s117
                                 + 0.00815279918075751 w_k1_AXP_s000
                                 + 0.00585242526804383 w_k1_AXP_s001
                                 + 0.00390494431149446 w_k1_AXP_s002
                                 + 0.00780228269695699 w_k1_AXP_s003
                                 + 0.0117243705767605 w_k1_AXP_s004
                                 + 0.0178007490729288 w_k1_AXP_s005
                                 + 0.007000087487115 w_k1_AXP_s006
                                 + 0.0136379504174328 w_k1_AXP_s007
                                 + 0.00755757674424593 w_k1_AXP_s010
                                 + 0.0168170274443295 w_k1_AXP_s011
                                 + 0.00768615842101907 w_k1_AXP_s012
                                 + 0.0106909498154522 w_k1_AXP_s013
                                 + 0.0140043443863895 w_k1_AXP_s014
                                 + 0.0206044076326394 w_k1_AXP_s015
                                 + 0.00968946849009554 w_k1_AXP_s016
                                 + 0.0191986155474958 w_k1_AXP_s017
                                 + 0.0240216130292834 w_k1_AXP_s100
                                 + 0.0127198144764026 w_k1_AXP_s101
                                 + 0.0139116269337579 w_k1_AXP_s102
                                 + 0.0331817932064544 w_k1_AXP_s103
                                 + 0.0134013282044144 w_k1_AXP_s104
                                 + 0.00960841515962876 w_k1_AXP_s105
                                 + 0.0224722011881955 w_k1_AXP_s106
                                 + 0.0337009328637654 w_k1_AXP_s107
                                 + 0.0750593199316218 w_k1_AXP_s110
                                 + 0.20927521789752 w_k1_AXP_s111
                                 + 0.0488257455283955 w_k1_AXP_s112
                                 + 0.0788237179979452 w_k1_AXP_s113
                                 + 0.0560641783171847 w_k1_AXP_s114
                                 + 0.0701038322987038 w_k1_AXP_s115
                                 + 0.0971245422239711 w_k1_AXP_s116
                                 + 0.0482615384630171 w_k1_AXP_s117
                                 + 0.00878859039759114 w_k1_DIS_s000
                                 + 0.00512731860232918 w_k1_DIS_s001
                                 + 0.00368815603446743 w_k1_DIS_s002
                                 + 0.00729854038661371 w_k1_DIS_s003
                                 + 0.0113368942894135 w_k1_DIS_s004
                                 + 0.0172285279655795 w_k1_DIS_s005
                                 + 0.00728616043243872 w_k1_DIS_s006
                                 + 0.0137367955053631 w_k1_DIS_s007
                                 + 0.00893083977525399 w_k1_DIS_s010
                                 + 0.0186299964302036 w_k1_DIS_s011
                                 + 0.00882552670067909 w_k1_DIS_s012
                                 + 0.0107999406355527 w_k1_DIS_s013
                                 + 0.0148060162902179 w_k1_DIS_s014
                                 + 0.0219069449446616 w_k1_DIS_s015
                                 + 0.0107128146199752 w_k1_DIS_s016
                                 + 0.0208561538376042 w_k1_DIS_s017
                                 + 0.0251075441081052 w_k1_DIS_s100
                                 + 0.0103928692902782 w_k1_DIS_s101
                                 + 0.0135998972428599 w_k1_DIS_s102
                                 + 0.0311100568856511 w_k1_DIS_s103
                                 + 0.0151034726648687 w_k1_DIS_s104
                                 + 0.00955306833011457 w_k1_DIS_s105
                                 + 0.0238323600719303 w_k1_DIS_s106
                                 + 0.0361729472538086 w_k1_DIS_s107
                                 + 0.0733297712072274 w_k1_DIS_s110
                                 + 0.215270939490609 w_k1_DIS_s111
                                 + 0.0484484232412635 w_k1_DIS_s112
                                 + 0.0790819619737595 w_k1_DIS_s113
                                 + 0.0588461366111026 w_k1_DIS_s114
                                 + 0.0682064008803367 w_k1_DIS_s115
                                 + 0.101311594724409 w_k1_DIS_s116
                                 + 0.0514391226525392 w_k1_DIS_s117
                                 + 0.00869798580279992 w_k1_FDX_s000
                                 + 0.0053056351380375 w_k1_FDX_s001
                                 + 0.00364875897384611 w_k1_FDX_s002
                                 + 0.00773748860784917 w_k1_FDX_s003
                                 + 0.0120790865109132 w_k1_FDX_s004
                                 + 0.0164032128133856 w_k1_FDX_s005
                                 + 0.00679483377880086 w_k1_FDX_s006
                                 + 0.0137212027328358 w_k1_FDX_s007
                                 + 0.00784609646429855 w_k1_FDX_s010
                                 + 0.0175761376928874 w_k1_FDX_s011
                                 + 0.00892992413272919 w_k1_FDX_s012
                                 + 0.0105033412430335 w_k1_FDX_s013
                                 + 0.0134936996563683 w_k1_FDX_s014
                                 + 0.0192183228895759 w_k1_FDX_s015
                                 + 0.0103531227817628 w_k1_FDX_s016
                                 + 0.0187167794734141 w_k1_FDX_s017
                                 + 0.025304996564638 w_k1_FDX_s100
                                 + 0.00996073277307595 w_k1_FDX_s101
                                 + 0.0139886189934722 w_k1_FDX_s102
                                 + 0.0299309655131385 w_k1_FDX_s103
                                 + 0.0132430085443205 w_k1_FDX_s104
                                 + 0.00943703016593771 w_k1_FDX_s105
                                 + 0.0232861816131005 w_k1_FDX_s106
                                 + 0.0356281240546417 w_k1_FDX_s107
                                 + 0.071222434610714 w_k1_FDX_s110
                                 + 0.220643933497859 w_k1_FDX_s111
                                 + 0.0477674912827335 w_k1_FDX_s112
                                 + 0.0764264423521629 w_k1_FDX_s113
                                 + 0.0569482932654623 w_k1_FDX_s114
                                 + 0.0655912067568694 w_k1_FDX_s115
                                 + 0.0967987718385881 w_k1_FDX_s116
                                 + 0.0505687220203342 w_k1_FDX_s117
                                 + 0.00781633124827742 w_k1_IBM_s000
                                 + 0.00539092403464551 w_k1_IBM_s001
                                 + 0.00365514832827524 w_k1_IBM_s002
                                 + 0.00759241237553826 w_k1_IBM_s003
                                 + 0.0114473603333633 w_k1_IBM_s004
                                 + 0.0167534092402657 w_k1_IBM_s005
                                 + 0.00685636836724307 w_k1_IBM_s006
                                 + 0.0140663150199926 w_k1_IBM_s007
                                 + 0.00844639863206476 w_k1_IBM_s010
                                 + 0.0175327619920683 w_k1_IBM_s011
                                 + 0.0087734275087256 w_k1_IBM_s012
                                 + 0.00977160395444406 w_k1_IBM_s013
                                 + 0.0149805687580704 w_k1_IBM_s014
                                 + 0.0206913662443345 w_k1_IBM_s015
                                 + 0.0104892512206096 w_k1_IBM_s016
                                 + 0.0191550094281144 w_k1_IBM_s017
                                 + 0.0247084240786586 w_k1_IBM_s100
                                 + 0.0106621998876181 w_k1_IBM_s101
                                 + 0.0145727433271612 w_k1_IBM_s102
                                 + 0.0296302959366584 w_k1_IBM_s103
                                 + 0.013923016573658 w_k1_IBM_s104
                                 + 0.0097051520685242 w_k1_IBM_s105
                                 + 0.0228798253039523 w_k1_IBM_s106
                                 + 0.0348875716384686 w_k1_IBM_s107
                                 + 0.0668850910424832 w_k1_IBM_s110
                                 + 0.216034792401878 w_k1_IBM_s111
                                 + 0.0464846246862332 w_k1_IBM_s112
                                 + 0.0745387182780045 w_k1_IBM_s113
                                 + 0.0540895172394352 w_k1_IBM_s114
                                 + 0.0657548465655738 w_k1_IBM_s115
                                 + 0.0947878114251404 w_k1_IBM_s116
                                 + 0.0482049472133336 w_k1_IBM_s117
                                 + 0.0080197214862502 w_k1_INTC_s000
                                 + 0.00465499742801461 w_k1_INTC_s001
                                 + 0.00393000524394907 w_k1_INTC_s002
                                 + 0.00808492100857145 w_k1_INTC_s003
                                 + 0.0117007031375342 w_k1_INTC_s004
                                 + 0.0176325850286891 w_k1_INTC_s005
                                 + 0.00672476328540915 w_k1_INTC_s006
                                 + 0.0132661153794141 w_k1_INTC_s007
                                 + 0.00721741731051365 w_k1_INTC_s010
                                 + 0.0199097491297932 w_k1_INTC_s011
                                 + 0.0084162213660582 w_k1_INTC_s012
                                 + 0.00926404388466278 w_k1_INTC_s013
                                 + 0.0125274490917983 w_k1_INTC_s014
                                 + 0.0211698586481848 w_k1_INTC_s015
                                 + 0.0123945220202022 w_k1_INTC_s016
                                 + 0.0186903905070624 w_k1_INTC_s017
                                 + 0.0239920058934339 w_k1_INTC_s100
                                 + 0.0109046612944807 w_k1_INTC_s101
                                 + 0.014716133663604 w_k1_INTC_s102
                                 + 0.0324834118648783 w_k1_INTC_s103
                                 + 0.0156427779096986 w_k1_INTC_s104
                                 + 0.0102857001591595 w_k1_INTC_s105
                                 + 0.0222353710212227 w_k1_INTC_s106
                                 + 0.0385624009651639 w_k1_INTC_s107
                                 + 0.0690052015675031 w_k1_INTC_s110
                                 + 0.223839295309352 w_k1_INTC_s111
                                 + 0.0449144188246674 w_k1_INTC_s112
                                 + 0.0879580864879039 w_k1_INTC_s113
                                 + 0.0533703927192681 w_k1_INTC_s114
                                 + 0.0659966066495043 w_k1_INTC_s115
                                 + 0.0967381189331987 w_k1_INTC_s116
                                 + 0.0627519254450503 w_k1_INTC_s117
                                 + 0.00758351450019587 w_k1_KO_s000
                                 + 0.00449232157011645 w_k1_KO_s001
                                 + 0.00361800416475145 w_k1_KO_s002
                                 + 0.00673679148940417 w_k1_KO_s003
                                 + 0.0118344981360071 w_k1_KO_s004
                                 + 0.0155793084489592 w_k1_KO_s005
                                 + 0.00726399712597024 w_k1_KO_s006
                                 + 0.0141528879554123 w_k1_KO_s007
                                 + 0.00815388775689855 w_k1_KO_s010
                                 + 0.0193244865656463 w_k1_KO_s011
                                 + 0.00902513775863622 w_k1_KO_s012
                                 + 0.0114397308601663 w_k1_KO_s013
                                 + 0.0133124332642124 w_k1_KO_s014
                                 + 0.0212099519220255 w_k1_KO_s015
                                 + 0.0103811790833793 w_k1_KO_s016
                                 + 0.0215830675469336 w_k1_KO_s017
                                 + 0.0231233291812167 w_k1_KO_s100
                                 + 0.0112595385259933 w_k1_KO_s101
                                 + 0.0135092433738247 w_k1_KO_s102
                                 + 0.0295541976816391 w_k1_KO_s103
                                 + 0.0130514574500414 w_k1_KO_s104
                                 + 0.00894257032850397 w_k1_KO_s105
                                 + 0.0241212656055325 w_k1_KO_s106
                                 + 0.034321248756279 w_k1_KO_s107
                                 + 0.0663851623649263 w_k1_KO_s110
                                 + 0.226446990182434 w_k1_KO_s111
                                 + 0.0481905524446599 w_k1_KO_s112
                                 + 0.0893803760785061 w_k1_KO_s113
                                 + 0.0550891518758668 w_k1_KO_s114
                                 + 0.0652508851971612 w_k1_KO_s115
                                 + 0.104340379644466 w_k1_KO_s116
                                 + 0.0517190684515955 w_k1_KO_s117
                                 + 0.00709156984157369 w_k1_MAT_s000
                                 + 0.00476721082330914 w_k1_MAT_s001
                                 + 0.00376247344185728 w_k1_MAT_s002
                                 + 0.0067909855735176 w_k1_MAT_s003
                                 + 0.0103512345213217 w_k1_MAT_s004
                                 + 0.0189411145326637 w_k1_MAT_s005
                                 + 0.00740441974720519 w_k1_MAT_s006
                                 + 0.0166747575242684 w_k1_MAT_s007
                                 + 0.00886497041323556 w_k1_MAT_s010
                                 + 0.0160041565331727 w_k1_MAT_s011
                                 + 0.00818958659978512 w_k1_MAT_s012
                                 + 0.0113432916264724 w_k1_MAT_s013
                                 + 0.0125873777097733 w_k1_MAT_s014
                                 + 0.0202732916107514 w_k1_MAT_s015
                                 + 0.00911161674158356 w_k1_MAT_s016
                                 + 0.0162564839858357 w_k1_MAT_s017
                                 + 0.0283213571276569 w_k1_MAT_s100
                                 + 0.00955351536043717 w_k1_MAT_s101
                                 + 0.0151302066927603 w_k1_MAT_s102
                                 + 0.0273162681996922 w_k1_MAT_s103
                                 + 0.0129469121962309 w_k1_MAT_s104
                                 + 0.0101022838165779 w_k1_MAT_s105
                                 + 0.0215416675401027 w_k1_MAT_s106
                                 + 0.034395671820694 w_k1_MAT_s107
                                 + 0.067459656558905 w_k1_MAT_s110
                                 + 0.232617416168545 w_k1_MAT_s111
                                 + 0.0455352378584815 w_k1_MAT_s112
                                 + 0.0725634408571572 w_k1_MAT_s113
                                 + 0.058285224922139 w_k1_MAT_s114
                                 + 0.062468922793741 w_k1_MAT_s115
                                 + 0.113778369890919 w_k1_MAT_s116
                                 + 0.0520697939686571 w_k1_MAT_s117
                                 + 0.00773763916270239 w_k1_MSFT_s000
                                 + 0.00499143324539716 w_k1_MSFT_s001
                                 + 0.00345353679865271 w_k1_MSFT_s002
                                 + 0.00757861449403921 w_k1_MSFT_s003
                                 + 0.0106437044053142 w_k1_MSFT_s004
                                 + 0.0155512566023698 w_k1_MSFT_s005
                                 + 0.00657278115687829 w_k1_MSFT_s006
                                 + 0.0125992751950072 w_k1_MSFT_s007
                                 + 0.00761043715088978 w_k1_MSFT_s010
                                 + 0.0192746974761182 w_k1_MSFT_s011
                                 + 0.00826073730259002 w_k1_MSFT_s012
                                 + 0.00985512563592085 w_k1_MSFT_s013
                                 + 0.0128756610357987 w_k1_MSFT_s014
                                 + 0.0192008027508132 w_k1_MSFT_s015
                                 + 0.011244624893574 w_k1_MSFT_s016
                                 + 0.0182496331374303 w_k1_MSFT_s017
                                 + 0.0250104839744553 w_k1_MSFT_s100
                                 + 0.00985348914532317 w_k1_MSFT_s101
                                 + 0.0130844202342144 w_k1_MSFT_s102
                                 + 0.0296547435470097 w_k1_MSFT_s103
                                 + 0.0133099326223036 w_k1_MSFT_s104
                                 + 0.00899132982319417 w_k1_MSFT_s105
                                 + 0.0246807354549481 w_k1_MSFT_s106
                                 + 0.0397531131674567 w_k1_MSFT_s107
                                 + 0.0729098213213994 w_k1_MSFT_s110
                                 + 0.219595157558851 w_k1_MSFT_s111
                                 + 0.0473763537964427 w_k1_MSFT_s112
                                 + 0.0790497461921938 w_k1_MSFT_s113
                                 + 0.0576242558287685 w_k1_MSFT_s114
                                 + 0.0628359448822663 w_k1_MSFT_s115
                                 + 0.0980067976663485 w_k1_MSFT_s116
                                 + 0.0531875405216345 w_k1_MSFT_s117
                                 + 0.00814496075322227 w_k1_PG_s000
                                 + 0.00505801659489019 w_k1_PG_s001
                                 + 0.00375199338327937 w_k1_PG_s002
                                 + 0.0076421584900899 w_k1_PG_s003
                                 + 0.0118980525725073 w_k1_PG_s004
                                 + 0.0149583499043199 w_k1_PG_s005
                                 + 0.006717930826781 w_k1_PG_s006
                                 + 0.0154269994330151 w_k1_PG_s007
                                 + 0.00834897868634261 w_k1_PG_s010
                                 + 0.018505369249828 w_k1_PG_s011
                                 + 0.00877045694487814 w_k1_PG_s012
                                 + 0.0115568330521068 w_k1_PG_s013
                                 + 0.0122255263160111 w_k1_PG_s014
                                 + 0.0187731629824292 w_k1_PG_s015
                                 + 0.0102389709928151 w_k1_PG_s016
                                 + 0.0190183394045883 w_k1_PG_s017
                                 + 0.0240237838901378 w_k1_PG_s100
                                 + 0.00986222721416317 w_k1_PG_s101
                                 + 0.0136261985550854 w_k1_PG_s102
                                 + 0.0298495480927553 w_k1_PG_s103
                                 + 0.0130374305992387 w_k1_PG_s104
                                 + 0.00835539126253271 w_k1_PG_s105
                                 + 0.0222792899114161 w_k1_PG_s106
                                 + 0.0317655241680805 w_k1_PG_s107
                                 + 0.0680891827970928 w_k1_PG_s110
                                 + 0.225474413710228 w_k1_PG_s111
                                 + 0.0468084377520991 w_k1_PG_s112
                                 + 0.0773883003903872 w_k1_PG_s113
                                 + 0.0557469106813599 w_k1_PG_s114
                                 + 0.0649319388880399 w_k1_PG_s115
                                 + 0.0988274395895379 w_k1_PG_s116
                                 + 0.0497986891795103 w_k1_PG_s117 >= 1.13
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
