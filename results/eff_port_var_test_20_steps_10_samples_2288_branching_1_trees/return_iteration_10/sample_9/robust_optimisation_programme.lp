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
 w_k1_ADM_s0_return_constraint:  1.09681 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.981562 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.93512 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.26879 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.891356 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.974165 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.33675 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.07585 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.91163 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.925738 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.20687 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.751675 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.896523 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.01998 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.923844 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.08594 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.777765 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.30822 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.05948 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.957002 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.11818 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.720765 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.07509 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.780794 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.769663 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.41904 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.04787 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.45282 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.02725 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.09024 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.867514 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.9253 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.30752 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.09914 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.11572 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.719538 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.889679 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.97933 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.26408 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.890982 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.828908 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.28559 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.10374 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.906831 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.45381 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.832493 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.30153 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.873556 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.01517 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.722689 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.904207 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.31436 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.820165 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.00831 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.988822 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.51791 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.905968 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.24132 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.06201 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.69853 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.813031 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.561015 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.681404 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.365 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.21398 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.65124 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.841988 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.01758 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.00551 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.721311 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.63999 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.17716 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.36406 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.562966 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.858096 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.09622 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.33447 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.877204 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.789316 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.27124 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.06878 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.931456 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.54326 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.917106 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.23647 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.980972 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.53867 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.677137 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.746252 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.11001 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.779374 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.05277 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.89484 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.94266 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.889743 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.38064 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.36043 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.676352 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.13812 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.504112 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.650264 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.49111 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.28698 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 2.05402 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.738988 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.963077 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.772905 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.820962 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 2.04737 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.34739 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.32521 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.463433 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.858242 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.998707 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.18732 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.922519 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.831677 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.20169 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.01385 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.05304 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.3999 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.953481 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.0767 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.10604 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.32173 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.743887 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.795561 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.12548 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.80075 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.0622 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.843194 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.50231 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.06153 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.10335 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.13545 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.774478 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.09362 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.6696 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.744765 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.29909 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.01253 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.6557 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.961266 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.981924 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.865433 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.869375 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.60846 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.21422 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.26943 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.593945 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.821145 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.03368 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.23631 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.909051 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.913407 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.14663 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.13447 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.626466 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.07455 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.840417 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.49065 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.531716 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.786601 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.814788 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.36354 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.04214 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.04076 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.771615 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.41843 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.15538 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.450093 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.78204 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.0542 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.675606 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.783083 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.579776 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.913061 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.05065 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.87177 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.37644 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.629881 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.13318 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.892756 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.654788 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.35249 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.02512 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.36322 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.806552 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.19945 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.01906 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.13109 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.919085 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.939115 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.15905 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.0603 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.04425 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.28913 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.934414 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.08016 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.01585 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.10808 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.807712 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.78707 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.98604 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.781889 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.875955 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.816729 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.58447 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.07322 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.27555 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.21497 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.924839 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.973426 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.6484 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.808366 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.29166 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.05693 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.41159 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.818116 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.959593 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.923303 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.850793 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.37782 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.1488 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.23617 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.684723 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.909656 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.946781 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.23439 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.915686 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.876989 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.09268 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.06725 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.788978 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.16217 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.955661 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.2562 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.713713 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.13737 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.812733 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.01532 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.956351 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.969928 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.996327 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.22207 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.34681 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.727769 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.52784 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.13754 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.751608 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.944672 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.581086 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.839583 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.24849 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.48733 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.54794 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.664678 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.01308 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.855175 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.76491 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.54709 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.21233 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.23759 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.739733 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.938576 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.928498 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.26509 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.953342 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.89297 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.20656 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.02174 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.633326 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.908274 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.757404 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.45927 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.407254 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.78759 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.758551 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.14303 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.22098 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.900252 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.857876 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.68121 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.21644 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.53368 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.74875 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.058 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.676653 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.727914 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.516223 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.830173 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.30585 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.63827 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.52263 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.729103 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.3963 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.00737 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.69274 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.38826 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.898955 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.4064 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.661088 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.943157 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.0105 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.12242 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.940018 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.976793 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.09382 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.938889 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.998033 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.20656 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.932654 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.1226 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.951297 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.16899 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.722296 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.70446 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.901399 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.883012 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.0653 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.951738 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.54439 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.14763 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.29029 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.26978 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.90506 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.10238 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.727276 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.880467 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.36174 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.05666 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.52786 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.889325 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.04103 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.841725 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.940374 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.55211 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.22255 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.1738 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.689733 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.869256 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.13243 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.939574 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.888176 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.16828 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.996604 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.941313 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.26293 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.986277 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.20979 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.958946 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.09251 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.873966 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.00829 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.05205 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.882212 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.969304 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.866854 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.16221 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.91594 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.12015 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.954006 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.806463 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.01911 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.749528 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.878411 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.12275 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.16015 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.49967 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.94849 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.05598 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.876434 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.863227 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.30386 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.07386 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.08032 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.717852 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.00111 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.02002 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.41500087088878 z_k1_s0000
                                 - 0.180000025433444 z_k1_s0001
                                 - 0.0970000051557947 z_k1_s0002
                                 - 0.178999674741905 z_k1_s0003
                                 - 0.155000492151347 z_k1_s0004
                                 - 0.168000434138165 z_k1_s0005
                                 - 0.243000572985522 z_k1_s0006
                                 - 0.102000065712321 z_k1_s0007
                                 - 0.215000483405213 z_k1_s0010
                                 - 0.292000625722353 z_k1_s0011
                                 - 0.326000965759177 z_k1_s0012
                                 - 0.434001872699251 z_k1_s0013
                                 - 0.673000063418316 z_k1_s0014
                                 - 0.353000824493286 z_k1_s0015
                                 - 0.47900102392125 z_k1_s0016
                                 - 0.90800176425947 z_k1_s0017
                                 - 0.29600007434596 z_k1_s0020
                                 - 0.895999989652664 z_k1_s0021
                                 - 0.342999973357663 z_k1_s0022
                                 - 0.282000119870131 z_k1_s0023
                                 - 0.421998213028595 z_k1_s0024
                                 - 0.218999961828948 z_k1_s0025
                                 - 0.84700123768713 z_k1_s0026
                                 - 0.323999983297617 z_k1_s0027
                                 - 0.198000132728441 z_k1_s0030
                                 - 0.209000557702462 z_k1_s0031
                                 - 0.290000384218523 z_k1_s0032
                                 - 0.391001173232904 z_k1_s0033
                                 - 0.0859999627405861 z_k1_s0034
                                 - 0.218999994951074 z_k1_s0035
                                 - 0.361000773485971 z_k1_s0036
                                 - 0.334000831313951 z_k1_s0037
                                 - 0.107000005395453 z_k1_s0040
                                 - 0.393999428258056 z_k1_s0041
                                 - 0.121999977867714 z_k1_s0042
                                 - 0.137000076439969 z_k1_s0043
                                 - 0.102999987516185 z_k1_s0044
                                 - 0.151999670612251 z_k1_s0045
                                 - 0.156999535336345 z_k1_s0046
                                 - 0.0889999879887422 z_k1_s0047
                                 - 0.151000086220696 z_k1_s0050
                                 - 0.283999337394527 z_k1_s0051
                                 - 0.421000636913664 z_k1_s0052
                                 - 0.290001212411484 z_k1_s0053
                                 - 0.434999423615224 z_k1_s0054
                                 - 0.175000161782316 z_k1_s0055
                                 - 0.188000232649051 z_k1_s0056
                                 - 0.451000431990437 z_k1_s0057
                                 - 0.197000221349816 z_k1_s0060
                                 - 0.0960000904039692 z_k1_s0061
                                 - 0.138000086955701 z_k1_s0062
                                 - 0.114000064354709 z_k1_s0063
                                 - 0.457000609527248 z_k1_s0064
                                 - 0.134000054522199 z_k1_s0065
                                 - 0.132000038305448 z_k1_s0066
                                 - 0.45200056898537 z_k1_s0067
                                 - 0.151999460622093 z_k1_s0070
                                 - 0.147999903289969 z_k1_s0071
                                 - 0.112000049400527 z_k1_s0072
                                 - 0.112000049400527 z_k1_s0073
                                 - 0.121999873130916 z_k1_s0074
                                 - 0.252000023926179 z_k1_s0075
                                 - 0.0570000303333044 z_k1_s0076
                                 - 0.20800024127242 z_k1_s0077
                                 - 0.527000123338887 z_k1_s0100
                                 - 0.29500036626352 z_k1_s0101
                                 - 0.195999880053275 z_k1_s0102
                                 - 0.252999960138085 z_k1_s0103
                                 - 0.537000089178357 z_k1_s0104
                                 - 0.376000364363058 z_k1_s0105
                                 - 0.408000804649024 z_k1_s0106
                                 - 0.156000016695394 z_k1_s0107
                                 - 0.302000038512898 z_k1_s0110
                                 - 0.246999945250627 z_k1_s0111
                                 - 0.777000505687101 z_k1_s0112
                                 - 0.712999314691157 z_k1_s0113
                                 - 0.317999964161937 z_k1_s0114
                                 - 0.735998812455455 z_k1_s0115
                                 - 0.224999907945719 z_k1_s0116
                                 - 0.402999127303766 z_k1_s0117
                                 - 0.264000560078368 z_k1_s0120
                                 - 0.338999993985594 z_k1_s0121
                                 - 0.151000249249232 z_k1_s0122
                                 - 0.0820000943332785 z_k1_s0123
                                 - 0.359001344630422 z_k1_s0124
                                 - 0.28799992844781 z_k1_s0125
                                 - 0.122000167817858 z_k1_s0126
                                 - 0.272000349534848 z_k1_s0127
                                 - 0.0599999991417497 z_k1_s0130
                                 - 0.210000372896265 z_k1_s0131
                                 - 0.183000002752339 z_k1_s0132
                                 - 0.124999684961861 z_k1_s0133
                                 - 0.0640000062445357 z_k1_s0134
                                 - 0.155999632608412 z_k1_s0135
                                 - 0.185999685879307 z_k1_s0136
                                 - 0.0900000524126447 z_k1_s0137
                                 - 0.0870003788802174 z_k1_s0140
                                 - 0.0389999873944568 z_k1_s0141
                                 - 0.127999958627961 z_k1_s0142
                                 - 0.145000360133512 z_k1_s0143
                                 - 0.0810000552193114 z_k1_s0144
                                 - 0.159999948284951 z_k1_s0145
                                 - 0.0580000626533497 z_k1_s0146
                                 - 0.115999962506589 z_k1_s0147
                                 - 0.155999476008359 z_k1_s0150
                                 - 0.289997960974828 z_k1_s0151
                                 - 0.303998032276398 z_k1_s0152
                                 - 0.222998205743597 z_k1_s0153
                                 - 0.282997925324043 z_k1_s0154
                                 - 0.290998845064363 z_k1_s0155
                                 - 0.239000045224237 z_k1_s0156
                                 - 0.264998419648307 z_k1_s0157
                                 - 0.177000181159545 z_k1_s0160
                                 - 0.164000221081212 z_k1_s0161
                                 - 0.125000079145959 z_k1_s0162
                                 - 0.413999332572116 z_k1_s0163
                                 - 0.606000957346564 z_k1_s0164
                                 - 0.315000743784343 z_k1_s0165
                                 - 0.252000333327221 z_k1_s0166
                                 - 0.564000683708482 z_k1_s0167
                                 - 0.445997796901317 z_k1_s0170
                                 - 0.274997929661085 z_k1_s0171
                                 - 0.19899948806079 z_k1_s0172
                                 - 0.305998369802616 z_k1_s0173
                                 - 0.518998065172459 z_k1_s0174
                                 - 0.140999612005965 z_k1_s0175
                                 - 0.504998122462588 z_k1_s0176
                                 - 0.254999258900271 z_k1_s0177
                                 - 0.664002250721469 z_k1_s1000
                                 - 0.736004599842778 z_k1_s1001
                                 - 0.393001319849263 z_k1_s1002
                                 - 0.350001421440103 z_k1_s1003
                                 - 1.68600696043241 z_k1_s1004
                                 - 0.621999424395911 z_k1_s1005
                                 - 0.49000152409461 z_k1_s1006
                                 - 0.883002843943873 z_k1_s1007
                                 - 0.416999312360294 z_k1_s1010
                                 - 0.854999013738253 z_k1_s1011
                                 - 0.624999738205806 z_k1_s1012
                                 - 0.404999463949777 z_k1_s1013
                                 - 0.538998643599567 z_k1_s1014
                                 - 0.745997119179758 z_k1_s1015
                                 - 0.397999334277222 z_k1_s1016
                                 - 0.376999381459064 z_k1_s1017
                                 - 0.847999632853577 z_k1_s1020
                                 - 2.08199809671701 z_k1_s1021
                                 - 3.04399171133062 z_k1_s1022
                                 - 1.03099907244323 z_k1_s1023
                                 - 0.63599942549049 z_k1_s1024
                                 - 1.90800066966901 z_k1_s1025
                                 - 0.719999462498552 z_k1_s1026
                                 - 1.6969924426857 z_k1_s1027
                                 - 1.55200204153112 z_k1_s1030
                                 - 1.5319989318768 z_k1_s1031
                                 - 0.897002683871031 z_k1_s1032
                                 - 0.863998807841641 z_k1_s1033
                                 - 0.493999841247515 z_k1_s1034
                                 - 0.681000053810086 z_k1_s1035
                                 - 1.28100215402287 z_k1_s1036
                                 - 1.06500036056208 z_k1_s1037
                                 - 0.712997467867203 z_k1_s1040
                                 - 0.760998722190047 z_k1_s1041
                                 - 0.389999914502538 z_k1_s1042
                                 - 0.85299915755968 z_k1_s1043
                                 - 0.285999874301889 z_k1_s1044
                                 - 0.718998215282304 z_k1_s1045
                                 - 0.690997877343808 z_k1_s1046
                                 - 0.311999648102153 z_k1_s1047
                                 - 0.457999215359063 z_k1_s1050
                                 - 0.212999833575174 z_k1_s1051
                                 - 0.209999937243073 z_k1_s1052
                                 - 0.156999806710379 z_k1_s1053
                                 - 0.674999564710008 z_k1_s1054
                                 - 0.297999839316401 z_k1_s1055
                                 - 0.651001048052836 z_k1_s1056
                                 - 0.608000571960479 z_k1_s1057
                                 - 0.443998527481024 z_k1_s1060
                                 - 0.217999604422216 z_k1_s1061
                                 - 0.341000764643865 z_k1_s1062
                                 - 0.79200013052994 z_k1_s1063
                                 - 0.180999727132131 z_k1_s1064
                                 - 0.674997502887047 z_k1_s1065
                                 - 0.325998057205717 z_k1_s1066
                                 - 0.210999722415854 z_k1_s1067
                                 - 1.37999564830399 z_k1_s1070
                                 - 0.799999816063725 z_k1_s1071
                                 - 1.17600042810784 z_k1_s1072
                                 - 0.564999885148759 z_k1_s1073
                                 - 1.0360005205117 z_k1_s1074
                                 - 2.34700060145147 z_k1_s1075
                                 - 2.04000037397269 z_k1_s1076
                                 - 2.69900553040867 z_k1_s1077
                                 - 0.131000582342136 z_k1_s1100
                                 - 0.0920003647713921 z_k1_s1101
                                 - 0.145000117170392 z_k1_s1102
                                 - 0.174000930207766 z_k1_s1103
                                 - 0.217001409673946 z_k1_s1104
                                 - 0.113000522417346 z_k1_s1105
                                 - 0.0880003660770554 z_k1_s1106
                                 - 0.356001857804206 z_k1_s1107
                                 - 0.0849998495593703 z_k1_s1110
                                 - 0.16700049066592 z_k1_s1111
                                 - 0.0859998573777428 z_k1_s1112
                                 - 0.209000819037567 z_k1_s1113
                                 - 0.0660001900116699 z_k1_s1114
                                 - 0.0650001821932973 z_k1_s1115
                                 - 0.0899998886512331 z_k1_s1116
                                 - 0.047000122962821 z_k1_s1117
                                 - 0.0580000776724919 z_k1_s1120
                                 - 0.243000489455541 z_k1_s1121
                                 - 0.234000523920183 z_k1_s1122
                                 - 0.146999910411139 z_k1_s1123
                                 - 0.158000657177062 z_k1_s1124
                                 - 0.240000027610131 z_k1_s1125
                                 - 0.165999522096701 z_k1_s1126
                                 - 0.173999807017214 z_k1_s1127
                                 - 0.121000762755091 z_k1_s1130
                                 - 0.273002095694334 z_k1_s1131
                                 - 0.249001636312946 z_k1_s1132
                                 - 0.103000557319892 z_k1_s1133
                                 - 0.174001244002427 z_k1_s1134
                                 - 0.155001459912338 z_k1_s1135
                                 - 0.144001365279904 z_k1_s1136
                                 - 0.172000973885907 z_k1_s1137
                                 - 0.13899998135298 z_k1_s1140
                                 - 0.0919999356494714 z_k1_s1141
                                 - 0.19200049432913 z_k1_s1142
                                 - 0.137999553674319 z_k1_s1143
                                 - 0.16899998571693 z_k1_s1144
                                 - 0.0720000104734798 z_k1_s1145
                                 - 0.1200000174558 z_k1_s1146
                                 - 0.243999413626992 z_k1_s1147
                                 - 0.194000758379814 z_k1_s1150
                                 - 0.0680000935679493 z_k1_s1151
                                 - 0.237000488535572 z_k1_s1152
                                 - 0.173999731329065 z_k1_s1153
                                 - 0.283000726749422 z_k1_s1154
                                 - 0.0620000686531077 z_k1_s1155
                                 - 0.140000250945857 z_k1_s1156
                                 - 0.257999796936465 z_k1_s1157
                                 - 0.119000084070846 z_k1_s1160
                                 - 0.253999996050149 z_k1_s1161
                                 - 0.117000193582398 z_k1_s1162
                                 - 0.24100061987513 z_k1_s1163
                                 - 0.210999622545131 z_k1_s1164
                                 - 0.259000162271816 z_k1_s1165
                                 - 0.134000230735408 z_k1_s1166
                                 - 0.425000048824137 z_k1_s1167
                                 - 0.265000924791139 z_k1_s1170
                                 - 0.425002880558375 z_k1_s1171
                                 - 0.305002388237514 z_k1_s1172
                                 - 0.266000766476385 z_k1_s1173
                                 - 0.141000675927468 z_k1_s1174
                                 - 0.256000400614791 z_k1_s1175
                                 - 0.176000787037566 z_k1_s1176
                                 - 0.115000504290977 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.02891 w_k1_ADM_s000
                                 + 1.09141 w_k1_AXP_s000
                                 + 0.899489 w_k1_DIS_s000
                                 + 1.0593 w_k1_FDX_s000
                                 + 0.780162 w_k1_IBM_s000
                                 + 0.913558 w_k1_INTC_s000
                                 + 0.907426 w_k1_KO_s000
                                 + 1.03757 w_k1_MAT_s000
                                 + 0.922471 w_k1_MSFT_s000
                                 + 0.895381 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.873188 w_k1_ADM_s000
                                 + 1.13676 w_k1_AXP_s000 + 1.3607 w_k1_DIS_s000
                                 + 1.08288 w_k1_FDX_s000
                                 + 1.62827 w_k1_IBM_s000
                                 + 1.1022 w_k1_INTC_s000 + 1.54257 w_k1_KO_s000
                                 + 2.87598 w_k1_MAT_s000
                                 + 1.24019 w_k1_MSFT_s000
                                 + 1.24568 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.51652 w_k1_ADM_s000
                                 + 1.32947 w_k1_AXP_s000
                                 + 1.49916 w_k1_DIS_s000
                                 + 1.24441 w_k1_FDX_s000
                                 + 1.33185 w_k1_IBM_s000
                                 + 1.10165 w_k1_INTC_s000
                                 + 1.55572 w_k1_KO_s000 + 1.71271 w_k1_MAT_s000
                                 + 1.24209 w_k1_MSFT_s000
                                 + 1.21922 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.824423 w_k1_ADM_s000
                                 + 0.779291 w_k1_AXP_s000
                                 + 0.670938 w_k1_DIS_s000
                                 + 0.740176 w_k1_FDX_s000
                                 + 0.824732 w_k1_IBM_s000
                                 + 0.922565 w_k1_INTC_s000
                                 + 0.977231 w_k1_KO_s000
                                 + 0.838652 w_k1_MAT_s000
                                 + 0.798938 w_k1_MSFT_s000
                                 + 0.797263 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.976202 w_k1_ADM_s000
                                 + 0.753026 w_k1_AXP_s000
                                 + 0.613329 w_k1_DIS_s000
                                 + 0.889199 w_k1_FDX_s000
                                 + 0.575074 w_k1_IBM_s000
                                 + 0.795116 w_k1_INTC_s000
                                 + 0.778902 w_k1_KO_s000
                                 + 0.948081 w_k1_MAT_s000
                                 + 0.930302 w_k1_MSFT_s000
                                 + 0.799714 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.00389 w_k1_ADM_s000
                                 + 1.3103 w_k1_AXP_s000 + 1.44349 w_k1_DIS_s000
                                 + 1.43592 w_k1_FDX_s000
                                 + 1.03225 w_k1_IBM_s000
                                 + 1.07614 w_k1_INTC_s000 + 1.4136 w_k1_KO_s000
                                 + 1.81058 w_k1_MAT_s000
                                 + 1.36277 w_k1_MSFT_s000
                                 + 1.23445 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.91568 w_k1_ADM_s000
                                 + 1.10974 w_k1_AXP_s000
                                 + 0.986578 w_k1_DIS_s000
                                 + 0.999035 w_k1_FDX_s000
                                 + 1.03559 w_k1_IBM_s000
                                 + 0.9354 w_k1_INTC_s000 + 1.09111 w_k1_KO_s000
                                 + 1.72313 w_k1_MAT_s000
                                 + 1.06167 w_k1_MSFT_s000
                                 + 0.964153 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.88681 w_k1_ADM_s000
                                 + 0.979985 w_k1_AXP_s000
                                 + 0.92847 w_k1_DIS_s000
                                 + 0.873615 w_k1_FDX_s000
                                 + 1.24224 w_k1_IBM_s000
                                 + 1.00407 w_k1_INTC_s000
                                 + 1.19132 w_k1_KO_s000 + 1.51129 w_k1_MAT_s000
                                 + 0.861484 w_k1_MSFT_s000
                                 + 0.967812 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.07873 w_k1_ADM_s001
                                 + 1.18349 w_k1_AXP_s001
                                 + 1.22695 w_k1_DIS_s001
                                 + 1.28304 w_k1_FDX_s001
                                 + 0.902006 w_k1_IBM_s001
                                 + 1.13041 w_k1_INTC_s001
                                 + 0.939985 w_k1_KO_s001
                                 + 0.985874 w_k1_MAT_s001
                                 + 1.1117 w_k1_MSFT_s001 + 1.07007 w_k1_PG_s001
                                 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.615892 w_k1_ADM_s001
                                 + 0.857773 w_k1_AXP_s001
                                 + 0.770847 w_k1_DIS_s001
                                 + 0.811535 w_k1_FDX_s001
                                 + 1.23175 w_k1_IBM_s001
                                 + 0.946502 w_k1_INTC_s001
                                 + 0.953221 w_k1_KO_s001
                                 + 1.10154 w_k1_MAT_s001
                                 + 0.839886 w_k1_MSFT_s001
                                 + 0.915937 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.607669 w_k1_ADM_s001
                                 + 0.677709 w_k1_AXP_s001
                                 + 0.684014 w_k1_DIS_s001
                                 + 0.761401 w_k1_FDX_s001
                                 + 0.889658 w_k1_IBM_s001
                                 + 0.840249 w_k1_INTC_s001
                                 + 0.785607 w_k1_KO_s001
                                 + 0.866699 w_k1_MAT_s001
                                 + 0.781795 w_k1_MSFT_s001
                                 + 0.869925 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.915355 w_k1_ADM_s001
                                 + 1.11919 w_k1_AXP_s001
                                 + 1.39449 w_k1_DIS_s001
                                 + 1.21544 w_k1_FDX_s001
                                 + 1.10089 w_k1_IBM_s001
                                 + 1.14749 w_k1_INTC_s001 + 1.079 w_k1_KO_s001
                                 + 0.810378 w_k1_MAT_s001
                                 + 1.14314 w_k1_MSFT_s001
                                 + 1.02609 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.831163 w_k1_ADM_s001
                                 + 1.24638 w_k1_AXP_s001
                                 + 1.15865 w_k1_DIS_s001
                                 + 1.03866 w_k1_FDX_s001
                                 + 1.36436 w_k1_IBM_s001
                                 + 0.984154 w_k1_INTC_s001
                                 + 1.10471 w_k1_KO_s001 + 1.24415 w_k1_MAT_s001
                                 + 1.00718 w_k1_MSFT_s001
                                 + 1.08329 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.23041 w_k1_ADM_s001
                                 + 1.68389 w_k1_AXP_s001
                                 + 1.68928 w_k1_DIS_s001
                                 + 1.39631 w_k1_FDX_s001
                                 + 1.44646 w_k1_IBM_s001
                                 + 1.16332 w_k1_INTC_s001
                                 + 1.28351 w_k1_KO_s001 + 1.37726 w_k1_MAT_s001
                                 + 1.10998 w_k1_MSFT_s001
                                 + 1.28628 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.830046 w_k1_ADM_s001
                                 + 0.707543 w_k1_AXP_s001
                                 + 0.713189 w_k1_DIS_s001
                                 + 0.840746 w_k1_FDX_s001
                                 + 0.64713 w_k1_IBM_s001
                                 + 0.826342 w_k1_INTC_s001
                                 + 0.747885 w_k1_KO_s001
                                 + 0.911509 w_k1_MAT_s001
                                 + 0.789412 w_k1_MSFT_s001
                                 + 0.807299 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.1291 w_k1_ADM_s001
                                 + 1.16389 w_k1_AXP_s001
                                 + 0.921197 w_k1_DIS_s001
                                 + 1.06326 w_k1_FDX_s001
                                 + 0.861898 w_k1_IBM_s001
                                 + 0.971421 w_k1_INTC_s001
                                 + 0.870321 w_k1_KO_s001
                                 + 0.872745 w_k1_MAT_s001
                                 + 0.984721 w_k1_MSFT_s001
                                 + 0.987348 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.34153 w_k1_ADM_s002
                                 + 1.08877 w_k1_AXP_s002
                                 + 1.81329 w_k1_DIS_s002 + 1.3223 w_k1_FDX_s002
                                 + 1.38348 w_k1_IBM_s002
                                 + 1.16438 w_k1_INTC_s002
                                 + 1.48464 w_k1_KO_s002 + 1.24172 w_k1_MAT_s002
                                 + 1.13979 w_k1_MSFT_s002 + 1.1889 w_k1_PG_s002
                                 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.32144 w_k1_ADM_s002
                                 + 0.948346 w_k1_AXP_s002
                                 + 1.08477 w_k1_DIS_s002 + 1.0785 w_k1_FDX_s002
                                 + 0.752578 w_k1_IBM_s002
                                 + 0.971169 w_k1_INTC_s002
                                 + 0.945825 w_k1_KO_s002
                                 + 0.676995 w_k1_MAT_s002
                                 + 0.955451 w_k1_MSFT_s002
                                 + 0.898114 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.978704 w_k1_ADM_s002
                                 + 0.880102 w_k1_AXP_s002
                                 + 0.898299 w_k1_DIS_s002
                                 + 0.917002 w_k1_FDX_s002
                                 + 1.16657 w_k1_IBM_s002
                                 + 1.15015 w_k1_INTC_s002
                                 + 1.10226 w_k1_KO_s002 + 1.18532 w_k1_MAT_s002
                                 + 0.888224 w_k1_MSFT_s002
                                 + 0.949731 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.42128 w_k1_ADM_s002
                                 + 0.939246 w_k1_AXP_s002
                                 + 1.0735 w_k1_DIS_s002 + 0.95676 w_k1_FDX_s002
                                 + 1.79609 w_k1_IBM_s002
                                 + 0.818509 w_k1_INTC_s002
                                 + 1.23667 w_k1_KO_s002 + 1.662 w_k1_MAT_s002
                                 + 0.757036 w_k1_MSFT_s002
                                 + 1.0832 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.07738 w_k1_ADM_s002
                                 + 1.01982 w_k1_AXP_s002
                                 + 0.884155 w_k1_DIS_s002
                                 + 1.11257 w_k1_FDX_s002
                                 + 0.786562 w_k1_IBM_s002
                                 + 1.03551 w_k1_INTC_s002
                                 + 0.757576 w_k1_KO_s002
                                 + 0.81052 w_k1_MAT_s002
                                 + 0.789529 w_k1_MSFT_s002
                                 + 0.814264 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.22436 w_k1_ADM_s002
                                 + 1.58748 w_k1_AXP_s002
                                 + 1.36656 w_k1_DIS_s002
                                 + 1.39456 w_k1_FDX_s002
                                 + 1.05899 w_k1_IBM_s002
                                 + 1.3066 w_k1_INTC_s002 + 1.13931 w_k1_KO_s002
                                 + 0.913727 w_k1_MAT_s002
                                 + 1.13111 w_k1_MSFT_s002
                                 + 1.11775 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.986161 w_k1_ADM_s002
                                 + 0.797611 w_k1_AXP_s002
                                 + 0.810147 w_k1_DIS_s002
                                 + 0.866196 w_k1_FDX_s002
                                 + 0.876933 w_k1_IBM_s002
                                 + 0.97197 w_k1_INTC_s002
                                 + 0.900625 w_k1_KO_s002
                                 + 0.874026 w_k1_MAT_s002
                                 + 0.758015 w_k1_MSFT_s002
                                 + 0.890353 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.05103 w_k1_ADM_s002
                                 + 0.915745 w_k1_AXP_s002
                                 + 1.14938 w_k1_DIS_s002
                                 + 1.00081 w_k1_FDX_s002
                                 + 1.30421 w_k1_IBM_s002
                                 + 1.01981 w_k1_INTC_s002
                                 + 1.04057 w_k1_KO_s002 + 1.50652 w_k1_MAT_s002
                                 + 0.983479 w_k1_MSFT_s002
                                 + 1.06628 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.27552 w_k1_ADM_s003
                                 + 1.26392 w_k1_AXP_s003
                                 + 1.58979 w_k1_DIS_s003
                                 + 1.17648 w_k1_FDX_s003
                                 + 1.55687 w_k1_IBM_s003
                                 + 1.02026 w_k1_INTC_s003
                                 + 1.65311 w_k1_KO_s003 + 1.81043 w_k1_MAT_s003
                                 + 1.29182 w_k1_MSFT_s003 + 1.1659 w_k1_PG_s003
                                 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.32643 w_k1_ADM_s003
                                 + 1.19236 w_k1_AXP_s003
                                 + 1.34597 w_k1_DIS_s003
                                 + 1.23963 w_k1_FDX_s003
                                 + 1.04728 w_k1_IBM_s003
                                 + 0.880489 w_k1_INTC_s003
                                 + 1.20322 w_k1_KO_s003 + 1.51749 w_k1_MAT_s003
                                 + 1.07684 w_k1_MSFT_s003 + 1.1675 w_k1_PG_s003
                                 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.12387 w_k1_ADM_s003
                                 + 0.697835 w_k1_AXP_s003
                                 + 0.875528 w_k1_DIS_s003
                                 + 0.909288 w_k1_FDX_s003
                                 + 0.834306 w_k1_IBM_s003
                                 + 0.80342 w_k1_INTC_s003
                                 + 0.95472 w_k1_KO_s003
                                 + 0.997043 w_k1_MAT_s003
                                 + 0.912365 w_k1_MSFT_s003
                                 + 1.00398 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.947159 w_k1_ADM_s003
                                 + 0.722025 w_k1_AXP_s003
                                 + 0.620695 w_k1_DIS_s003
                                 + 0.751121 w_k1_FDX_s003
                                 + 0.818801 w_k1_IBM_s003
                                 + 0.762076 w_k1_INTC_s003
                                 + 0.807041 w_k1_KO_s003
                                 + 0.996368 w_k1_MAT_s003
                                 + 0.878451 w_k1_MSFT_s003
                                 + 0.845026 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.45268 w_k1_ADM_s003
                                 + 2.06968 w_k1_AXP_s003 + 1.3758 w_k1_DIS_s003
                                 + 1.47912 w_k1_FDX_s003
                                 + 1.11051 w_k1_IBM_s003
                                 + 1.21569 w_k1_INTC_s003 + 1.3475 w_k1_KO_s003
                                 + 2.32898 w_k1_MAT_s003
                                 + 1.37981 w_k1_MSFT_s003
                                 + 1.41229 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.96339 w_k1_ADM_s003
                                 + 1.20369 w_k1_AXP_s003
                                 + 0.934343 w_k1_DIS_s003
                                 + 1.17689 w_k1_FDX_s003
                                 + 0.803032 w_k1_IBM_s003
                                 + 0.971228 w_k1_INTC_s003
                                 + 0.861804 w_k1_KO_s003
                                 + 1.09313 w_k1_MAT_s003
                                 + 1.03436 w_k1_MSFT_s003
                                 + 0.935236 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.999582 w_k1_ADM_s003
                                 + 1.24502 w_k1_AXP_s003
                                 + 0.732504 w_k1_DIS_s003
                                 + 0.937385 w_k1_FDX_s003
                                 + 1.25147 w_k1_IBM_s003
                                 + 0.939639 w_k1_INTC_s003
                                 + 1.01776 w_k1_KO_s003 + 1.83245 w_k1_MAT_s003
                                 + 0.885849 w_k1_MSFT_s003
                                 + 0.976864 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.885934 w_k1_ADM_s003
                                 + 1.0163 w_k1_AXP_s003 + 1.08295 w_k1_DIS_s003
                                 + 0.980365 w_k1_FDX_s003
                                 + 1.13995 w_k1_IBM_s003
                                 + 1.06412 w_k1_INTC_s003
                                 + 1.16127 w_k1_KO_s003 + 1.63177 w_k1_MAT_s003
                                 + 1.07864 w_k1_MSFT_s003
                                 + 1.00895 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.741083 w_k1_ADM_s004
                                 + 1.17548 w_k1_AXP_s004
                                 + 1.06991 w_k1_DIS_s004
                                 + 1.17278 w_k1_FDX_s004
                                 + 1.17342 w_k1_IBM_s004
                                 + 1.12202 w_k1_INTC_s004
                                 + 0.912756 w_k1_KO_s004
                                 + 1.20057 w_k1_MAT_s004
                                 + 1.10692 w_k1_MSFT_s004
                                 + 1.01518 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.738847 w_k1_ADM_s004
                                 + 0.804445 w_k1_AXP_s004
                                 + 0.769013 w_k1_DIS_s004
                                 + 0.874672 w_k1_FDX_s004
                                 + 0.850541 w_k1_IBM_s004
                                 + 0.90178 w_k1_INTC_s004
                                 + 0.740038 w_k1_KO_s004
                                 + 0.823399 w_k1_MAT_s004
                                 + 0.964573 w_k1_MSFT_s004
                                 + 0.986854 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.30633 w_k1_ADM_s004
                                 + 1.45771 w_k1_AXP_s004
                                 + 1.76162 w_k1_DIS_s004
                                 + 1.51573 w_k1_FDX_s004
                                 + 1.41598 w_k1_IBM_s004
                                 + 1.38831 w_k1_INTC_s004
                                 + 1.43792 w_k1_KO_s004 + 1.55567 w_k1_MAT_s004
                                 + 1.45524 w_k1_MSFT_s004
                                 + 1.36828 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.01666 w_k1_ADM_s004
                                 + 1.15183 w_k1_AXP_s004
                                 + 1.18605 w_k1_DIS_s004
                                 + 1.02827 w_k1_FDX_s004
                                 + 1.55534 w_k1_IBM_s004
                                 + 1.09756 w_k1_INTC_s004
                                 + 1.17235 w_k1_KO_s004 + 1.34687 w_k1_MAT_s004
                                 + 1.17214 w_k1_MSFT_s004
                                 + 1.05448 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.85697 w_k1_ADM_s004
                                 + 0.894262 w_k1_AXP_s004
                                 + 0.841651 w_k1_DIS_s004
                                 + 0.775059 w_k1_FDX_s004
                                 + 1.30031 w_k1_IBM_s004
                                 + 0.892834 w_k1_INTC_s004
                                 + 1.05596 w_k1_KO_s004 + 1.28848 w_k1_MAT_s004
                                 + 0.969461 w_k1_MSFT_s004
                                 + 1.03372 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.984075 w_k1_ADM_s004
                                 + 1.23637 w_k1_AXP_s004
                                 + 0.966482 w_k1_DIS_s004
                                 + 1.12316 w_k1_FDX_s004
                                 + 0.722797 w_k1_IBM_s004
                                 + 1.08578 w_k1_INTC_s004
                                 + 0.857664 w_k1_KO_s004
                                 + 0.965966 w_k1_MAT_s004
                                 + 1.04941 w_k1_MSFT_s004 + 1.0132 w_k1_PG_s004
                                 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.11961 w_k1_ADM_s004
                                 + 1.18485 w_k1_AXP_s004
                                 + 1.11862 w_k1_DIS_s004 + 1.0399 w_k1_FDX_s004
                                 + 1.17443 w_k1_IBM_s004
                                 + 1.07297 w_k1_INTC_s004
                                 + 1.09662 w_k1_KO_s004 + 1.39685 w_k1_MAT_s004
                                 + 1.09292 w_k1_MSFT_s004
                                 + 1.17677 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.28922 w_k1_ADM_s004
                                 + 1.37876 w_k1_AXP_s004
                                 + 1.35398 w_k1_DIS_s004 + 1.3343 w_k1_FDX_s004
                                 + 0.834773 w_k1_IBM_s004
                                 + 1.06777 w_k1_INTC_s004
                                 + 0.912841 w_k1_KO_s004
                                 + 0.877536 w_k1_MAT_s004
                                 + 1.2458 w_k1_MSFT_s004 + 1.09024 w_k1_PG_s004
                                 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.29825 w_k1_ADM_s005
                                 + 1.17161 w_k1_AXP_s005
                                 + 1.16651 w_k1_DIS_s005
                                 + 1.02699 w_k1_FDX_s005
                                 + 1.67953 w_k1_IBM_s005
                                 + 0.779441 w_k1_INTC_s005
                                 + 1.2802 w_k1_KO_s005 + 1.76166 w_k1_MAT_s005
                                 + 0.95625 w_k1_MSFT_s005 + 1.0249 w_k1_PG_s005
                                 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.975882 w_k1_ADM_s005
                                 + 0.867883 w_k1_AXP_s005
                                 + 0.693883 w_k1_DIS_s005
                                 + 0.870698 w_k1_FDX_s005
                                 + 1.02676 w_k1_IBM_s005
                                 + 0.790304 w_k1_INTC_s005
                                 + 0.875449 w_k1_KO_s005
                                 + 1.28593 w_k1_MAT_s005
                                 + 0.866914 w_k1_MSFT_s005
                                 + 0.848857 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.922576 w_k1_ADM_s005
                                 + 0.701422 w_k1_AXP_s005
                                 + 0.640967 w_k1_DIS_s005
                                 + 0.781618 w_k1_FDX_s005
                                 + 0.75243 w_k1_IBM_s005
                                 + 0.834918 w_k1_INTC_s005
                                 + 0.723457 w_k1_KO_s005
                                 + 0.587864 w_k1_MAT_s005
                                 + 0.877295 w_k1_MSFT_s005
                                 + 0.832079 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.0518 w_k1_ADM_s005
                                 + 1.25752 w_k1_AXP_s005
                                 + 1.12304 w_k1_DIS_s005 + 1.0637 w_k1_FDX_s005
                                 + 1.33766 w_k1_IBM_s005
                                 + 0.871589 w_k1_INTC_s005
                                 + 1.07232 w_k1_KO_s005 + 1.83436 w_k1_MAT_s005
                                 + 1.07967 w_k1_MSFT_s005
                                 + 0.962155 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.03611 w_k1_ADM_s005
                                 + 1.13903 w_k1_AXP_s005
                                 + 1.01603 w_k1_DIS_s005
                                 + 1.14639 w_k1_FDX_s005
                                 + 0.82398 w_k1_IBM_s005
                                 + 1.01211 w_k1_INTC_s005
                                 + 0.881646 w_k1_KO_s005
                                 + 1.22359 w_k1_MAT_s005
                                 + 0.999027 w_k1_MSFT_s005
                                 + 0.933411 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.91074 w_k1_ADM_s005
                                 + 0.834325 w_k1_AXP_s005
                                 + 0.75893 w_k1_DIS_s005
                                 + 0.956624 w_k1_FDX_s005
                                 + 0.545359 w_k1_IBM_s005
                                 + 0.904964 w_k1_INTC_s005
                                 + 0.643086 w_k1_KO_s005
                                 + 0.62434 w_k1_MAT_s005
                                 + 0.922069 w_k1_MSFT_s005
                                 + 0.817199 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.22376 w_k1_ADM_s005
                                 + 1.69261 w_k1_AXP_s005
                                 + 1.61946 w_k1_DIS_s005 + 1.3585 w_k1_FDX_s005
                                 + 1.29651 w_k1_IBM_s005
                                 + 1.29261 w_k1_INTC_s005
                                 + 1.19254 w_k1_KO_s005 + 1.53099 w_k1_MAT_s005
                                 + 1.27451 w_k1_MSFT_s005
                                 + 1.19778 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.984455 w_k1_ADM_s005
                                 + 1.09588 w_k1_AXP_s005
                                 + 1.10403 w_k1_DIS_s005
                                 + 0.993518 w_k1_FDX_s005
                                 + 1.08843 w_k1_IBM_s005
                                 + 1.05795 w_k1_INTC_s005
                                 + 1.03138 w_k1_KO_s005 + 1.11966 w_k1_MAT_s005
                                 + 1.01995 w_k1_MSFT_s005 + 1.0111 w_k1_PG_s005
                                 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.850559 w_k1_ADM_s006
                                 + 1.21565 w_k1_AXP_s006
                                 + 1.19977 w_k1_DIS_s006
                                 + 1.23563 w_k1_FDX_s006
                                 + 0.80648 w_k1_IBM_s006
                                 + 1.29256 w_k1_INTC_s006
                                 + 0.950541 w_k1_KO_s006
                                 + 0.929501 w_k1_MAT_s006
                                 + 1.37435 w_k1_MSFT_s006
                                 + 1.03283 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.916272 w_k1_ADM_s006
                                 + 0.884895 w_k1_AXP_s006
                                 + 0.954386 w_k1_DIS_s006
                                 + 1.10626 w_k1_FDX_s006
                                 + 0.78302 w_k1_IBM_s006
                                 + 1.02888 w_k1_INTC_s006
                                 + 0.757446 w_k1_KO_s006
                                 + 0.896709 w_k1_MAT_s006
                                 + 1.04801 w_k1_MSFT_s006
                                 + 0.859865 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.06838 w_k1_ADM_s006
                                 + 1.13197 w_k1_AXP_s006
                                 + 1.66639 w_k1_DIS_s006
                                 + 1.37775 w_k1_FDX_s006
                                 + 1.02192 w_k1_IBM_s006
                                 + 1.62231 w_k1_INTC_s006
                                 + 1.18329 w_k1_KO_s006 + 0.88469 w_k1_MAT_s006
                                 + 1.47914 w_k1_MSFT_s006 + 1.1108 w_k1_PG_s006
                                 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.00166 w_k1_ADM_s006
                                 + 1.16517 w_k1_AXP_s006
                                 + 0.937659 w_k1_DIS_s006
                                 + 0.899458 w_k1_FDX_s006
                                 + 0.951853 w_k1_IBM_s006
                                 + 0.845492 w_k1_INTC_s006
                                 + 0.928767 w_k1_KO_s006 + 1.3466 w_k1_MAT_s006
                                 + 0.912544 w_k1_MSFT_s006
                                 + 0.977654 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.985165 w_k1_ADM_s006
                                 + 0.8164 w_k1_AXP_s006
                                 + 0.962178 w_k1_DIS_s006
                                 + 0.947888 w_k1_FDX_s006
                                 + 0.967684 w_k1_IBM_s006
                                 + 0.783413 w_k1_INTC_s006
                                 + 0.974547 w_k1_KO_s006
                                 + 1.02339 w_k1_MAT_s006
                                 + 1.00294 w_k1_MSFT_s006
                                 + 0.994716 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.964094 w_k1_ADM_s006
                                 + 0.922346 w_k1_AXP_s006
                                 + 0.912238 w_k1_DIS_s006
                                 + 1.00106 w_k1_FDX_s006
                                 + 0.691652 w_k1_IBM_s006
                                 + 0.853528 w_k1_INTC_s006
                                 + 0.845768 w_k1_KO_s006
                                 + 0.856391 w_k1_MAT_s006
                                 + 0.962274 w_k1_MSFT_s006
                                 + 0.942042 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.760916 w_k1_ADM_s006
                                 + 0.703337 w_k1_AXP_s006
                                 + 0.592202 w_k1_DIS_s006
                                 + 0.730554 w_k1_FDX_s006
                                 + 0.670609 w_k1_IBM_s006
                                 + 0.799414 w_k1_INTC_s006
                                 + 0.699873 w_k1_KO_s006
                                 + 0.857456 w_k1_MAT_s006
                                 + 0.862004 w_k1_MSFT_s006
                                 + 0.730956 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.8092 w_k1_ADM_s006
                                 + 1.01218 w_k1_AXP_s006
                                 + 1.39279 w_k1_DIS_s006
                                 + 1.09974 w_k1_FDX_s006
                                 + 1.37986 w_k1_IBM_s006
                                 + 0.993814 w_k1_INTC_s006
                                 + 1.09168 w_k1_KO_s006 + 1.64115 w_k1_MAT_s006
                                 + 1.11733 w_k1_MSFT_s006
                                 + 0.94807 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.34811 w_k1_ADM_s007
                                 + 1.34517 w_k1_AXP_s007
                                 + 1.49838 w_k1_DIS_s007
                                 + 1.27492 w_k1_FDX_s007
                                 + 1.27953 w_k1_IBM_s007
                                 + 1.02409 w_k1_INTC_s007
                                 + 1.41811 w_k1_KO_s007 + 1.60006 w_k1_MAT_s007
                                 + 1.26495 w_k1_MSFT_s007
                                 + 1.31016 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.984491 w_k1_ADM_s007
                                 + 1.2108 w_k1_AXP_s007 + 1.3043 w_k1_DIS_s007
                                 + 1.23457 w_k1_FDX_s007
                                 + 1.01563 w_k1_IBM_s007
                                 + 1.22718 w_k1_INTC_s007
                                 + 1.08919 w_k1_KO_s007 + 1.31007 w_k1_MAT_s007
                                 + 1.1645 w_k1_MSFT_s007 + 1.04484 w_k1_PG_s007
                                 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.957689 w_k1_ADM_s007
                                 + 1.04504 w_k1_AXP_s007
                                 + 0.975715 w_k1_DIS_s007
                                 + 0.919786 w_k1_FDX_s007
                                 + 1.34972 w_k1_IBM_s007
                                 + 1.21378 w_k1_INTC_s007 + 1.3464 w_k1_KO_s007
                                 + 1.31828 w_k1_MAT_s007
                                 + 1.1369 w_k1_MSFT_s007 + 1.07332 w_k1_PG_s007
                                 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.0769 w_k1_ADM_s007
                                 + 1.24249 w_k1_AXP_s007
                                 + 1.04031 w_k1_DIS_s007
                                 + 1.08484 w_k1_FDX_s007
                                 + 1.08288 w_k1_IBM_s007
                                 + 0.992267 w_k1_INTC_s007
                                 + 0.985189 w_k1_KO_s007
                                 + 1.40229 w_k1_MAT_s007
                                 + 1.02843 w_k1_MSFT_s007
                                 + 1.03751 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.012 w_k1_ADM_s007
                                 + 0.987611 w_k1_AXP_s007
                                 + 0.741183 w_k1_DIS_s007
                                 + 0.905388 w_k1_FDX_s007
                                 + 0.667296 w_k1_IBM_s007
                                 + 0.942494 w_k1_INTC_s007
                                 + 0.903241 w_k1_KO_s007
                                 + 1.26331 w_k1_MAT_s007
                                 + 0.977145 w_k1_MSFT_s007
                                 + 0.956082 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.11014 w_k1_ADM_s007
                                 + 0.856612 w_k1_AXP_s007
                                 + 0.971231 w_k1_DIS_s007
                                 + 1.04362 w_k1_FDX_s007
                                 + 0.86602 w_k1_IBM_s007
                                 + 1.2903 w_k1_INTC_s007 + 1.04278 w_k1_KO_s007
                                 + 0.813013 w_k1_MAT_s007
                                 + 1.28003 w_k1_MSFT_s007
                                 + 1.05956 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.11632 w_k1_ADM_s007
                                 + 1.2252 w_k1_AXP_s007 + 1.5183 w_k1_DIS_s007
                                 + 1.2142 w_k1_FDX_s007 + 1.45571 w_k1_IBM_s007
                                 + 1.35298 w_k1_INTC_s007
                                 + 1.42942 w_k1_KO_s007 + 1.43673 w_k1_MAT_s007
                                 + 1.41249 w_k1_MSFT_s007
                                 + 1.24481 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.780831 w_k1_ADM_s007
                                 + 0.773128 w_k1_AXP_s007
                                 + 0.593101 w_k1_DIS_s007
                                 + 0.803744 w_k1_FDX_s007
                                 + 0.851186 w_k1_IBM_s007
                                 + 0.866947 w_k1_INTC_s007
                                 + 0.860214 w_k1_KO_s007
                                 + 0.673721 w_k1_MAT_s007
                                 + 0.749259 w_k1_MSFT_s007
                                 + 0.89429 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.975574 w_k1_ADM_s010
                                 + 1.47059 w_k1_AXP_s010
                                 + 1.51328 w_k1_DIS_s010
                                 + 1.27386 w_k1_FDX_s010
                                 + 1.11009 w_k1_IBM_s010
                                 + 1.52972 w_k1_INTC_s010
                                 + 1.16304 w_k1_KO_s010 + 1.46569 w_k1_MAT_s010
                                 + 1.23495 w_k1_MSFT_s010
                                 + 1.28685 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.853223 w_k1_ADM_s010
                                 + 0.991929 w_k1_AXP_s010
                                 + 0.870236 w_k1_DIS_s010
                                 + 1.00886 w_k1_FDX_s010
                                 + 0.762726 w_k1_IBM_s010
                                 + 1.10153 w_k1_INTC_s010
                                 + 0.896474 w_k1_KO_s010
                                 + 0.642895 w_k1_MAT_s010
                                 + 0.926256 w_k1_MSFT_s010
                                 + 0.90976 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.04149 w_k1_ADM_s010
                                 + 1.25425 w_k1_AXP_s010 + 1.5332 w_k1_DIS_s010
                                 + 1.22967 w_k1_FDX_s010
                                 + 1.65894 w_k1_IBM_s010
                                 + 1.28194 w_k1_INTC_s010
                                 + 1.41213 w_k1_KO_s010 + 2.41423 w_k1_MAT_s010
                                 + 1.18613 w_k1_MSFT_s010
                                 + 1.28681 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.771425 w_k1_ADM_s010
                                 + 0.858286 w_k1_AXP_s010
                                 + 0.814095 w_k1_DIS_s010
                                 + 0.725969 w_k1_FDX_s010
                                 + 1.32348 w_k1_IBM_s010
                                 + 0.833878 w_k1_INTC_s010
                                 + 0.95063 w_k1_KO_s010 + 1.08049 w_k1_MAT_s010
                                 + 0.801229 w_k1_MSFT_s010
                                 + 0.97605 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.609037 w_k1_ADM_s010
                                 + 0.710768 w_k1_AXP_s010
                                 + 0.684293 w_k1_DIS_s010
                                 + 0.754616 w_k1_FDX_s010
                                 + 0.892367 w_k1_IBM_s010
                                 + 0.989126 w_k1_INTC_s010
                                 + 0.775045 w_k1_KO_s010
                                 + 1.13133 w_k1_MAT_s010
                                 + 0.829856 w_k1_MSFT_s010
                                 + 0.794798 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.696576 w_k1_ADM_s010
                                 + 0.930184 w_k1_AXP_s010
                                 + 1.24464 w_k1_DIS_s010
                                 + 0.99549 w_k1_FDX_s010
                                 + 1.21964 w_k1_IBM_s010
                                 + 1.14007 w_k1_INTC_s010
                                 + 1.09526 w_k1_KO_s010 + 1.39371 w_k1_MAT_s010
                                 + 1.03766 w_k1_MSFT_s010 + 1.019 w_k1_PG_s010
                                 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.831924 w_k1_ADM_s010
                                 + 1.22615 w_k1_AXP_s010
                                 + 0.979827 w_k1_DIS_s010
                                 + 1.0825 w_k1_FDX_s010
                                 + 0.937584 w_k1_IBM_s010
                                 + 1.20284 w_k1_INTC_s010 + 1.043 w_k1_KO_s010
                                 + 1.30028 w_k1_MAT_s010
                                 + 1.11576 w_k1_MSFT_s010
                                 + 1.02492 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.758933 w_k1_ADM_s010
                                 + 0.757356 w_k1_AXP_s010
                                 + 0.722135 w_k1_DIS_s010
                                 + 0.801791 w_k1_FDX_s010
                                 + 0.602808 w_k1_IBM_s010
                                 + 0.837804 w_k1_INTC_s010
                                 + 0.599729 w_k1_KO_s010
                                 + 0.822927 w_k1_MAT_s010
                                 + 0.815445 w_k1_MSFT_s010
                                 + 0.73601 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.02429 w_k1_ADM_s011
                                 + 1.36229 w_k1_AXP_s011
                                 + 1.54532 w_k1_DIS_s011 + 1.3311 w_k1_FDX_s011
                                 + 0.948578 w_k1_IBM_s011
                                 + 1.16793 w_k1_INTC_s011 + 1.0279 w_k1_KO_s011
                                 + 1.17095 w_k1_MAT_s011
                                 + 1.33723 w_k1_MSFT_s011
                                 + 1.10137 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.1924 w_k1_ADM_s011
                                 + 1.69392 w_k1_AXP_s011
                                 + 1.54488 w_k1_DIS_s011
                                 + 1.40772 w_k1_FDX_s011
                                 + 1.71614 w_k1_IBM_s011
                                 + 1.56467 w_k1_INTC_s011 + 1.3674 w_k1_KO_s011
                                 + 1.41373 w_k1_MAT_s011
                                 + 1.27237 w_k1_MSFT_s011
                                 + 1.36931 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.10543 w_k1_ADM_s011
                                 + 1.29479 w_k1_AXP_s011
                                 + 1.11875 w_k1_DIS_s011
                                 + 1.12956 w_k1_FDX_s011
                                 + 1.21051 w_k1_IBM_s011
                                 + 1.31509 w_k1_INTC_s011
                                 + 1.17133 w_k1_KO_s011 + 1.43299 w_k1_MAT_s011
                                 + 1.15553 w_k1_MSFT_s011
                                 + 1.23483 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.954226 w_k1_ADM_s011
                                 + 0.923557 w_k1_AXP_s011
                                 + 0.74395 w_k1_DIS_s011
                                 + 0.857744 w_k1_FDX_s011
                                 + 1.25519 w_k1_IBM_s011
                                 + 0.975605 w_k1_INTC_s011
                                 + 0.985179 w_k1_KO_s011
                                 + 1.13489 w_k1_MAT_s011
                                 + 0.94505 w_k1_MSFT_s011
                                 + 0.844525 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.869997 w_k1_ADM_s011
                                 + 0.640243 w_k1_AXP_s011
                                 + 0.598639 w_k1_DIS_s011
                                 + 0.755758 w_k1_FDX_s011
                                 + 0.716737 w_k1_IBM_s011
                                 + 0.77601 w_k1_INTC_s011
                                 + 0.675661 w_k1_KO_s011
                                 + 0.526829 w_k1_MAT_s011
                                 + 0.789157 w_k1_MSFT_s011
                                 + 0.771378 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.869182 w_k1_ADM_s011
                                 + 0.978342 w_k1_AXP_s011
                                 + 0.825655 w_k1_DIS_s011
                                 + 0.94134 w_k1_FDX_s011
                                 + 0.961409 w_k1_IBM_s011
                                 + 0.917985 w_k1_INTC_s011
                                 + 0.769307 w_k1_KO_s011
                                 + 1.15069 w_k1_MAT_s011
                                 + 0.867585 w_k1_MSFT_s011
                                 + 0.843662 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.12698 w_k1_ADM_s011
                                 + 0.85972 w_k1_AXP_s011
                                 + 1.01478 w_k1_DIS_s011
                                 + 1.14858 w_k1_FDX_s011
                                 + 0.748544 w_k1_IBM_s011
                                 + 1.29975 w_k1_INTC_s011
                                 + 0.875451 w_k1_KO_s011
                                 + 0.680825 w_k1_MAT_s011
                                 + 1.24929 w_k1_MSFT_s011
                                 + 0.864406 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.93172 w_k1_ADM_s011
                                 + 1.0331 w_k1_AXP_s011
                                 + 0.868041 w_k1_DIS_s011
                                 + 0.966607 w_k1_FDX_s011
                                 + 0.711236 w_k1_IBM_s011
                                 + 1.28282 w_k1_INTC_s011
                                 + 0.846524 w_k1_KO_s011
                                 + 0.853226 w_k1_MAT_s011
                                 + 0.965869 w_k1_MSFT_s011
                                 + 0.899542 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.941002 w_k1_ADM_s012
                                 + 0.877905 w_k1_AXP_s012
                                 + 0.835934 w_k1_DIS_s012
                                 + 0.780572 w_k1_FDX_s012
                                 + 1.18833 w_k1_IBM_s012
                                 + 1.03208 w_k1_INTC_s012
                                 + 0.903895 w_k1_KO_s012
                                 + 0.964482 w_k1_MAT_s012
                                 + 0.900305 w_k1_MSFT_s012
                                 + 0.917567 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.22876 w_k1_ADM_s012
                                 + 0.924458 w_k1_AXP_s012
                                 + 0.950475 w_k1_DIS_s012
                                 + 0.993443 w_k1_FDX_s012
                                 + 0.938053 w_k1_IBM_s012
                                 + 0.991351 w_k1_INTC_s012
                                 + 0.913628 w_k1_KO_s012
                                 + 0.667621 w_k1_MAT_s012
                                 + 0.903824 w_k1_MSFT_s012
                                 + 0.97852 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.18499 w_k1_ADM_s012
                                 + 1.42026 w_k1_AXP_s012
                                 + 1.38648 w_k1_DIS_s012
                                 + 1.36494 w_k1_FDX_s012
                                 + 0.966925 w_k1_IBM_s012
                                 + 1.28066 w_k1_INTC_s012
                                 + 0.967945 w_k1_KO_s012
                                 + 1.19424 w_k1_MAT_s012
                                 + 1.55706 w_k1_MSFT_s012 + 1.354 w_k1_PG_s012
                                 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.62942 w_k1_ADM_s012
                                 + 1.91081 w_k1_AXP_s012
                                 + 1.73265 w_k1_DIS_s012
                                 + 1.53756 w_k1_FDX_s012
                                 + 1.25997 w_k1_IBM_s012
                                 + 1.3023 w_k1_INTC_s012 + 1.17761 w_k1_KO_s012
                                 + 1.52147 w_k1_MAT_s012
                                 + 1.42117 w_k1_MSFT_s012
                                 + 1.38135 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.04364 w_k1_ADM_s012
                                 + 0.747167 w_k1_AXP_s012
                                 + 0.739211 w_k1_DIS_s012
                                 + 0.850364 w_k1_FDX_s012
                                 + 0.721635 w_k1_IBM_s012
                                 + 0.999342 w_k1_INTC_s012
                                 + 0.702461 w_k1_KO_s012
                                 + 0.383237 w_k1_MAT_s012
                                 + 0.78066 w_k1_MSFT_s012
                                 + 0.949163 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.03062 w_k1_ADM_s012
                                 + 1.21121 w_k1_AXP_s012
                                 + 1.35748 w_k1_DIS_s012
                                 + 0.987098 w_k1_FDX_s012
                                 + 1.49891 w_k1_IBM_s012
                                 + 1.02828 w_k1_INTC_s012
                                 + 1.32645 w_k1_KO_s012 + 1.32967 w_k1_MAT_s012
                                 + 1.1168 w_k1_MSFT_s012 + 1.23271 w_k1_PG_s012
                                 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.44245 w_k1_ADM_s012
                                 + 1.30895 w_k1_AXP_s012
                                 + 1.32683 w_k1_DIS_s012
                                 + 1.33007 w_k1_FDX_s012
                                 + 0.771154 w_k1_IBM_s012
                                 + 1.39843 w_k1_INTC_s012
                                 + 0.911006 w_k1_KO_s012
                                 + 0.77387 w_k1_MAT_s012
                                 + 1.36785 w_k1_MSFT_s012
                                 + 1.24734 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.35094 w_k1_ADM_s012
                                 + 1.14208 w_k1_AXP_s012
                                 + 1.13532 w_k1_DIS_s012
                                 + 1.14041 w_k1_FDX_s012
                                 + 1.10357 w_k1_IBM_s012
                                 + 1.31505 w_k1_INTC_s012
                                 + 1.03116 w_k1_KO_s012
                                 + 0.915754 w_k1_MAT_s012
                                 + 1.04073 w_k1_MSFT_s012
                                 + 1.18807 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.989885 w_k1_ADM_s013
                                 + 0.609891 w_k1_AXP_s013
                                 + 0.726124 w_k1_DIS_s013
                                 + 0.921831 w_k1_FDX_s013
                                 + 0.541042 w_k1_IBM_s013
                                 + 0.724166 w_k1_INTC_s013
                                 + 0.758144 w_k1_KO_s013
                                 + 0.451557 w_k1_MAT_s013
                                 + 0.872826 w_k1_MSFT_s013
                                 + 0.974829 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.32992 w_k1_ADM_s013
                                 + 1.40228 w_k1_AXP_s013
                                 + 1.67075 w_k1_DIS_s013
                                 + 1.17195 w_k1_FDX_s013 + 1.413 w_k1_IBM_s013
                                 + 1.01518 w_k1_INTC_s013
                                 + 1.63521 w_k1_KO_s013 + 1.7729 w_k1_MAT_s013
                                 + 1.32083 w_k1_MSFT_s013
                                 + 1.32714 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.15776 w_k1_ADM_s013
                                 + 0.955092 w_k1_AXP_s013
                                 + 0.795871 w_k1_DIS_s013
                                 + 0.969905 w_k1_FDX_s013
                                 + 0.717646 w_k1_IBM_s013
                                 + 0.813937 w_k1_INTC_s013
                                 + 0.958425 w_k1_KO_s013
                                 + 0.887567 w_k1_MAT_s013
                                 + 1.04177 w_k1_MSFT_s013
                                 + 1.00764 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.17196 w_k1_ADM_s013
                                 + 1.25312 w_k1_AXP_s013
                                 + 0.879422 w_k1_DIS_s013
                                 + 0.964644 w_k1_FDX_s013
                                 + 1.33323 w_k1_IBM_s013
                                 + 0.703217 w_k1_INTC_s013
                                 + 1.09651 w_k1_KO_s013 + 1.69957 w_k1_MAT_s013
                                 + 0.957081 w_k1_MSFT_s013
                                 + 1.25932 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.914051 w_k1_ADM_s013
                                 + 0.913972 w_k1_AXP_s013
                                 + 1.14524 w_k1_DIS_s013
                                 + 0.952162 w_k1_FDX_s013
                                 + 1.12952 w_k1_IBM_s013
                                 + 0.842222 w_k1_INTC_s013
                                 + 1.09811 w_k1_KO_s013 + 1.20553 w_k1_MAT_s013
                                 + 1.16089 w_k1_MSFT_s013
                                 + 1.10262 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.31452 w_k1_ADM_s013
                                 + 1.34032 w_k1_AXP_s013
                                 + 1.19181 w_k1_DIS_s013
                                 + 1.23664 w_k1_FDX_s013
                                 + 0.957326 w_k1_IBM_s013
                                 + 1.09139 w_k1_INTC_s013
                                 + 1.27696 w_k1_KO_s013 + 1.1917 w_k1_MAT_s013
                                 + 1.24936 w_k1_MSFT_s013
                                 + 1.16066 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.681475 w_k1_ADM_s013
                                 + 0.711554 w_k1_AXP_s013
                                 + 0.765114 w_k1_DIS_s013
                                 + 0.835796 w_k1_FDX_s013
                                 + 0.877611 w_k1_IBM_s013
                                 + 0.767505 w_k1_INTC_s013
                                 + 0.843325 w_k1_KO_s013
                                 + 0.658317 w_k1_MAT_s013
                                 + 0.833491 w_k1_MSFT_s013
                                 + 0.859942 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.977692 w_k1_ADM_s013
                                 + 0.932757 w_k1_AXP_s013
                                 + 1.00145 w_k1_DIS_s013
                                 + 1.11277 w_k1_FDX_s013
                                 + 0.461692 w_k1_IBM_s013
                                 + 0.920873 w_k1_INTC_s013
                                 + 0.876817 w_k1_KO_s013
                                 + 0.609518 w_k1_MAT_s013
                                 + 1.13849 w_k1_MSFT_s013
                                 + 1.01918 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.723364 w_k1_ADM_s014
                                 + 0.558138 w_k1_AXP_s014
                                 + 0.544467 w_k1_DIS_s014
                                 + 0.705917 w_k1_FDX_s014
                                 + 0.764003 w_k1_IBM_s014
                                 + 0.639618 w_k1_INTC_s014
                                 + 0.781347 w_k1_KO_s014
                                 + 0.772667 w_k1_MAT_s014
                                 + 0.735661 w_k1_MSFT_s014
                                 + 0.755319 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.729249 w_k1_ADM_s014
                                 + 0.962862 w_k1_AXP_s014
                                 + 0.587931 w_k1_DIS_s014
                                 + 0.686954 w_k1_FDX_s014
                                 + 1.13479 w_k1_IBM_s014
                                 + 0.603662 w_k1_INTC_s014
                                 + 0.698591 w_k1_KO_s014
                                 + 1.28151 w_k1_MAT_s014
                                 + 0.614758 w_k1_MSFT_s014
                                 + 0.77873 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.920989 w_k1_ADM_s014
                                 + 1.12273 w_k1_AXP_s014
                                 + 0.826371 w_k1_DIS_s014
                                 + 0.954916 w_k1_FDX_s014
                                 + 1.02936 w_k1_IBM_s014
                                 + 0.967644 w_k1_INTC_s014
                                 + 0.969322 w_k1_KO_s014
                                 + 1.06719 w_k1_MAT_s014
                                 + 0.886418 w_k1_MSFT_s014
                                 + 0.851724 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.01638 w_k1_ADM_s014
                                 + 0.726359 w_k1_AXP_s014
                                 + 1.06782 w_k1_DIS_s014
                                 + 1.01024 w_k1_FDX_s014
                                 + 1.05897 w_k1_IBM_s014
                                 + 0.964264 w_k1_INTC_s014
                                 + 1.22123 w_k1_KO_s014
                                 + 0.778763 w_k1_MAT_s014
                                 + 0.967636 w_k1_MSFT_s014
                                 + 0.955374 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.11342 w_k1_ADM_s014
                                 + 1.42601 w_k1_AXP_s014
                                 + 1.18758 w_k1_DIS_s014
                                 + 1.25629 w_k1_FDX_s014
                                 + 1.16172 w_k1_IBM_s014
                                 + 1.16592 w_k1_INTC_s014
                                 + 1.08537 w_k1_KO_s014 + 1.51127 w_k1_MAT_s014
                                 + 1.03084 w_k1_MSFT_s014
                                 + 1.05351 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.841246 w_k1_ADM_s014
                                 + 0.746901 w_k1_AXP_s014
                                 + 0.88076 w_k1_DIS_s014
                                 + 0.873489 w_k1_FDX_s014
                                 + 0.982148 w_k1_IBM_s014
                                 + 0.727972 w_k1_INTC_s014
                                 + 0.92381 w_k1_KO_s014
                                 + 0.836146 w_k1_MAT_s014
                                 + 0.853911 w_k1_MSFT_s014
                                 + 0.860086 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.04102 w_k1_ADM_s014
                                 + 0.959264 w_k1_AXP_s014
                                 + 0.689211 w_k1_DIS_s014
                                 + 1.05342 w_k1_FDX_s014
                                 + 0.609209 w_k1_IBM_s014
                                 + 0.741005 w_k1_INTC_s014
                                 + 0.756078 w_k1_KO_s014
                                 + 0.806557 w_k1_MAT_s014
                                 + 0.93374 w_k1_MSFT_s014
                                 + 0.856427 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.06462 w_k1_ADM_s014
                                 + 1.0908 w_k1_AXP_s014 + 1.6801 w_k1_DIS_s014
                                 + 1.33758 w_k1_FDX_s014
                                 + 1.16345 w_k1_IBM_s014
                                 + 1.13064 w_k1_INTC_s014 + 1.374 w_k1_KO_s014
                                 + 0.992122 w_k1_MAT_s014
                                 + 1.28903 w_k1_MSFT_s014
                                 + 1.09207 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.58691 w_k1_ADM_s015
                                 + 1.61806 w_k1_AXP_s015 + 1.7108 w_k1_DIS_s015
                                 + 1.43065 w_k1_FDX_s015
                                 + 1.62634 w_k1_IBM_s015
                                 + 1.01961 w_k1_INTC_s015
                                 + 1.35359 w_k1_KO_s015 + 1.62413 w_k1_MAT_s015
                                 + 1.14006 w_k1_MSFT_s015
                                 + 1.16647 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.804134 w_k1_ADM_s015
                                 + 0.551922 w_k1_AXP_s015
                                 + 0.696145 w_k1_DIS_s015
                                 + 0.681063 w_k1_FDX_s015
                                 + 0.759038 w_k1_IBM_s015
                                 + 0.763489 w_k1_INTC_s015
                                 + 0.834729 w_k1_KO_s015
                                 + 0.608138 w_k1_MAT_s015
                                 + 0.77983 w_k1_MSFT_s015
                                 + 0.772084 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.00756 w_k1_ADM_s015
                                 + 1.07418 w_k1_AXP_s015
                                 + 0.968708 w_k1_DIS_s015
                                 + 0.935871 w_k1_FDX_s015
                                 + 0.939794 w_k1_IBM_s015
                                 + 0.922546 w_k1_INTC_s015
                                 + 1.0383 w_k1_KO_s015 + 1.09382 w_k1_MAT_s015
                                 + 0.989324 w_k1_MSFT_s015
                                 + 1.06244 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.31813 w_k1_ADM_s015
                                 + 1.25585 w_k1_AXP_s015
                                 + 1.58773 w_k1_DIS_s015
                                 + 1.20025 w_k1_FDX_s015
                                 + 1.08808 w_k1_IBM_s015
                                 + 1.00571 w_k1_INTC_s015
                                 + 1.13541 w_k1_KO_s015 + 1.11671 w_k1_MAT_s015
                                 + 1.21887 w_k1_MSFT_s015
                                 + 1.06052 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.12412 w_k1_ADM_s015
                                 + 0.812059 w_k1_AXP_s015
                                 + 0.838175 w_k1_DIS_s015
                                 + 0.965592 w_k1_FDX_s015
                                 + 0.640803 w_k1_IBM_s015
                                 + 0.709902 w_k1_INTC_s015
                                 + 0.831656 w_k1_KO_s015
                                 + 0.773677 w_k1_MAT_s015
                                 + 1.00846 w_k1_MSFT_s015
                                 + 0.815986 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.0228 w_k1_ADM_s015
                                 + 0.99397 w_k1_AXP_s015
                                 + 1.26269 w_k1_DIS_s015 + 1.164 w_k1_FDX_s015
                                 + 0.769079 w_k1_IBM_s015
                                 + 0.846545 w_k1_INTC_s015
                                 + 1.12011 w_k1_KO_s015 + 0.92995 w_k1_MAT_s015
                                 + 1.237 w_k1_MSFT_s015 + 1.07213 w_k1_PG_s015
                                 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.09727 w_k1_ADM_s015
                                 + 1.23568 w_k1_AXP_s015
                                 + 0.911443 w_k1_DIS_s015
                                 + 0.996251 w_k1_FDX_s015
                                 + 1.10311 w_k1_IBM_s015
                                 + 0.910399 w_k1_INTC_s015
                                 + 1.04011 w_k1_KO_s015 + 1.22768 w_k1_MAT_s015
                                 + 0.907447 w_k1_MSFT_s015
                                 + 1.07128 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.16502 w_k1_ADM_s015
                                 + 0.878114 w_k1_AXP_s015
                                 + 0.879339 w_k1_DIS_s015
                                 + 0.880829 w_k1_FDX_s015
                                 + 1.44806 w_k1_IBM_s015
                                 + 0.808918 w_k1_INTC_s015
                                 + 1.02819 w_k1_KO_s015 + 1.29128 w_k1_MAT_s015
                                 + 0.889823 w_k1_MSFT_s015
                                 + 1.01801 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.815675 w_k1_ADM_s016
                                 + 0.675895 w_k1_AXP_s016
                                 + 0.933283 w_k1_DIS_s016
                                 + 0.855204 w_k1_FDX_s016
                                 + 0.887811 w_k1_IBM_s016
                                 + 0.805146 w_k1_INTC_s016
                                 + 1.04979 w_k1_KO_s016
                                 + 0.604824 w_k1_MAT_s016
                                 + 0.906962 w_k1_MSFT_s016
                                 + 1.01439 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.676399 w_k1_ADM_s016
                                 + 0.607571 w_k1_AXP_s016
                                 + 0.553033 w_k1_DIS_s016
                                 + 0.66747 w_k1_FDX_s016
                                 + 0.681287 w_k1_IBM_s016
                                 + 0.696743 w_k1_INTC_s016
                                 + 0.766093 w_k1_KO_s016
                                 + 0.662489 w_k1_MAT_s016
                                 + 0.860958 w_k1_MSFT_s016
                                 + 0.941736 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.682576 w_k1_ADM_s016
                                 + 0.723659 w_k1_AXP_s016
                                 + 0.717259 w_k1_DIS_s016
                                 + 0.814598 w_k1_FDX_s016
                                 + 0.840729 w_k1_IBM_s016
                                 + 0.791779 w_k1_INTC_s016
                                 + 0.833353 w_k1_KO_s016
                                 + 0.836067 w_k1_MAT_s016
                                 + 0.763429 w_k1_MSFT_s016
                                 + 0.989091 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.976764 w_k1_ADM_s016
                                 + 0.927483 w_k1_AXP_s016
                                 + 1.13007 w_k1_DIS_s016
                                 + 1.16157 w_k1_FDX_s016
                                 + 0.738146 w_k1_IBM_s016
                                 + 1.03739 w_k1_INTC_s016
                                 + 1.00011 w_k1_KO_s016
                                 + 0.729413 w_k1_MAT_s016
                                 + 1.05974 w_k1_MSFT_s016
                                 + 1.14136 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.905622 w_k1_ADM_s016
                                 + 1.13091 w_k1_AXP_s016 + 1.1907 w_k1_DIS_s016
                                 + 1.09275 w_k1_FDX_s016
                                 + 1.32057 w_k1_IBM_s016
                                 + 1.01069 w_k1_INTC_s016 + 1.3614 w_k1_KO_s016
                                 + 1.27374 w_k1_MAT_s016
                                 + 1.12396 w_k1_MSFT_s016 + 1.159 w_k1_PG_s016
                                 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.42925 w_k1_ADM_s016
                                 + 1.42662 w_k1_AXP_s016
                                 + 1.53979 w_k1_DIS_s016
                                 + 1.32821 w_k1_FDX_s016
                                 + 1.06727 w_k1_IBM_s016
                                 + 1.27219 w_k1_INTC_s016
                                 + 1.38687 w_k1_KO_s016 + 1.10202 w_k1_MAT_s016
                                 + 1.33152 w_k1_MSFT_s016
                                 + 1.46062 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.95577 w_k1_ADM_s016
                                 + 0.808333 w_k1_AXP_s016
                                 + 0.784246 w_k1_DIS_s016
                                 + 0.900101 w_k1_FDX_s016
                                 + 0.651453 w_k1_IBM_s016
                                 + 0.856449 w_k1_INTC_s016
                                 + 0.905331 w_k1_KO_s016
                                 + 0.616401 w_k1_MAT_s016
                                 + 1.05034 w_k1_MSFT_s016
                                 + 0.96615 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.801787 w_k1_ADM_s016
                                 + 0.876788 w_k1_AXP_s016
                                 + 0.861808 w_k1_DIS_s016
                                 + 0.869979 w_k1_FDX_s016
                                 + 0.99524 w_k1_IBM_s016
                                 + 0.648673 w_k1_INTC_s016
                                 + 0.967394 w_k1_KO_s016
                                 + 1.20665 w_k1_MAT_s016
                                 + 0.868709 w_k1_MSFT_s016
                                 + 1.05982 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.973303 w_k1_ADM_s017
                                 + 0.993715 w_k1_AXP_s017
                                 + 1.11513 w_k1_DIS_s017
                                 + 0.927667 w_k1_FDX_s017
                                 + 1.20146 w_k1_IBM_s017
                                 + 0.955987 w_k1_INTC_s017
                                 + 1.20448 w_k1_KO_s017 + 1.13614 w_k1_MAT_s017
                                 + 1.02265 w_k1_MSFT_s017
                                 + 1.00858 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.922807 w_k1_ADM_s017
                                 + 0.639163 w_k1_AXP_s017
                                 + 0.691737 w_k1_DIS_s017
                                 + 0.74912 w_k1_FDX_s017 + 1.067 w_k1_IBM_s017
                                 + 0.754507 w_k1_INTC_s017
                                 + 0.888644 w_k1_KO_s017
                                 + 0.575073 w_k1_MAT_s017
                                 + 0.748249 w_k1_MSFT_s017
                                 + 0.785336 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.868504 w_k1_ADM_s017
                                 + 0.621853 w_k1_AXP_s017
                                 + 0.56796 w_k1_DIS_s017
                                 + 0.776705 w_k1_FDX_s017
                                 + 0.669609 w_k1_IBM_s017
                                 + 0.754398 w_k1_INTC_s017
                                 + 0.652203 w_k1_KO_s017
                                 + 0.618814 w_k1_MAT_s017
                                 + 0.745377 w_k1_MSFT_s017
                                 + 0.789522 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.2564 w_k1_ADM_s017
                                 + 1.20306 w_k1_AXP_s017
                                 + 1.18367 w_k1_DIS_s017
                                 + 1.11707 w_k1_FDX_s017
                                 + 1.59688 w_k1_IBM_s017
                                 + 1.00767 w_k1_INTC_s017
                                 + 1.36286 w_k1_KO_s017 + 1.61838 w_k1_MAT_s017
                                 + 1.09652 w_k1_MSFT_s017
                                 + 1.06356 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.0439 w_k1_ADM_s017
                                 + 0.905595 w_k1_AXP_s017
                                 + 0.965549 w_k1_DIS_s017
                                 + 1.0707 w_k1_FDX_s017
                                 + 0.828067 w_k1_IBM_s017
                                 + 0.90431 w_k1_INTC_s017
                                 + 0.852847 w_k1_KO_s017
                                 + 0.90602 w_k1_MAT_s017
                                 + 0.976558 w_k1_MSFT_s017
                                 + 1.03811 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.25242 w_k1_ADM_s017
                                 + 0.903099 w_k1_AXP_s017
                                 + 1.51881 w_k1_DIS_s017
                                 + 1.23155 w_k1_FDX_s017
                                 + 0.91774 w_k1_IBM_s017
                                 + 0.964611 w_k1_INTC_s017
                                 + 1.2315 w_k1_KO_s017 + 0.568295 w_k1_MAT_s017
                                 + 1.1278 w_k1_MSFT_s017 + 1.03258 w_k1_PG_s017
                                 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.34871 w_k1_ADM_s017
                                 + 1.35461 w_k1_AXP_s017
                                 + 1.53951 w_k1_DIS_s017
                                 + 1.32512 w_k1_FDX_s017
                                 + 1.11566 w_k1_IBM_s017
                                 + 1.15308 w_k1_INTC_s017
                                 + 1.29581 w_k1_KO_s017 + 1.14939 w_k1_MAT_s017
                                 + 1.27312 w_k1_MSFT_s017
                                 + 1.22611 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.08003 w_k1_ADM_s017
                                 + 0.811122 w_k1_AXP_s017
                                 + 0.716165 w_k1_DIS_s017
                                 + 0.853761 w_k1_FDX_s017
                                 + 0.70415 w_k1_IBM_s017
                                 + 0.800124 w_k1_INTC_s017
                                 + 0.782426 w_k1_KO_s017
                                 + 0.549877 w_k1_MAT_s017
                                 + 0.896914 w_k1_MSFT_s017
                                 + 0.858787 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.744537 w_k1_ADM_s100
                                 + 0.852439 w_k1_AXP_s100
                                 + 0.891658 w_k1_DIS_s100
                                 + 0.791535 w_k1_FDX_s100
                                 + 1.29169 w_k1_IBM_s100
                                 + 1.03869 w_k1_INTC_s100
                                 + 1.10723 w_k1_KO_s100 + 0.99501 w_k1_MAT_s100
                                 + 1.01017 w_k1_MSFT_s100
                                 + 1.02556 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.828872 w_k1_ADM_s100
                                 + 0.675512 w_k1_AXP_s100
                                 + 0.634961 w_k1_DIS_s100
                                 + 0.763659 w_k1_FDX_s100
                                 + 0.769648 w_k1_IBM_s100
                                 + 0.877036 w_k1_INTC_s100
                                 + 0.787705 w_k1_KO_s100
                                 + 0.797627 w_k1_MAT_s100
                                 + 0.804109 w_k1_MSFT_s100
                                 + 0.774983 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.991083 w_k1_ADM_s100
                                 + 1.37162 w_k1_AXP_s100
                                 + 1.40252 w_k1_DIS_s100
                                 + 1.17273 w_k1_FDX_s100
                                 + 1.66794 w_k1_IBM_s100
                                 + 1.26618 w_k1_INTC_s100
                                 + 1.56404 w_k1_KO_s100 + 2.06863 w_k1_MAT_s100
                                 + 1.23259 w_k1_MSFT_s100
                                 + 1.25692 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.1871 w_k1_ADM_s100
                                 + 1.28905 w_k1_AXP_s100
                                 + 1.82963 w_k1_DIS_s100
                                 + 1.33836 w_k1_FDX_s100
                                 + 1.17985 w_k1_IBM_s100
                                 + 1.28118 w_k1_INTC_s100
                                 + 1.51542 w_k1_KO_s100 + 1.22578 w_k1_MAT_s100
                                 + 1.45965 w_k1_MSFT_s100
                                 + 1.19661 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.14204 w_k1_ADM_s100
                                 + 0.970657 w_k1_AXP_s100
                                 + 0.941296 w_k1_DIS_s100
                                 + 0.923313 w_k1_FDX_s100
                                 + 1.08086 w_k1_IBM_s100
                                 + 1.06785 w_k1_INTC_s100
                                 + 1.13479 w_k1_KO_s100 + 1.10818 w_k1_MAT_s100
                                 + 1.1057 w_k1_MSFT_s100 + 1.02982 w_k1_PG_s100
                                 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.23388 w_k1_ADM_s100
                                 + 1.17295 w_k1_AXP_s100 + 1.2517 w_k1_DIS_s100
                                 + 1.20163 w_k1_FDX_s100
                                 + 1.15816 w_k1_IBM_s100
                                 + 1.11688 w_k1_INTC_s100
                                 + 1.23186 w_k1_KO_s100 + 1.47813 w_k1_MAT_s100
                                 + 1.2984 w_k1_MSFT_s100 + 1.27636 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.35193 w_k1_ADM_s100
                                 + 1.28316 w_k1_AXP_s100
                                 + 1.28537 w_k1_DIS_s100
                                 + 1.20584 w_k1_FDX_s100
                                 + 0.836956 w_k1_IBM_s100
                                 + 1.16137 w_k1_INTC_s100
                                 + 1.05005 w_k1_KO_s100
                                 + 0.801443 w_k1_MAT_s100
                                 + 1.22866 w_k1_MSFT_s100
                                 + 1.10589 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.16476 w_k1_ADM_s100
                                 + 1.07011 w_k1_AXP_s100
                                 + 0.857778 w_k1_DIS_s100
                                 + 1.10558 w_k1_FDX_s100
                                 + 0.637945 w_k1_IBM_s100
                                 + 1.15672 w_k1_INTC_s100
                                 + 0.783776 w_k1_KO_s100
                                 + 0.649066 w_k1_MAT_s100
                                 + 1.07931 w_k1_MSFT_s100
                                 + 1.03174 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.671244 w_k1_ADM_s101
                                 + 0.686247 w_k1_AXP_s101
                                 + 0.548443 w_k1_DIS_s101
                                 + 0.658082 w_k1_FDX_s101
                                 + 0.786798 w_k1_IBM_s101
                                 + 0.936433 w_k1_INTC_s101
                                 + 0.816447 w_k1_KO_s101
                                 + 1.01304 w_k1_MAT_s101
                                 + 0.819881 w_k1_MSFT_s101
                                 + 0.77626 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.01245 w_k1_ADM_s101
                                 + 1.03752 w_k1_AXP_s101
                                 + 1.13858 w_k1_DIS_s101
                                 + 1.12522 w_k1_FDX_s101
                                 + 0.839478 w_k1_IBM_s101
                                 + 1.22918 w_k1_INTC_s101
                                 + 1.10137 w_k1_KO_s101 + 1.22785 w_k1_MAT_s101
                                 + 1.09877 w_k1_MSFT_s101
                                 + 1.13263 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.40457 w_k1_ADM_s101
                                 + 1.27721 w_k1_AXP_s101
                                 + 1.36183 w_k1_DIS_s101
                                 + 1.43071 w_k1_FDX_s101
                                 + 0.852605 w_k1_IBM_s101
                                 + 1.57887 w_k1_INTC_s101
                                 + 1.21032 w_k1_KO_s101
                                 + 0.923568 w_k1_MAT_s101
                                 + 1.38582 w_k1_MSFT_s101
                                 + 1.20939 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.90159 w_k1_ADM_s101
                                 + 0.89892 w_k1_AXP_s101
                                 + 0.758161 w_k1_DIS_s101
                                 + 0.830867 w_k1_FDX_s101
                                 + 1.21782 w_k1_IBM_s101
                                 + 1.07236 w_k1_INTC_s101
                                 + 1.04053 w_k1_KO_s101
                                 + 0.989699 w_k1_MAT_s101
                                 + 0.842702 w_k1_MSFT_s101
                                 + 0.874531 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.07998 w_k1_ADM_s101
                                 + 1.23355 w_k1_AXP_s101
                                 + 1.48822 w_k1_DIS_s101
                                 + 1.19931 w_k1_FDX_s101
                                 + 1.33673 w_k1_IBM_s101
                                 + 1.38042 w_k1_INTC_s101
                                 + 1.45084 w_k1_KO_s101 + 1.77701 w_k1_MAT_s101
                                 + 1.37433 w_k1_MSFT_s101
                                 + 1.36952 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.902577 w_k1_ADM_s101
                                 + 0.87453 w_k1_AXP_s101
                                 + 0.819611 w_k1_DIS_s101
                                 + 0.927225 w_k1_FDX_s101
                                 + 0.978218 w_k1_IBM_s101
                                 + 1.01116 w_k1_INTC_s101
                                 + 0.957683 w_k1_KO_s101
                                 + 1.03064 w_k1_MAT_s101
                                 + 0.924749 w_k1_MSFT_s101
                                 + 1.01588 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.982934 w_k1_ADM_s101
                                 + 1.29482 w_k1_AXP_s101
                                 + 1.06596 w_k1_DIS_s101
                                 + 1.04711 w_k1_FDX_s101
                                 + 1.48978 w_k1_IBM_s101
                                 + 1.33371 w_k1_INTC_s101
                                 + 1.13018 w_k1_KO_s101 + 1.55151 w_k1_MAT_s101
                                 + 1.05614 w_k1_MSFT_s101 + 1.1603 w_k1_PG_s101
                                 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.969929 w_k1_ADM_s101
                                 + 0.736717 w_k1_AXP_s101
                                 + 0.81674 w_k1_DIS_s101
                                 + 0.928692 w_k1_FDX_s101
                                 + 0.720888 w_k1_IBM_s101
                                 + 0.950812 w_k1_INTC_s101
                                 + 0.77112 w_k1_KO_s101
                                 + 0.549635 w_k1_MAT_s101
                                 + 0.964486 w_k1_MSFT_s101
                                 + 0.89717 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.32659 w_k1_ADM_s102
                                 + 1.03245 w_k1_AXP_s102
                                 + 1.29032 w_k1_DIS_s102
                                 + 1.25484 w_k1_FDX_s102
                                 + 0.953826 w_k1_IBM_s102
                                 + 1.09147 w_k1_INTC_s102
                                 + 1.05917 w_k1_KO_s102 + 1.16156 w_k1_MAT_s102
                                 + 1.12471 w_k1_MSFT_s102
                                 + 1.11443 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.0756 w_k1_ADM_s102
                                 + 0.988835 w_k1_AXP_s102
                                 + 1.0313 w_k1_DIS_s102 + 0.98276 w_k1_FDX_s102
                                 + 1.06745 w_k1_IBM_s102
                                 + 0.935541 w_k1_INTC_s102
                                 + 0.851389 w_k1_KO_s102
                                 + 0.90679 w_k1_MAT_s102
                                 + 0.893675 w_k1_MSFT_s102
                                 + 1.00475 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.26996 w_k1_ADM_s102
                                 + 0.869174 w_k1_AXP_s102
                                 + 0.893337 w_k1_DIS_s102
                                 + 0.988966 w_k1_FDX_s102
                                 + 0.771535 w_k1_IBM_s102
                                 + 0.882225 w_k1_INTC_s102
                                 + 0.794968 w_k1_KO_s102
                                 + 0.732735 w_k1_MAT_s102
                                 + 0.840347 w_k1_MSFT_s102
                                 + 0.928702 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.30298 w_k1_ADM_s102
                                 + 1.56369 w_k1_AXP_s102
                                 + 1.51602 w_k1_DIS_s102
                                 + 1.48643 w_k1_FDX_s102
                                 + 1.05551 w_k1_IBM_s102
                                 + 1.38151 w_k1_INTC_s102
                                 + 1.14571 w_k1_KO_s102 + 1.211 w_k1_MAT_s102
                                 + 1.20973 w_k1_MSFT_s102
                                 + 1.27124 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.32001 w_k1_ADM_s102
                                 + 1.17209 w_k1_AXP_s102
                                 + 1.53679 w_k1_DIS_s102
                                 + 1.26341 w_k1_FDX_s102
                                 + 1.19886 w_k1_IBM_s102
                                 + 1.16504 w_k1_INTC_s102
                                 + 1.30166 w_k1_KO_s102 + 1.26486 w_k1_MAT_s102
                                 + 1.33201 w_k1_MSFT_s102
                                 + 1.33057 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.1686 w_k1_ADM_s102
                                 + 1.197 w_k1_AXP_s102 + 1.09799 w_k1_DIS_s102
                                 + 0.982855 w_k1_FDX_s102
                                 + 1.59927 w_k1_IBM_s102
                                 + 1.1131 w_k1_INTC_s102 + 1.14145 w_k1_KO_s102
                                 + 1.69599 w_k1_MAT_s102
                                 + 0.965234 w_k1_MSFT_s102
                                 + 1.02794 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.04243 w_k1_ADM_s102
                                 + 0.826116 w_k1_AXP_s102
                                 + 0.937091 w_k1_DIS_s102
                                 + 1.00378 w_k1_FDX_s102
                                 + 0.567265 w_k1_IBM_s102
                                 + 0.875712 w_k1_INTC_s102
                                 + 0.684095 w_k1_KO_s102
                                 + 0.634871 w_k1_MAT_s102
                                 + 0.942314 w_k1_MSFT_s102
                                 + 0.846297 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.815737 w_k1_ADM_s102
                                 + 0.709055 w_k1_AXP_s102
                                 + 0.622956 w_k1_DIS_s102
                                 + 0.740001 w_k1_FDX_s102
                                 + 1.01762 w_k1_IBM_s102
                                 + 0.876609 w_k1_INTC_s102
                                 + 0.855233 w_k1_KO_s102
                                 + 1.04378 w_k1_MAT_s102
                                 + 0.734267 w_k1_MSFT_s102
                                 + 0.806464 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.877864 w_k1_ADM_s103
                                 + 0.85306 w_k1_AXP_s103
                                 + 0.865401 w_k1_DIS_s103
                                 + 1.0002 w_k1_FDX_s103
                                 + 0.881302 w_k1_IBM_s103
                                 + 0.969019 w_k1_INTC_s103
                                 + 0.83071 w_k1_KO_s103
                                 + 0.768618 w_k1_MAT_s103
                                 + 1.04559 w_k1_MSFT_s103 + 1.0893 w_k1_PG_s103
                                 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.677161 w_k1_ADM_s103
                                 + 0.701281 w_k1_AXP_s103
                                 + 0.810658 w_k1_DIS_s103
                                 + 0.831649 w_k1_FDX_s103
                                 + 1.05123 w_k1_IBM_s103
                                 + 0.879739 w_k1_INTC_s103
                                 + 0.878601 w_k1_KO_s103
                                 + 0.752096 w_k1_MAT_s103
                                 + 0.810057 w_k1_MSFT_s103
                                 + 0.958074 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.88533 w_k1_ADM_s103
                                 + 0.794373 w_k1_AXP_s103
                                 + 0.780931 w_k1_DIS_s103
                                 + 1.05941 w_k1_FDX_s103
                                 + 0.603424 w_k1_IBM_s103
                                 + 0.806449 w_k1_INTC_s103
                                 + 0.705565 w_k1_KO_s103
                                 + 0.599934 w_k1_MAT_s103
                                 + 0.890702 w_k1_MSFT_s103
                                 + 1.0211 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.821162 w_k1_ADM_s103
                                 + 0.959498 w_k1_AXP_s103
                                 + 0.771077 w_k1_DIS_s103
                                 + 0.814698 w_k1_FDX_s103
                                 + 1.30415 w_k1_IBM_s103
                                 + 0.78105 w_k1_INTC_s103
                                 + 0.883726 w_k1_KO_s103
                                 + 1.44654 w_k1_MAT_s103
                                 + 0.776687 w_k1_MSFT_s103
                                 + 1.01765 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.813154 w_k1_ADM_s103
                                 + 0.899988 w_k1_AXP_s103
                                 + 1.3853 w_k1_DIS_s103 + 1.13399 w_k1_FDX_s103
                                 + 1.21316 w_k1_IBM_s103
                                 + 0.967838 w_k1_INTC_s103
                                 + 1.21099 w_k1_KO_s103
                                 + 0.759915 w_k1_MAT_s103
                                 + 1.17621 w_k1_MSFT_s103
                                 + 1.09901 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.01698 w_k1_ADM_s103
                                 + 1.15776 w_k1_AXP_s103
                                 + 1.50896 w_k1_DIS_s103
                                 + 1.36453 w_k1_FDX_s103
                                 + 0.95596 w_k1_IBM_s103
                                 + 1.07615 w_k1_INTC_s103
                                 + 1.17421 w_k1_KO_s103 + 1.04152 w_k1_MAT_s103
                                 + 1.17639 w_k1_MSFT_s103
                                 + 1.26187 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.15945 w_k1_ADM_s103
                                 + 1.18043 w_k1_AXP_s103
                                 + 1.06596 w_k1_DIS_s103
                                 + 1.16315 w_k1_FDX_s103
                                 + 0.933334 w_k1_IBM_s103
                                 + 0.865702 w_k1_INTC_s103
                                 + 0.9275 w_k1_KO_s103 + 1.08604 w_k1_MAT_s103
                                 + 1.06617 w_k1_MSFT_s103
                                 + 1.14347 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.02787 w_k1_ADM_s103
                                 + 1.25452 w_k1_AXP_s103
                                 + 1.30115 w_k1_DIS_s103 + 1.1704 w_k1_FDX_s103
                                 + 1.55664 w_k1_IBM_s103
                                 + 1.17191 w_k1_INTC_s103
                                 + 1.20401 w_k1_KO_s103 + 1.30183 w_k1_MAT_s103
                                 + 1.03905 w_k1_MSFT_s103
                                 + 1.16392 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.30552 w_k1_ADM_s104
                                 + 1.05545 w_k1_AXP_s104
                                 + 1.39729 w_k1_DIS_s104
                                 + 1.23867 w_k1_FDX_s104
                                 + 1.10936 w_k1_IBM_s104
                                 + 1.21217 w_k1_INTC_s104 + 1.1903 w_k1_KO_s104
                                 + 1.18103 w_k1_MAT_s104
                                 + 1.33921 w_k1_MSFT_s104
                                 + 1.16038 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.02055 w_k1_ADM_s104
                                 + 0.777761 w_k1_AXP_s104
                                 + 0.664821 w_k1_DIS_s104
                                 + 0.797238 w_k1_FDX_s104
                                 + 0.863061 w_k1_IBM_s104
                                 + 0.874753 w_k1_INTC_s104
                                 + 0.662734 w_k1_KO_s104
                                 + 0.777615 w_k1_MAT_s104
                                 + 0.817433 w_k1_MSFT_s104
                                 + 0.847242 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.30704 w_k1_ADM_s104
                                 + 1.21203 w_k1_AXP_s104
                                 + 0.739375 w_k1_DIS_s104
                                 + 1.10249 w_k1_FDX_s104
                                 + 0.757009 w_k1_IBM_s104
                                 + 1.13848 w_k1_INTC_s104
                                 + 0.711554 w_k1_KO_s104
                                 + 1.19164 w_k1_MAT_s104
                                 + 1.08722 w_k1_MSFT_s104 + 1.0354 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.15722 w_k1_ADM_s104
                                 + 1.30154 w_k1_AXP_s104
                                 + 1.29191 w_k1_DIS_s104
                                 + 1.25824 w_k1_FDX_s104
                                 + 1.49911 w_k1_IBM_s104
                                 + 1.25049 w_k1_INTC_s104
                                 + 1.29259 w_k1_KO_s104 + 2.04501 w_k1_MAT_s104
                                 + 1.3503 w_k1_MSFT_s104 + 1.29721 w_k1_PG_s104
                                 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.690102 w_k1_ADM_s104
                                 + 0.586044 w_k1_AXP_s104
                                 + 0.518429 w_k1_DIS_s104
                                 + 0.698983 w_k1_FDX_s104
                                 + 0.584734 w_k1_IBM_s104
                                 + 0.727083 w_k1_INTC_s104
                                 + 0.578611 w_k1_KO_s104
                                 + 0.645315 w_k1_MAT_s104
                                 + 0.704081 w_k1_MSFT_s104
                                 + 0.742438 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.29285 w_k1_ADM_s104
                                 + 0.951635 w_k1_AXP_s104
                                 + 0.925643 w_k1_DIS_s104
                                 + 0.947187 w_k1_FDX_s104
                                 + 1.00951 w_k1_IBM_s104
                                 + 0.97253 w_k1_INTC_s104
                                 + 0.844456 w_k1_KO_s104
                                 + 0.992983 w_k1_MAT_s104
                                 + 1.17395 w_k1_MSFT_s104
                                 + 1.04559 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.08821 w_k1_ADM_s104
                                 + 0.969949 w_k1_AXP_s104
                                 + 1.2278 w_k1_DIS_s104 + 1.12451 w_k1_FDX_s104
                                 + 0.746809 w_k1_IBM_s104
                                 + 1.13853 w_k1_INTC_s104
                                 + 0.889452 w_k1_KO_s104
                                 + 0.962892 w_k1_MAT_s104
                                 + 1.19768 w_k1_MSFT_s104
                                 + 1.00476 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.977805 w_k1_ADM_s104
                                 + 0.930187 w_k1_AXP_s104
                                 + 0.730321 w_k1_DIS_s104
                                 + 0.793953 w_k1_FDX_s104
                                 + 1.21463 w_k1_IBM_s104
                                 + 0.782718 w_k1_INTC_s104
                                 + 0.863714 w_k1_KO_s104
                                 + 1.39924 w_k1_MAT_s104
                                 + 0.828028 w_k1_MSFT_s104
                                 + 0.867177 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.992548 w_k1_ADM_s105
                                 + 0.77543 w_k1_AXP_s105
                                 + 0.733792 w_k1_DIS_s105
                                 + 0.814298 w_k1_FDX_s105
                                 + 0.764516 w_k1_IBM_s105
                                 + 0.735551 w_k1_INTC_s105
                                 + 0.792512 w_k1_KO_s105
                                 + 0.755806 w_k1_MAT_s105
                                 + 0.957981 w_k1_MSFT_s105
                                 + 0.925721 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.955239 w_k1_ADM_s105
                                 + 0.90926 w_k1_AXP_s105
                                 + 0.751541 w_k1_DIS_s105
                                 + 0.880069 w_k1_FDX_s105
                                 + 1.01227 w_k1_IBM_s105
                                 + 1.07237 w_k1_INTC_s105
                                 + 0.861861 w_k1_KO_s105
                                 + 0.902292 w_k1_MAT_s105
                                 + 0.870504 w_k1_MSFT_s105
                                 + 0.82737 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.29453 w_k1_ADM_s105
                                 + 1.35445 w_k1_AXP_s105
                                 + 1.14401 w_k1_DIS_s105
                                 + 1.03788 w_k1_FDX_s105
                                 + 1.80263 w_k1_IBM_s105
                                 + 1.20997 w_k1_INTC_s105
                                 + 1.33732 w_k1_KO_s105 + 1.91636 w_k1_MAT_s105
                                 + 1.09751 w_k1_MSFT_s105 + 1.1011 w_k1_PG_s105
                                 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.99796 w_k1_ADM_s105
                                 + 0.670603 w_k1_AXP_s105
                                 + 0.894473 w_k1_DIS_s105
                                 + 0.761536 w_k1_FDX_s105
                                 + 1.04033 w_k1_IBM_s105
                                 + 0.926675 w_k1_INTC_s105
                                 + 0.895497 w_k1_KO_s105
                                 + 0.658148 w_k1_MAT_s105
                                 + 0.935651 w_k1_MSFT_s105
                                 + 0.803445 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.34398 w_k1_ADM_s105
                                 + 1.41189 w_k1_AXP_s105
                                 + 1.65347 w_k1_DIS_s105
                                 + 1.28391 w_k1_FDX_s105 + 1.2308 w_k1_IBM_s105
                                 + 1.10645 w_k1_INTC_s105 + 1.3495 w_k1_KO_s105
                                 + 1.66534 w_k1_MAT_s105
                                 + 1.1546 w_k1_MSFT_s105 + 1.10748 w_k1_PG_s105
                                 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.902183 w_k1_ADM_s105
                                 + 0.811802 w_k1_AXP_s105
                                 + 0.978174 w_k1_DIS_s105
                                 + 0.850094 w_k1_FDX_s105
                                 + 0.944807 w_k1_IBM_s105
                                 + 1.01701 w_k1_INTC_s105
                                 + 0.956118 w_k1_KO_s105
                                 + 0.818776 w_k1_MAT_s105
                                 + 0.993228 w_k1_MSFT_s105
                                 + 0.995787 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.10407 w_k1_ADM_s105
                                 + 1.22276 w_k1_AXP_s105 + 1.015 w_k1_DIS_s105
                                 + 1.1274 w_k1_FDX_s105
                                 + 0.748753 w_k1_IBM_s105
                                 + 0.973688 w_k1_INTC_s105
                                 + 0.895774 w_k1_KO_s105
                                 + 0.75876 w_k1_MAT_s105
                                 + 1.00786 w_k1_MSFT_s105
                                 + 0.823202 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.1369 w_k1_ADM_s105
                                 + 1.15712 w_k1_AXP_s105
                                 + 0.958182 w_k1_DIS_s105
                                 + 0.89598 w_k1_FDX_s105
                                 + 1.33742 w_k1_IBM_s105
                                 + 1.15286 w_k1_INTC_s105
                                 + 1.12388 w_k1_KO_s105 + 1.35552 w_k1_MAT_s105
                                 + 1.12621 w_k1_MSFT_s105
                                 + 1.02441 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.00367 w_k1_ADM_s106
                                 + 1.11592 w_k1_AXP_s106
                                 + 1.27248 w_k1_DIS_s106
                                 + 1.07607 w_k1_FDX_s106
                                 + 1.17401 w_k1_IBM_s106
                                 + 1.18267 w_k1_INTC_s106
                                 + 1.33531 w_k1_KO_s106 + 1.39038 w_k1_MAT_s106
                                 + 1.31368 w_k1_MSFT_s106
                                 + 1.01124 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.33888 w_k1_ADM_s106
                                 + 1.08079 w_k1_AXP_s106
                                 + 1.25692 w_k1_DIS_s106
                                 + 1.28956 w_k1_FDX_s106
                                 + 0.78121 w_k1_IBM_s106
                                 + 1.08024 w_k1_INTC_s106
                                 + 1.03721 w_k1_KO_s106
                                 + 0.897451 w_k1_MAT_s106
                                 + 1.23905 w_k1_MSFT_s106
                                 + 0.923082 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.76294 w_k1_ADM_s106
                                 + 0.624939 w_k1_AXP_s106
                                 + 0.703636 w_k1_DIS_s106
                                 + 0.740158 w_k1_FDX_s106
                                 + 0.762487 w_k1_IBM_s106
                                 + 0.877064 w_k1_INTC_s106
                                 + 0.803774 w_k1_KO_s106
                                 + 0.604706 w_k1_MAT_s106
                                 + 0.895679 w_k1_MSFT_s106
                                 + 0.820464 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.880888 w_k1_ADM_s106
                                 + 0.660523 w_k1_AXP_s106
                                 + 0.941211 w_k1_DIS_s106
                                 + 0.875095 w_k1_FDX_s106
                                 + 0.929931 w_k1_IBM_s106
                                 + 0.987955 w_k1_INTC_s106
                                 + 1.08589 w_k1_KO_s106
                                 + 0.644435 w_k1_MAT_s106
                                 + 1.04451 w_k1_MSFT_s106
                                 + 0.878745 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.976561 w_k1_ADM_s106
                                 + 1.12173 w_k1_AXP_s106 + 1.2809 w_k1_DIS_s106
                                 + 0.995642 w_k1_FDX_s106
                                 + 1.57766 w_k1_IBM_s106
                                 + 1.27062 w_k1_INTC_s106 + 1.5481 w_k1_KO_s106
                                 + 1.32649 w_k1_MAT_s106
                                 + 1.16152 w_k1_MSFT_s106
                                 + 1.08797 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.0176 w_k1_ADM_s106
                                 + 0.998776 w_k1_AXP_s106
                                 + 1.01943 w_k1_DIS_s106
                                 + 0.99926 w_k1_FDX_s106
                                 + 0.911648 w_k1_IBM_s106
                                 + 1.08013 w_k1_INTC_s106
                                 + 1.01511 w_k1_KO_s106
                                 + 0.971758 w_k1_MAT_s106
                                 + 1.05024 w_k1_MSFT_s106
                                 + 0.984077 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.26115 w_k1_ADM_s106
                                 + 1.39523 w_k1_AXP_s106
                                 + 1.54163 w_k1_DIS_s106
                                 + 1.32947 w_k1_FDX_s106
                                 + 1.25111 w_k1_IBM_s106
                                 + 1.32048 w_k1_INTC_s106 + 1.5295 w_k1_KO_s106
                                 + 1.46787 w_k1_MAT_s106
                                 + 1.48878 w_k1_MSFT_s106
                                 + 1.18724 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.08562 w_k1_ADM_s106
                                 + 0.87424 w_k1_AXP_s106
                                 + 1.46907 w_k1_DIS_s106
                                 + 1.05756 w_k1_FDX_s106
                                 + 1.08835 w_k1_IBM_s106
                                 + 1.17205 w_k1_INTC_s106 + 1.4338 w_k1_KO_s106
                                 + 1.07007 w_k1_MAT_s106
                                 + 1.3191 w_k1_MSFT_s106 + 1.13922 w_k1_PG_s106
                                 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.894712 w_k1_ADM_s107
                                 + 0.897236 w_k1_AXP_s107
                                 + 0.909592 w_k1_DIS_s107
                                 + 0.896486 w_k1_FDX_s107
                                 + 1.25312 w_k1_IBM_s107
                                 + 0.968593 w_k1_INTC_s107
                                 + 1.05333 w_k1_KO_s107
                                 + 0.918964 w_k1_MAT_s107
                                 + 0.879752 w_k1_MSFT_s107
                                 + 0.926028 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.42355 w_k1_ADM_s107
                                 + 1.5204 w_k1_AXP_s107 + 1.36732 w_k1_DIS_s107
                                 + 1.39539 w_k1_FDX_s107
                                 + 0.870155 w_k1_IBM_s107
                                 + 0.997079 w_k1_INTC_s107
                                 + 1.12881 w_k1_KO_s107
                                 + 0.787314 w_k1_MAT_s107
                                 + 1.21291 w_k1_MSFT_s107
                                 + 1.14711 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.924028 w_k1_ADM_s107
                                 + 0.787788 w_k1_AXP_s107
                                 + 0.693338 w_k1_DIS_s107
                                 + 0.891591 w_k1_FDX_s107
                                 + 0.603954 w_k1_IBM_s107
                                 + 0.82446 w_k1_INTC_s107
                                 + 0.760538 w_k1_KO_s107
                                 + 0.565732 w_k1_MAT_s107
                                 + 0.888022 w_k1_MSFT_s107
                                 + 0.834273 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.47148 w_k1_ADM_s107
                                 + 1.05688 w_k1_AXP_s107
                                 + 0.894979 w_k1_DIS_s107
                                 + 1.06407 w_k1_FDX_s107
                                 + 0.798753 w_k1_IBM_s107
                                 + 0.946759 w_k1_INTC_s107
                                 + 0.874851 w_k1_KO_s107
                                 + 0.49566 w_k1_MAT_s107
                                 + 0.945485 w_k1_MSFT_s107
                                 + 0.884648 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.17519 w_k1_ADM_s107
                                 + 1.23031 w_k1_AXP_s107
                                 + 1.23271 w_k1_DIS_s107
                                 + 0.981918 w_k1_FDX_s107
                                 + 1.63514 w_k1_IBM_s107
                                 + 0.903513 w_k1_INTC_s107
                                 + 1.36559 w_k1_KO_s107 + 1.67857 w_k1_MAT_s107
                                 + 1.06042 w_k1_MSFT_s107
                                 + 1.23666 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.89437 w_k1_ADM_s107
                                 + 0.957312 w_k1_AXP_s107
                                 + 1.08052 w_k1_DIS_s107 + 1.0154 w_k1_FDX_s107
                                 + 0.818638 w_k1_IBM_s107
                                 + 0.959113 w_k1_INTC_s107
                                 + 0.890531 w_k1_KO_s107
                                 + 0.832605 w_k1_MAT_s107
                                 + 1.01252 w_k1_MSFT_s107
                                 + 0.963347 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.971036 w_k1_ADM_s107
                                 + 0.836025 w_k1_AXP_s107
                                 + 0.771961 w_k1_DIS_s107
                                 + 0.780726 w_k1_FDX_s107
                                 + 0.898386 w_k1_IBM_s107
                                 + 0.939735 w_k1_INTC_s107
                                 + 0.873202 w_k1_KO_s107
                                 + 0.775085 w_k1_MAT_s107
                                 + 0.893961 w_k1_MSFT_s107
                                 + 0.804256 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.07787 w_k1_ADM_s107
                                 + 1.20422 w_k1_AXP_s107 + 1.3582 w_k1_DIS_s107
                                 + 1.21082 w_k1_FDX_s107
                                 + 1.11586 w_k1_IBM_s107
                                 + 1.24275 w_k1_INTC_s107
                                 + 1.20954 w_k1_KO_s107
                                 + 0.987038 w_k1_MAT_s107
                                 + 1.156 w_k1_MSFT_s107 + 1.05622 w_k1_PG_s107
                                 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.00005 w_k1_ADM_s110
                                 + 0.809696 w_k1_AXP_s110
                                 + 0.785358 w_k1_DIS_s110
                                 + 0.800326 w_k1_FDX_s110
                                 + 1.09839 w_k1_IBM_s110
                                 + 0.861525 w_k1_INTC_s110
                                 + 0.920258 w_k1_KO_s110
                                 + 0.779466 w_k1_MAT_s110
                                 + 0.826781 w_k1_MSFT_s110
                                 + 0.92775 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.695857 w_k1_ADM_s110
                                 + 0.617933 w_k1_AXP_s110
                                 + 0.5763 w_k1_DIS_s110 + 0.77437 w_k1_FDX_s110
                                 + 0.695737 w_k1_IBM_s110
                                 + 0.739106 w_k1_INTC_s110
                                 + 0.641481 w_k1_KO_s110
                                 + 0.71494 w_k1_MAT_s110
                                 + 0.746441 w_k1_MSFT_s110
                                 + 0.687565 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.11368 w_k1_ADM_s110
                                 + 1.26104 w_k1_AXP_s110
                                 + 1.01399 w_k1_DIS_s110
                                 + 1.08634 w_k1_FDX_s110
                                 + 1.61942 w_k1_IBM_s110
                                 + 0.975671 w_k1_INTC_s110
                                 + 1.30378 w_k1_KO_s110 + 1.36261 w_k1_MAT_s110
                                 + 1.10658 w_k1_MSFT_s110
                                 + 1.02195 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.26364 w_k1_ADM_s110
                                 + 1.05587 w_k1_AXP_s110
                                 + 1.42347 w_k1_DIS_s110
                                 + 1.15797 w_k1_FDX_s110
                                 + 1.33571 w_k1_IBM_s110
                                 + 1.06251 w_k1_INTC_s110
                                 + 1.33599 w_k1_KO_s110
                                 + 0.929077 w_k1_MAT_s110
                                 + 1.22954 w_k1_MSFT_s110 + 1.0863 w_k1_PG_s110
                                 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.04827 w_k1_ADM_s110
                                 + 1.02853 w_k1_AXP_s110
                                 + 1.10848 w_k1_DIS_s110
                                 + 1.14099 w_k1_FDX_s110
                                 + 0.868723 w_k1_IBM_s110
                                 + 1.11569 w_k1_INTC_s110
                                 + 0.93984 w_k1_KO_s110
                                 + 0.683513 w_k1_MAT_s110
                                 + 1.17652 w_k1_MSFT_s110
                                 + 0.949022 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.813313 w_k1_ADM_s110
                                 + 0.662957 w_k1_AXP_s110
                                 + 0.651663 w_k1_DIS_s110
                                 + 0.847896 w_k1_FDX_s110
                                 + 0.727538 w_k1_IBM_s110
                                 + 0.83966 w_k1_INTC_s110
                                 + 0.730859 w_k1_KO_s110
                                 + 0.581204 w_k1_MAT_s110
                                 + 0.7765 w_k1_MSFT_s110
                                 + 0.820243 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.58034 w_k1_ADM_s110
                                 + 1.66483 w_k1_AXP_s110
                                 + 1.41879 w_k1_DIS_s110
                                 + 1.39826 w_k1_FDX_s110 + 1.2057 w_k1_IBM_s110
                                 + 1.41291 w_k1_INTC_s110
                                 + 1.16778 w_k1_KO_s110 + 1.16338 w_k1_MAT_s110
                                 + 1.19163 w_k1_MSFT_s110
                                 + 1.37594 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.01458 w_k1_ADM_s110
                                 + 0.896299 w_k1_AXP_s110
                                 + 0.98957 w_k1_DIS_s110
                                 + 0.965963 w_k1_FDX_s110
                                 + 1.0182 w_k1_IBM_s110
                                 + 0.94045 w_k1_INTC_s110
                                 + 1.07284 w_k1_KO_s110
                                 + 0.793528 w_k1_MAT_s110
                                 + 0.915929 w_k1_MSFT_s110
                                 + 1.0082 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.0244 w_k1_ADM_s111
                                 + 1.05478 w_k1_AXP_s111
                                 + 1.27032 w_k1_DIS_s111
                                 + 1.20556 w_k1_FDX_s111
                                 + 1.06968 w_k1_IBM_s111
                                 + 1.28605 w_k1_INTC_s111
                                 + 1.05773 w_k1_KO_s111
                                 + 0.638211 w_k1_MAT_s111
                                 + 1.24703 w_k1_MSFT_s111
                                 + 1.03392 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.952025 w_k1_ADM_s111
                                 + 0.814637 w_k1_AXP_s111
                                 + 0.689066 w_k1_DIS_s111
                                 + 0.968896 w_k1_FDX_s111
                                 + 0.86476 w_k1_IBM_s111
                                 + 0.921641 w_k1_INTC_s111
                                 + 0.728492 w_k1_KO_s111
                                 + 0.586193 w_k1_MAT_s111
                                 + 0.779495 w_k1_MSFT_s111
                                 + 0.958456 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.13501 w_k1_ADM_s111
                                 + 1.10846 w_k1_AXP_s111
                                 + 1.00949 w_k1_DIS_s111
                                 + 0.987811 w_k1_FDX_s111
                                 + 1.16198 w_k1_IBM_s111
                                 + 0.967427 w_k1_INTC_s111
                                 + 1.17816 w_k1_KO_s111 + 1.2058 w_k1_MAT_s111
                                 + 0.941778 w_k1_MSFT_s111
                                 + 1.06095 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.810012 w_k1_ADM_s111
                                 + 0.797297 w_k1_AXP_s111
                                 + 0.860288 w_k1_DIS_s111
                                 + 0.775005 w_k1_FDX_s111
                                 + 1.09129 w_k1_IBM_s111
                                 + 0.759199 w_k1_INTC_s111
                                 + 0.999945 w_k1_KO_s111
                                 + 0.537313 w_k1_MAT_s111
                                 + 0.902405 w_k1_MSFT_s111
                                 + 0.961865 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.97072 w_k1_ADM_s111
                                 + 0.940364 w_k1_AXP_s111
                                 + 0.853203 w_k1_DIS_s111
                                 + 0.94016 w_k1_FDX_s111
                                 + 1.28429 w_k1_IBM_s111
                                 + 0.910939 w_k1_INTC_s111
                                 + 1.07568 w_k1_KO_s111
                                 + 0.959895 w_k1_MAT_s111
                                 + 0.937021 w_k1_MSFT_s111
                                 + 0.924437 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.42666 w_k1_ADM_s111
                                 + 1.72019 w_k1_AXP_s111
                                 + 1.73031 w_k1_DIS_s111 + 1.4243 w_k1_FDX_s111
                                 + 1.46992 w_k1_IBM_s111
                                 + 1.29424 w_k1_INTC_s111
                                 + 1.31551 w_k1_KO_s111 + 1.59508 w_k1_MAT_s111
                                 + 1.26164 w_k1_MSFT_s111
                                 + 1.13038 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.791179 w_k1_ADM_s111
                                 + 0.816994 w_k1_AXP_s111
                                 + 0.929821 w_k1_DIS_s111
                                 + 0.951855 w_k1_FDX_s111
                                 + 0.739387 w_k1_IBM_s111
                                 + 0.933935 w_k1_INTC_s111
                                 + 0.957832 w_k1_KO_s111
                                 + 0.828034 w_k1_MAT_s111
                                 + 1.06343 w_k1_MSFT_s111
                                 + 1.01249 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.14068 w_k1_ADM_s111
                                 + 1.39475 w_k1_AXP_s111
                                 + 1.43689 w_k1_DIS_s111
                                 + 1.04925 w_k1_FDX_s111 + 1.6874 w_k1_IBM_s111
                                 + 1.26735 w_k1_INTC_s111 + 1.4846 w_k1_KO_s111
                                 + 0.896816 w_k1_MAT_s111
                                 + 1.02372 w_k1_MSFT_s111
                                 + 1.21119 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.07477 w_k1_ADM_s112
                                 + 1.36985 w_k1_AXP_s112
                                 + 0.803796 w_k1_DIS_s112
                                 + 0.881817 w_k1_FDX_s112
                                 + 1.37779 w_k1_IBM_s112
                                 + 1.11553 w_k1_INTC_s112
                                 + 1.16099 w_k1_KO_s112 + 1.98243 w_k1_MAT_s112
                                 + 0.855303 w_k1_MSFT_s112
                                 + 1.05154 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.926623 w_k1_ADM_s112
                                 + 0.792433 w_k1_AXP_s112
                                 + 0.698043 w_k1_DIS_s112
                                 + 0.730087 w_k1_FDX_s112
                                 + 1.02485 w_k1_IBM_s112
                                 + 0.852057 w_k1_INTC_s112
                                 + 1.01203 w_k1_KO_s112 + 1.11687 w_k1_MAT_s112
                                 + 0.77598 w_k1_MSFT_s112
                                 + 0.994344 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.0487 w_k1_ADM_s112
                                 + 0.924061 w_k1_AXP_s112
                                 + 1.00337 w_k1_DIS_s112
                                 + 0.950287 w_k1_FDX_s112
                                 + 1.18766 w_k1_IBM_s112
                                 + 1.01785 w_k1_INTC_s112
                                 + 1.03042 w_k1_KO_s112
                                 + 0.855249 w_k1_MAT_s112
                                 + 0.85841 w_k1_MSFT_s112
                                 + 0.947205 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.176 w_k1_ADM_s112
                                 + 1.10812 w_k1_AXP_s112
                                 + 1.66112 w_k1_DIS_s112
                                 + 1.27206 w_k1_FDX_s112
                                 + 1.21428 w_k1_IBM_s112
                                 + 1.18418 w_k1_INTC_s112
                                 + 1.56011 w_k1_KO_s112 + 1.48844 w_k1_MAT_s112
                                 + 1.33225 w_k1_MSFT_s112
                                 + 1.10065 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.01164 w_k1_ADM_s112
                                 + 0.884227 w_k1_AXP_s112
                                 + 0.842611 w_k1_DIS_s112
                                 + 0.962246 w_k1_FDX_s112
                                 + 0.728826 w_k1_IBM_s112
                                 + 0.934499 w_k1_INTC_s112
                                 + 0.764634 w_k1_KO_s112
                                 + 0.670122 w_k1_MAT_s112
                                 + 0.827681 w_k1_MSFT_s112
                                 + 0.867956 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.04599 w_k1_ADM_s112
                                 + 1.35541 w_k1_AXP_s112
                                 + 1.00606 w_k1_DIS_s112
                                 + 1.17523 w_k1_FDX_s112
                                 + 0.882159 w_k1_IBM_s112
                                 + 1.16071 w_k1_INTC_s112
                                 + 1.00978 w_k1_KO_s112 + 1.24569 w_k1_MAT_s112
                                 + 0.939239 w_k1_MSFT_s112
                                 + 1.02879 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.42784 w_k1_ADM_s112
                                 + 1.89007 w_k1_AXP_s112
                                 + 1.36104 w_k1_DIS_s112 + 1.2541 w_k1_FDX_s112
                                 + 1.68771 w_k1_IBM_s112
                                 + 1.25847 w_k1_INTC_s112
                                 + 1.52098 w_k1_KO_s112 + 2.40343 w_k1_MAT_s112
                                 + 1.22223 w_k1_MSFT_s112
                                 + 1.43946 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.836 w_k1_ADM_s112
                                 + 0.959035 w_k1_AXP_s112
                                 + 0.706088 w_k1_DIS_s112
                                 + 0.828127 w_k1_FDX_s112
                                 + 0.747104 w_k1_IBM_s112
                                 + 0.90156 w_k1_INTC_s112
                                 + 0.794707 w_k1_KO_s112
                                 + 0.939639 w_k1_MAT_s112
                                 + 0.853445 w_k1_MSFT_s112
                                 + 0.910357 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.12804 w_k1_ADM_s113
                                 + 1.36481 w_k1_AXP_s113
                                 + 1.64035 w_k1_DIS_s113
                                 + 1.18305 w_k1_FDX_s113
                                 + 1.50443 w_k1_IBM_s113
                                 + 1.17341 w_k1_INTC_s113
                                 + 1.68985 w_k1_KO_s113 + 1.68753 w_k1_MAT_s113
                                 + 1.19448 w_k1_MSFT_s113
                                 + 1.13912 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.898648 w_k1_ADM_s113
                                 + 0.795132 w_k1_AXP_s113
                                 + 0.926753 w_k1_DIS_s113
                                 + 0.885075 w_k1_FDX_s113
                                 + 1.30041 w_k1_IBM_s113
                                 + 0.883351 w_k1_INTC_s113
                                 + 1.10519 w_k1_KO_s113 + 1.43109 w_k1_MAT_s113
                                 + 0.938748 w_k1_MSFT_s113
                                 + 1.0904 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.10785 w_k1_ADM_s113
                                 + 1.12647 w_k1_AXP_s113
                                 + 1.30475 w_k1_DIS_s113
                                 + 1.05893 w_k1_FDX_s113
                                 + 1.22666 w_k1_IBM_s113
                                 + 1.07657 w_k1_INTC_s113
                                 + 1.22059 w_k1_KO_s113 + 1.41024 w_k1_MAT_s113
                                 + 1.21236 w_k1_MSFT_s113
                                 + 1.08975 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.36341 w_k1_ADM_s113
                                 + 1.04676 w_k1_AXP_s113 + 1.1142 w_k1_DIS_s113
                                 + 1.04684 w_k1_FDX_s113
                                 + 0.982014 w_k1_IBM_s113
                                 + 0.918849 w_k1_INTC_s113
                                 + 0.951691 w_k1_KO_s113
                                 + 0.925239 w_k1_MAT_s113
                                 + 1.08822 w_k1_MSFT_s113
                                 + 1.04623 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.18249 w_k1_ADM_s113
                                 + 0.990467 w_k1_AXP_s113
                                 + 0.945931 w_k1_DIS_s113
                                 + 1.04275 w_k1_FDX_s113
                                 + 0.623589 w_k1_IBM_s113
                                 + 0.931375 w_k1_INTC_s113
                                 + 0.813599 w_k1_KO_s113
                                 + 0.785165 w_k1_MAT_s113
                                 + 1.03831 w_k1_MSFT_s113
                                 + 0.896667 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.36177 w_k1_ADM_s113
                                 + 1.01153 w_k1_AXP_s113
                                 + 1.48408 w_k1_DIS_s113
                                 + 1.29892 w_k1_FDX_s113
                                 + 0.948116 w_k1_IBM_s113
                                 + 1.15756 w_k1_INTC_s113
                                 + 1.10093 w_k1_KO_s113
                                 + 0.880014 w_k1_MAT_s113
                                 + 1.16599 w_k1_MSFT_s113
                                 + 1.05399 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.964857 w_k1_ADM_s113
                                 + 0.883311 w_k1_AXP_s113
                                 + 0.676966 w_k1_DIS_s113
                                 + 0.827806 w_k1_FDX_s113
                                 + 0.875601 w_k1_IBM_s113
                                 + 0.830843 w_k1_INTC_s113
                                 + 0.748714 w_k1_KO_s113
                                 + 0.854328 w_k1_MAT_s113
                                 + 0.755604 w_k1_MSFT_s113
                                 + 0.970777 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.899237 w_k1_ADM_s113
                                 + 0.631438 w_k1_AXP_s113
                                 + 0.791552 w_k1_DIS_s113
                                 + 0.854397 w_k1_FDX_s113
                                 + 0.773678 w_k1_IBM_s113
                                 + 0.828651 w_k1_INTC_s113
                                 + 0.842051 w_k1_KO_s113
                                 + 0.699552 w_k1_MAT_s113
                                 + 0.917943 w_k1_MSFT_s113
                                 + 0.908513 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.960095 w_k1_ADM_s114
                                 + 0.71092 w_k1_AXP_s114
                                 + 0.660745 w_k1_DIS_s114
                                 + 0.754078 w_k1_FDX_s114
                                 + 0.922967 w_k1_IBM_s114
                                 + 0.81351 w_k1_INTC_s114
                                 + 0.862439 w_k1_KO_s114
                                 + 1.16203 w_k1_MAT_s114
                                 + 0.814379 w_k1_MSFT_s114
                                 + 0.880085 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.944479 w_k1_ADM_s114
                                 + 1.18496 w_k1_AXP_s114
                                 + 1.20316 w_k1_DIS_s114
                                 + 0.88858 w_k1_FDX_s114
                                 + 1.57109 w_k1_IBM_s114
                                 + 0.896296 w_k1_INTC_s114
                                 + 1.35143 w_k1_KO_s114 + 1.52823 w_k1_MAT_s114
                                 + 0.99934 w_k1_MSFT_s114
                                 + 1.23464 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.10507 w_k1_ADM_s114
                                 + 1.45553 w_k1_AXP_s114
                                 + 1.30785 w_k1_DIS_s114
                                 + 1.15181 w_k1_FDX_s114
                                 + 1.49044 w_k1_IBM_s114
                                 + 1.33799 w_k1_INTC_s114
                                 + 1.33552 w_k1_KO_s114 + 1.04207 w_k1_MAT_s114
                                 + 1.09307 w_k1_MSFT_s114
                                 + 1.22162 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.869724 w_k1_ADM_s114
                                 + 1.15881 w_k1_AXP_s114
                                 + 1.12216 w_k1_DIS_s114 + 1.0912 w_k1_FDX_s114
                                 + 0.896297 w_k1_IBM_s114
                                 + 0.974415 w_k1_INTC_s114
                                 + 0.989039 w_k1_KO_s114
                                 + 0.98881 w_k1_MAT_s114
                                 + 1.12377 w_k1_MSFT_s114
                                 + 1.20987 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.98908 w_k1_ADM_s114
                                 + 0.899381 w_k1_AXP_s114
                                 + 1.02404 w_k1_DIS_s114
                                 + 0.871939 w_k1_FDX_s114
                                 + 1.07605 w_k1_IBM_s114
                                 + 0.916723 w_k1_INTC_s114 + 1.127 w_k1_KO_s114
                                 + 1.17212 w_k1_MAT_s114
                                 + 0.968826 w_k1_MSFT_s114
                                 + 0.987104 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.13162 w_k1_ADM_s114
                                 + 1.42552 w_k1_AXP_s114
                                 + 0.919414 w_k1_DIS_s114
                                 + 1.06251 w_k1_FDX_s114
                                 + 0.716732 w_k1_IBM_s114
                                 + 0.990516 w_k1_INTC_s114
                                 + 0.845988 w_k1_KO_s114
                                 + 0.853678 w_k1_MAT_s114
                                 + 1.04155 w_k1_MSFT_s114
                                 + 0.936757 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.29155 w_k1_ADM_s114
                                 + 1.21475 w_k1_AXP_s114
                                 + 1.52574 w_k1_DIS_s114
                                 + 1.21706 w_k1_FDX_s114
                                 + 0.951802 w_k1_IBM_s114
                                 + 1.10498 w_k1_INTC_s114
                                 + 1.20853 w_k1_KO_s114 + 1.06069 w_k1_MAT_s114
                                 + 1.3272 w_k1_MSFT_s114 + 1.18233 w_k1_PG_s114
                                 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.953325 w_k1_ADM_s114
                                 + 1.05016 w_k1_AXP_s114
                                 + 0.816132 w_k1_DIS_s114
                                 + 0.910094 w_k1_FDX_s114
                                 + 0.78325 w_k1_IBM_s114
                                 + 0.810651 w_k1_INTC_s114
                                 + 0.819381 w_k1_KO_s114
                                 + 0.971775 w_k1_MAT_s114
                                 + 0.929502 w_k1_MSFT_s114
                                 + 0.955106 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.791839 w_k1_ADM_s115
                                 + 0.910249 w_k1_AXP_s115
                                 + 0.974045 w_k1_DIS_s115
                                 + 0.833394 w_k1_FDX_s115
                                 + 1.38718 w_k1_IBM_s115
                                 + 0.937117 w_k1_INTC_s115
                                 + 0.899613 w_k1_KO_s115
                                 + 1.01998 w_k1_MAT_s115
                                 + 0.893868 w_k1_MSFT_s115
                                 + 1.01146 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.828494 w_k1_ADM_s115
                                 + 0.748989 w_k1_AXP_s115
                                 + 0.919132 w_k1_DIS_s115
                                 + 0.99063 w_k1_FDX_s115
                                 + 0.556151 w_k1_IBM_s115
                                 + 1.05674 w_k1_INTC_s115
                                 + 0.616834 w_k1_KO_s115
                                 + 0.696809 w_k1_MAT_s115
                                 + 0.985293 w_k1_MSFT_s115
                                 + 0.781979 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.06152 w_k1_ADM_s115
                                 + 1.2243 w_k1_AXP_s115 + 1.34381 w_k1_DIS_s115
                                 + 1.05799 w_k1_FDX_s115
                                 + 1.25066 w_k1_IBM_s115
                                 + 0.849797 w_k1_INTC_s115
                                 + 1.07199 w_k1_KO_s115 + 1.3705 w_k1_MAT_s115
                                 + 0.951793 w_k1_MSFT_s115
                                 + 1.1223 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.733209 w_k1_ADM_s115
                                 + 0.667858 w_k1_AXP_s115
                                 + 0.637197 w_k1_DIS_s115
                                 + 0.810442 w_k1_FDX_s115
                                 + 0.643471 w_k1_IBM_s115
                                 + 0.803908 w_k1_INTC_s115
                                 + 0.691255 w_k1_KO_s115
                                 + 0.93689 w_k1_MAT_s115
                                 + 0.834778 w_k1_MSFT_s115
                                 + 0.889811 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.17771 w_k1_ADM_s115
                                 + 1.13653 w_k1_AXP_s115
                                 + 1.00908 w_k1_DIS_s115
                                 + 1.05278 w_k1_FDX_s115
                                 + 0.912843 w_k1_IBM_s115
                                 + 0.890417 w_k1_INTC_s115
                                 + 0.835299 w_k1_KO_s115
                                 + 0.953858 w_k1_MAT_s115
                                 + 0.971864 w_k1_MSFT_s115
                                 + 1.05919 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.3625 w_k1_ADM_s115
                                 + 1.60587 w_k1_AXP_s115
                                 + 1.96714 w_k1_DIS_s115
                                 + 1.50604 w_k1_FDX_s115
                                 + 1.71521 w_k1_IBM_s115
                                 + 1.42264 w_k1_INTC_s115
                                 + 1.37395 w_k1_KO_s115 + 1.94239 w_k1_MAT_s115
                                 + 1.36379 w_k1_MSFT_s115
                                 + 1.19865 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.751608 w_k1_ADM_s115
                                 + 0.650313 w_k1_AXP_s115
                                 + 0.738154 w_k1_DIS_s115
                                 + 0.739673 w_k1_FDX_s115
                                 + 0.970775 w_k1_IBM_s115
                                 + 0.803676 w_k1_INTC_s115
                                 + 0.803348 w_k1_KO_s115
                                 + 1.02309 w_k1_MAT_s115
                                 + 0.759654 w_k1_MSFT_s115
                                 + 0.90787 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.31832 w_k1_ADM_s115
                                 + 1.50394 w_k1_AXP_s115
                                 + 1.43648 w_k1_DIS_s115
                                 + 1.34335 w_k1_FDX_s115
                                 + 1.20432 w_k1_IBM_s115
                                 + 1.4015 w_k1_INTC_s115 + 1.07208 w_k1_KO_s115
                                 + 1.0173 w_k1_MAT_s115
                                 + 1.21316 w_k1_MSFT_s115
                                 + 1.13884 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.00241 w_k1_ADM_s116
                                 + 1.09284 w_k1_AXP_s116
                                 + 1.26538 w_k1_DIS_s116
                                 + 1.25358 w_k1_FDX_s116
                                 + 0.879442 w_k1_IBM_s116
                                 + 1.34317 w_k1_INTC_s116
                                 + 1.31096 w_k1_KO_s116
                                 + 0.851146 w_k1_MAT_s116
                                 + 1.27988 w_k1_MSFT_s116
                                 + 1.15943 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.829327 w_k1_ADM_s116
                                 + 0.937159 w_k1_AXP_s116
                                 + 0.786923 w_k1_DIS_s116
                                 + 0.886367 w_k1_FDX_s116 + 1.278 w_k1_IBM_s116
                                 + 0.897147 w_k1_INTC_s116
                                 + 1.04023 w_k1_KO_s116 + 1.50865 w_k1_MAT_s116
                                 + 0.957083 w_k1_MSFT_s116
                                 + 0.873418 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.970264 w_k1_ADM_s116
                                 + 1.11144 w_k1_AXP_s116
                                 + 0.909605 w_k1_DIS_s116
                                 + 0.946975 w_k1_FDX_s116
                                 + 0.824393 w_k1_IBM_s116
                                 + 0.837816 w_k1_INTC_s116
                                 + 0.971656 w_k1_KO_s116
                                 + 0.81886 w_k1_MAT_s116
                                 + 0.828738 w_k1_MSFT_s116
                                 + 0.883073 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.07285 w_k1_ADM_s116
                                 + 1.1636 w_k1_AXP_s116 + 1.05563 w_k1_DIS_s116
                                 + 1.0908 w_k1_FDX_s116
                                 + 0.930026 w_k1_IBM_s116
                                 + 0.915468 w_k1_INTC_s116
                                 + 1.04976 w_k1_KO_s116
                                 + 0.879834 w_k1_MAT_s116
                                 + 0.995782 w_k1_MSFT_s116
                                 + 0.964165 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.19454 w_k1_ADM_s116
                                 + 1.34001 w_k1_AXP_s116 + 1.1233 w_k1_DIS_s116
                                 + 1.14966 w_k1_FDX_s116
                                 + 1.36248 w_k1_IBM_s116
                                 + 0.99592 w_k1_INTC_s116
                                 + 1.21982 w_k1_KO_s116 + 1.7796 w_k1_MAT_s116
                                 + 1.16398 w_k1_MSFT_s116 + 1.0276 w_k1_PG_s116
                                 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.0486 w_k1_ADM_s116
                                 + 1.47296 w_k1_AXP_s116
                                 + 1.52591 w_k1_DIS_s116
                                 + 1.26748 w_k1_FDX_s116
                                 + 1.36359 w_k1_IBM_s116
                                 + 1.35195 w_k1_INTC_s116
                                 + 1.55262 w_k1_KO_s116 + 1.61618 w_k1_MAT_s116
                                 + 1.34093 w_k1_MSFT_s116
                                 + 1.10758 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.894262 w_k1_ADM_s116
                                 + 0.973421 w_k1_AXP_s116
                                 + 0.971249 w_k1_DIS_s116
                                 + 0.948933 w_k1_FDX_s116
                                 + 1.00713 w_k1_IBM_s116
                                 + 1.01227 w_k1_INTC_s116
                                 + 1.21003 w_k1_KO_s116 + 1.14486 w_k1_MAT_s116
                                 + 1.14353 w_k1_MSFT_s116
                                 + 0.911383 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.770062 w_k1_ADM_s116
                                 + 0.760135 w_k1_AXP_s116
                                 + 0.637803 w_k1_DIS_s116
                                 + 0.821142 w_k1_FDX_s116
                                 + 0.70372 w_k1_IBM_s116
                                 + 0.859809 w_k1_INTC_s116
                                 + 0.740799 w_k1_KO_s116
                                 + 0.688303 w_k1_MAT_s116
                                 + 0.829809 w_k1_MSFT_s116
                                 + 0.765739 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.736707 w_k1_ADM_s117
                                 + 0.629436 w_k1_AXP_s117
                                 + 0.722481 w_k1_DIS_s117
                                 + 0.800211 w_k1_FDX_s117
                                 + 0.778328 w_k1_IBM_s117
                                 + 0.897346 w_k1_INTC_s117
                                 + 0.818008 w_k1_KO_s117
                                 + 0.728522 w_k1_MAT_s117
                                 + 0.880587 w_k1_MSFT_s117
                                 + 0.781092 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.03049 w_k1_ADM_s117
                                 + 0.836385 w_k1_AXP_s117
                                 + 0.939291 w_k1_DIS_s117
                                 + 0.943681 w_k1_FDX_s117
                                 + 0.699565 w_k1_IBM_s117
                                 + 1.01766 w_k1_INTC_s117
                                 + 0.802789 w_k1_KO_s117
                                 + 0.552682 w_k1_MAT_s117
                                 + 0.967242 w_k1_MSFT_s117
                                 + 0.877841 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.00383 w_k1_ADM_s117
                                 + 1.05959 w_k1_AXP_s117
                                 + 1.07681 w_k1_DIS_s117
                                 + 1.10629 w_k1_FDX_s117
                                 + 1.25648 w_k1_IBM_s117
                                 + 1.10977 w_k1_INTC_s117
                                 + 1.31953 w_k1_KO_s117 + 1.60102 w_k1_MAT_s117
                                 + 1.00124 w_k1_MSFT_s117
                                 + 1.09664 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.876725 w_k1_ADM_s117
                                 + 0.750553 w_k1_AXP_s117
                                 + 0.984368 w_k1_DIS_s117
                                 + 0.901734 w_k1_FDX_s117
                                 + 0.991585 w_k1_IBM_s117
                                 + 0.942399 w_k1_INTC_s117
                                 + 1.19429 w_k1_KO_s117 + 1.02154 w_k1_MAT_s117
                                 + 0.970331 w_k1_MSFT_s117
                                 + 0.904771 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.78072 w_k1_ADM_s117
                                 + 1.4392 w_k1_AXP_s117 + 1.73154 w_k1_DIS_s117
                                 + 1.41822 w_k1_FDX_s117
                                 + 1.21926 w_k1_IBM_s117
                                 + 1.42277 w_k1_INTC_s117
                                 + 1.55507 w_k1_KO_s117
                                 + 0.953379 w_k1_MAT_s117
                                 + 1.26624 w_k1_MSFT_s117
                                 + 1.29691 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.09971 w_k1_ADM_s117
                                 + 1.17415 w_k1_AXP_s117
                                 + 1.24697 w_k1_DIS_s117
                                 + 1.23756 w_k1_FDX_s117
                                 + 0.945666 w_k1_IBM_s117
                                 + 1.14115 w_k1_INTC_s117
                                 + 1.09032 w_k1_KO_s117 + 1.19084 w_k1_MAT_s117
                                 + 1.14053 w_k1_MSFT_s117
                                 + 1.10987 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.24661 w_k1_ADM_s117
                                 + 0.942112 w_k1_AXP_s117
                                 + 1.30666 w_k1_DIS_s117
                                 + 1.03165 w_k1_FDX_s117
                                 + 1.10688 w_k1_IBM_s117
                                 + 0.903303 w_k1_INTC_s117
                                 + 1.16021 w_k1_KO_s117 + 1.03098 w_k1_MAT_s117
                                 + 0.947911 w_k1_MSFT_s117
                                 + 1.02474 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.20256 w_k1_ADM_s117
                                 + 1.11545 w_k1_AXP_s117
                                 + 1.12257 w_k1_DIS_s117
                                 + 1.34451 w_k1_FDX_s117
                                 + 0.769793 w_k1_IBM_s117
                                 + 1.57836 w_k1_INTC_s117
                                 + 0.944242 w_k1_KO_s117
                                 + 0.451342 w_k1_MAT_s117
                                 + 1.25468 w_k1_MSFT_s117
                                 + 0.994816 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0151177749415478 w_k1_ADM_s000
                                 + 0.0342366095799048 w_k1_ADM_s001
                                 + 0.0421620030037972 w_k1_ADM_s002
                                 + 0.0221870614976269 w_k1_ADM_s003
                                 + 0.0119742169880232 w_k1_ADM_s004
                                 + 0.0245075775725227 w_k1_ADM_s005
                                 + 0.015627577990886 w_k1_ADM_s006
                                 + 0.0120775312258342 w_k1_ADM_s007
                                 + 0.0221191610035388 w_k1_ADM_s010
                                 + 0.0368856918500419 w_k1_ADM_s011
                                 + 0.0219244301054743 w_k1_ADM_s012
                                 + 0.0117535188298934 w_k1_ADM_s013
                                 + 0.00765297167089399 w_k1_ADM_s014
                                 + 0.0226772451618914 w_k1_ADM_s015
                                 + 0.0243709129323064 w_k1_ADM_s016
                                 + 0.0292002489303158 w_k1_ADM_s017
                                 + 0.0629330565474516 w_k1_ADM_s100
                                 + 0.044008486945297 w_k1_ADM_s101
                                 + 0.137775276190379 w_k1_ADM_s102
                                 + 0.0757768427377803 w_k1_ADM_s103
                                 + 0.053882744174437 w_k1_ADM_s104
                                 + 0.036726055804932 w_k1_ADM_s105
                                 + 0.0319916254876518 w_k1_ADM_s106
                                 + 0.12498257033048 w_k1_ADM_s107
                                 + 0.0139602822116298 w_k1_ADM_s110
                                 + 0.00794585344107854 w_k1_ADM_s111
                                 + 0.0149913684010289 w_k1_ADM_s112
                                 + 0.0150855637449395 w_k1_ADM_s113
                                 + 0.0118876862567339 w_k1_ADM_s114
                                 + 0.0145223344398365 w_k1_ADM_s115
                                 + 0.0167275728106561 w_k1_ADM_s116
                                 + 0.0206287759582689 w_k1_ADM_s117
                                 + 0.0163248072150826 w_k1_AXP_s000
                                 + 0.0404054189151453 w_k1_AXP_s001
                                 + 0.0348903560018515 w_k1_AXP_s002
                                 + 0.0221464400764233 w_k1_AXP_s003
                                 + 0.0136713595266439 w_k1_AXP_s004
                                 + 0.0253731067198322 w_k1_AXP_s005
                                 + 0.0166051089126097 w_k1_AXP_s006
                                 + 0.0120686877925068 w_k1_AXP_s007
                                 + 0.0288044944060681 w_k1_AXP_s010
                                 + 0.0402778889306835 w_k1_AXP_s011
                                 + 0.0200370551224791 w_k1_AXP_s012
                                 + 0.0114637486688914 w_k1_AXP_s013
                                 + 0.00752322674787195 w_k1_AXP_s014
                                 + 0.0206615520218407 w_k1_AXP_s015
                                 + 0.0252663821169124 w_k1_AXP_s016
                                 + 0.0259909570686714 w_k1_AXP_s017
                                 + 0.0599318836851841 w_k1_AXP_s100
                                 + 0.0444591912637833 w_k1_AXP_s101
                                 + 0.120195806208266 w_k1_AXP_s102
                                 + 0.0802109862204456 w_k1_AXP_s103
                                 + 0.0473959551827093 w_k1_AXP_s104
                                 + 0.0363302718093986 w_k1_AXP_s105
                                 + 0.0298382840486332 w_k1_AXP_s106
                                 + 0.124551789923659 w_k1_AXP_s107
                                 + 0.012931907246957 w_k1_AXP_s110
                                 + 0.00800623704576317 w_k1_AXP_s111
                                 + 0.015967668182882 w_k1_AXP_s112
                                 + 0.0133546254781082 w_k1_AXP_s113
                                 + 0.0130385312929806 w_k1_AXP_s114
                                 + 0.0153415019309437 w_k1_AXP_s115
                                 + 0.0189628798455849 w_k1_AXP_s116
                                 + 0.0184269311935196 w_k1_AXP_s117
                                 + 0.0155574722829513 w_k1_DIS_s000
                                 + 0.0387123866426296 w_k1_DIS_s001
                                 + 0.0385051457572251 w_k1_DIS_s002
                                 + 0.0204176361314909 w_k1_DIS_s003
                                 + 0.013245996118692 w_k1_DIS_s004
                                 + 0.0234589810048733 w_k1_DIS_s005
                                 + 0.0193449968246666 w_k1_DIS_s006
                                 + 0.0119166688961934 w_k1_DIS_s007
                                 + 0.029085654760264 w_k1_DIS_s010
                                 + 0.0362417058292837 w_k1_DIS_s011
                                 + 0.0202134715783224 w_k1_DIS_s012
                                 + 0.011416562359733 w_k1_DIS_s013
                                 + 0.00802889293463201 w_k1_DIS_s014
                                 + 0.0217281228244085 w_k1_DIS_s015
                                 + 0.0270368534168648 w_k1_DIS_s016
                                 + 0.0283813749305448 w_k1_DIS_s017
                                 + 0.0600380440726055 w_k1_DIS_s100
                                 + 0.0450612117440417 w_k1_DIS_s101
                                 + 0.123279010134656 w_k1_DIS_s102
                                 + 0.0841490021705768 w_k1_DIS_s103
                                 + 0.0478261987441687 w_k1_DIS_s104
                                 + 0.0352775687737518 w_k1_DIS_s105
                                 + 0.0355686395991779 w_k1_DIS_s106
                                 + 0.127237768988801 w_k1_DIS_s107
                                 + 0.0134193871834647 w_k1_DIS_s110
                                 + 0.00809668080144948 w_k1_DIS_s111
                                 + 0.0141859730277553 w_k1_DIS_s112
                                 + 0.0151939669881804 w_k1_DIS_s113
                                 + 0.0122998472999289 w_k1_DIS_s114
                                 + 0.015623087459386 w_k1_DIS_s115
                                 + 0.0174473092057449 w_k1_DIS_s116
                                 + 0.0210337421244195 w_k1_DIS_s117
                                 + 0.0159866422058956 w_k1_FDX_s000
                                 + 0.0384862062334991 w_k1_FDX_s001
                                 + 0.0377491762940507 w_k1_FDX_s002
                                 + 0.021001922230943 w_k1_FDX_s003
                                 + 0.0132846734743417 w_k1_FDX_s004
                                 + 0.0240945290633093 w_k1_FDX_s005
                                 + 0.0180313208326807 w_k1_FDX_s006
                                 + 0.0121085947282665 w_k1_FDX_s007
                                 + 0.0273989748962567 w_k1_FDX_s010
                                 + 0.0382006690032303 w_k1_FDX_s011
                                 + 0.0193706161607866 w_k1_FDX_s012
                                 + 0.0110895352139155 w_k1_FDX_s013
                                 + 0.00814696026161813 w_k1_FDX_s014
                                 + 0.0205634958595456 w_k1_FDX_s015
                                 + 0.0264163969658383 w_k1_FDX_s016
                                 + 0.0273236021851711 w_k1_FDX_s017
                                 + 0.0588816893347786 w_k1_FDX_s100
                                 + 0.0457217483134509 w_k1_FDX_s101
                                 + 0.123104337642523 w_k1_FDX_s102
                                 + 0.0870649949656193 w_k1_FDX_s103
                                 + 0.0489879623089151 w_k1_FDX_s104
                                 + 0.0329657963584254 w_k1_FDX_s105
                                 + 0.0321562530213619 w_k1_FDX_s106
                                 + 0.122642725257032 w_k1_FDX_s107
                                 + 0.0134543392279163 w_k1_FDX_s110
                                 + 0.00800819032083562 w_k1_FDX_s111
                                 + 0.0142428198641166 w_k1_FDX_s112
                                 + 0.0140521429215466 w_k1_FDX_s113
                                 + 0.0115026736636516 w_k1_FDX_s114
                                 + 0.01462253359178 w_k1_FDX_s115
                                 + 0.0179499105714151 w_k1_FDX_s116
                                 + 0.0204338319996761 w_k1_FDX_s117
                                 + 0.0153458549363395 w_k1_IBM_s000
                                 + 0.0384281796194301 w_k1_IBM_s001
                                 + 0.0371962501988508 w_k1_IBM_s002
                                 + 0.0219313749113108 w_k1_IBM_s003
                                 + 0.0134897641253752 w_k1_IBM_s004
                                 + 0.0243840018562383 w_k1_IBM_s005
                                 + 0.0173071458376859 w_k1_IBM_s006
                                 + 0.0117692182738542 w_k1_IBM_s007
                                 + 0.0288437377199686 w_k1_IBM_s010
                                 + 0.0393644161151925 w_k1_IBM_s011
                                 + 0.0196604957301018 w_k1_IBM_s012
                                 + 0.0105359506225751 w_k1_IBM_s013
                                 + 0.00817571744849664 w_k1_IBM_s014
                                 + 0.0205468704631242 w_k1_IBM_s015
                                 + 0.0254149701099919 w_k1_IBM_s016
                                 + 0.0275329591720149 w_k1_IBM_s017
                                 + 0.0600873063550095 w_k1_IBM_s100
                                 + 0.0438689068111111 w_k1_IBM_s101
                                 + 0.124172553889479 w_k1_IBM_s102
                                 + 0.0875005741765745 w_k1_IBM_s103
                                 + 0.0480980950665742 w_k1_IBM_s104
                                 + 0.0352057777656426 w_k1_IBM_s105
                                 + 0.0322649343589471 w_k1_IBM_s106
                                 + 0.120467393591835 w_k1_IBM_s107
                                 + 0.0141443808234962 w_k1_IBM_s110
                                 + 0.00889509808465518 w_k1_IBM_s111
                                 + 0.0152239126867184 w_k1_IBM_s112
                                 + 0.0145826590409611 w_k1_IBM_s113
                                 + 0.0122147232089167 w_k1_IBM_s114
                                 + 0.0152660453337827 w_k1_IBM_s115
                                 + 0.0182454613371207 w_k1_IBM_s116
                                 + 0.0184791224406273 w_k1_IBM_s117
                                 + 0.0148327558470959 w_k1_INTC_s000
                                 + 0.0364221193658615 w_k1_INTC_s001
                                 + 0.0371695197165048 w_k1_INTC_s002
                                 + 0.0192887461846572 w_k1_INTC_s003
                                 + 0.0131558394545531 w_k1_INTC_s004
                                 + 0.0226518734783054 w_k1_INTC_s005
                                 + 0.0170079243173858 w_k1_INTC_s006
                                 + 0.012819458874371 w_k1_INTC_s007
                                 + 0.0317463018057926 w_k1_INTC_s010
                                 + 0.0418844559363048 w_k1_INTC_s011
                                 + 0.0209191882564434 w_k1_INTC_s012
                                 + 0.00943283474103337 w_k1_INTC_s013
                                 + 0.00727914438728583 w_k1_INTC_s014
                                 + 0.0176438437741802 w_k1_INTC_s015
                                 + 0.0238012555197637 w_k1_INTC_s016
                                 + 0.0248400533100446 w_k1_INTC_s017
                                 + 0.0636678162981854 w_k1_INTC_s100
                                 + 0.0525017634923071 w_k1_INTC_s101
                                 + 0.119660531903469 w_k1_INTC_s102
                                 + 0.0781791258643189 w_k1_INTC_s103
                                 + 0.0497875660982716 w_k1_INTC_s104
                                 + 0.0334961136516032 w_k1_INTC_s105
                                 + 0.0347897868009488 w_k1_INTC_s106
                                 + 0.120971287812997 w_k1_INTC_s107
                                 + 0.0130333530179981 w_k1_INTC_s110
                                 + 0.00792967987314852 w_k1_INTC_s111
                                 + 0.0147600145463897 w_k1_INTC_s112
                                 + 0.0134950503387006 w_k1_INTC_s113
                                 + 0.0114353994987931 w_k1_INTC_s114
                                 + 0.0140923590060128 w_k1_INTC_s115
                                 + 0.0176772296105461 w_k1_INTC_s116
                                 + 0.0209270793174458 w_k1_INTC_s117
                                 + 0.0172494477917868 w_k1_KO_s000
                                 + 0.0354987569174664 w_k1_KO_s001
                                 + 0.036829086965199 w_k1_KO_s002
                                 + 0.0223111010916642 w_k1_KO_s003
                                 + 0.0121781787439366 w_k1_KO_s004
                                 + 0.0224289348303738 w_k1_KO_s005
                                 + 0.0167367006762752 w_k1_KO_s006
                                 + 0.0127126762821025 w_k1_KO_s007
                                 + 0.0274178901097065 w_k1_KO_s010
                                 + 0.0357992355874968 w_k1_KO_s011
                                 + 0.0181689345405392 w_k1_KO_s012
                                 + 0.0120659463007175 w_k1_KO_s013
                                 + 0.00835335592947201 w_k1_KO_s014
                                 + 0.0210442728945735 w_k1_KO_s015
                                 + 0.0286529588105423 w_k1_KO_s016
                                 + 0.0279855832028438 w_k1_KO_s017
                                 + 0.0634610924071724 w_k1_KO_s100
                                 + 0.0469694407852001 w_k1_KO_s101
                                 + 0.112214823453683 w_k1_KO_s102
                                 + 0.0789997784148357 w_k1_KO_s103
                                 + 0.0438983773160247 w_k1_KO_s104
                                 + 0.0343028051428831 w_k1_KO_s105
                                 + 0.0371964743700039 w_k1_KO_s106
                                 + 0.122960386359345 w_k1_KO_s107
                                 + 0.0137231472971712 w_k1_KO_s110
                                 + 0.00834360958247362 w_k1_KO_s111
                                 + 0.0153763686574399 w_k1_KO_s112
                                 + 0.0147300836139186 w_k1_KO_s113
                                 + 0.0123344415210216 w_k1_KO_s114
                                 + 0.0130145127422485 w_k1_KO_s115
                                 + 0.0192339381880548 w_k1_KO_s116
                                 + 0.0208927682792768 w_k1_KO_s117
                                 + 0.0228852474508632 w_k1_MAT_s000
                                 + 0.0372041872495721 w_k1_MAT_s001
                                 + 0.0361994338131051 w_k1_MAT_s002
                                 + 0.0300056160824955 w_k1_MAT_s003
                                 + 0.0140413816236179 w_k1_MAT_s004
                                 + 0.0284498598785935 w_k1_MAT_s005
                                 + 0.0198222708451181 w_k1_MAT_s006
                                 + 0.0132283320397205 w_k1_MAT_s007
                                 + 0.0349907747931089 w_k1_MAT_s010
                                 + 0.0413690089693412 w_k1_MAT_s011
                                 + 0.0165006366548729 w_k1_MAT_s012
                                 + 0.0121463565559663 w_k1_MAT_s013
                                 + 0.00784785128446357 w_k1_MAT_s014
                                 + 0.0213643175964416 w_k1_MAT_s015
                                 + 0.0257709669791017 w_k1_MAT_s016
                                 + 0.0255423515327752 w_k1_MAT_s017
                                 + 0.0624337570107835 w_k1_MAT_s100
                                 + 0.0500167729782588 w_k1_MAT_s101
                                 + 0.126207104906503 w_k1_MAT_s102
                                 + 0.0799540028161854 w_k1_MAT_s103
                                 + 0.0564339845556253 w_k1_MAT_s104
                                 + 0.0373032057137657 w_k1_MAT_s105
                                 + 0.0312990424078029 w_k1_MAT_s106
                                 + 0.107816848334705 w_k1_MAT_s107
                                 + 0.0112599906086564 w_k1_MAT_s110
                                 + 0.00651847015796275 w_k1_MAT_s111
                                 + 0.0177262089821425 w_k1_MAT_s112
                                 + 0.0155770635171852 w_k1_MAT_s113
                                 + 0.0126260094067553 w_k1_MAT_s114
                                 + 0.0152915461928465 w_k1_MAT_s115
                                 + 0.0203235698080904 w_k1_MAT_s116
                                 + 0.0186063624187663 w_k1_MAT_s117
                                 + 0.0158855394086636 w_k1_MSFT_s000
                                 + 0.0357716772136393 w_k1_MSFT_s001
                                 + 0.0325318675865162 w_k1_MSFT_s002
                                 + 0.0211414953843966 w_k1_MSFT_s003
                                 + 0.0137843273630685 w_k1_MSFT_s004
                                 + 0.0236858865726475 w_k1_MSFT_s005
                                 + 0.0188561514885097 w_k1_MSFT_s006
                                 + 0.0128527185434129 w_k1_MSFT_s007
                                 + 0.0277749059985975 w_k1_MSFT_s010
                                 + 0.0384961415503422 w_k1_MSFT_s011
                                 + 0.0194758304007578 w_k1_MSFT_s012
                                 + 0.0116671298023312 w_k1_MSFT_s013
                                 + 0.00765555598717306 w_k1_MSFT_s014
                                 + 0.0207459546676582 w_k1_MSFT_s015
                                 + 0.0269107920520236 w_k1_MSFT_s016
                                 + 0.0268321623665623 w_k1_MSFT_s017
                                 + 0.0648477316890125 w_k1_MSFT_s100
                                 + 0.0470334381316426 w_k1_MSFT_s101
                                 + 0.112329589632519 w_k1_MSFT_s102
                                 + 0.0818830689773124 w_k1_MSFT_s103
                                 + 0.0528411309160157 w_k1_MSFT_s104
                                 + 0.034177363329843 w_k1_MSFT_s105
                                 + 0.0366887697759199 w_k1_MSFT_s106
                                 + 0.121816067654797 w_k1_MSFT_s107
                                 + 0.0132536467614762 w_k1_MSFT_s110
                                 + 0.00793443509027902 w_k1_MSFT_s111
                                 + 0.0134246056968233 w_k1_MSFT_s112
                                 + 0.0144287253419764 w_k1_MSFT_s113
                                 + 0.0119487277747843 w_k1_MSFT_s114
                                 + 0.0139017771600093 w_k1_MSFT_s115
                                 + 0.0183115406912264 w_k1_MSFT_s116
                                 + 0.0198956513405578 w_k1_MSFT_s117
                                 + 0.0152113130897407 w_k1_PG_s000
                                 + 0.0369276402887401 w_k1_PG_s001
                                 + 0.0347585481607396 w_k1_PG_s002
                                 + 0.0211232961402751 w_k1_PG_s003
                                 + 0.0135110105103495 w_k1_PG_s004
                                 + 0.0225539937905292 w_k1_PG_s005
                                 + 0.0165659186896524 w_k1_PG_s006
                                 + 0.0123081066813894 w_k1_PG_s007
                                 + 0.0278864158382827 w_k1_PG_s010
                                 + 0.0365568087599515 w_k1_PG_s011
                                 + 0.0206278397348308 w_k1_PG_s012
                                 + 0.0118230777584941 w_k1_PG_s013
                                 + 0.00742934692591046 w_k1_PG_s014
                                 + 0.0203406316753401 w_k1_PG_s015
                                 + 0.0293381168007661 w_k1_PG_s016
                                 + 0.0267089608937414 w_k1_PG_s017
                                 + 0.0614724763391036 w_k1_PG_s100
                                 + 0.0469819457717965 w_k1_PG_s101
                                 + 0.11959980932033 w_k1_PG_s102
                                 + 0.0906014905195626 w_k1_PG_s103
                                 + 0.0491138349447963 w_k1_PG_s104
                                 + 0.0316062091914689 w_k1_PG_s105
                                 + 0.031145504158285 w_k1_PG_s106
                                 + 0.117101163769689 w_k1_PG_s107
                                 + 0.0130062339735796 w_k1_PG_s110
                                 + 0.00822755778019909 w_k1_PG_s111
                                 + 0.0146745625921664 w_k1_PG_s112
                                 + 0.0143007717663077 w_k1_PG_s113
                                 + 0.0124662376332797 w_k1_PG_s114
                                 + 0.0146520161408224 w_k1_PG_s115
                                 + 0.0164675576716495 w_k1_PG_s116
                                 + 0.0191697439397464 w_k1_PG_s117 >= 1.13
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
