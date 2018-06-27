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
 w_k1_ADM_s0_return_constraint:  1.2027 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.951102 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.892558 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.20859 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.885725 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.15502 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.12669 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.799962 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.880542 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.11168 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.32916 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.06693 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.38214 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.915499 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.32613 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.00873 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.00523 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.763936 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.968281 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.899732 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.204 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.858795 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.09133 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.20997 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.48144 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.0416 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.965672 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.867542 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.815082 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.13835 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.829721 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.865267 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.19031 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.35192 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.956465 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.730511 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.15813 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.960879 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.30329 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.908639 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.878226 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.34787 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.80729 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.21321 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.21319 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.735846 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.837498 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.961096 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.79562 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.09235 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.29431 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.10554 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.25152 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.843496 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.15494 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.691927 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.867042 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.966295 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.30287 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.919487 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.33248 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.30278 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.81737 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.04303 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.747215 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.830233 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.844936 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.40789 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.75605 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.752467 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.08778 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.49673 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.29078 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.709191 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.18345 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.14627 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.36208 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.872395 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.92493 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.45603 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.756736 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.2103 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.942146 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.64685 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.64516 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.12258 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.72779 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.21726 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.55583 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.887998 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.50648 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.773029 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.935525 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.634712 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.961317 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.08591 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.4414 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.94332 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.43882 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.27584 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 2.00555 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.22791 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.753518 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.937206 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.735779 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.22537 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.751154 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.98633 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.18479 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.84783 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.30095 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.659284 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.43285 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.01889 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.20055 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.92982 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.91802 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.19338 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.875943 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.09735 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.0994 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.773892 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.898803 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.20113 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.49565 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.09366 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.45811 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.93476 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.52304 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.926592 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.985917 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.732468 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.04961 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.13689 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.25828 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.901743 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.28251 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.38084 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.59733 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.06977 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.940575 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.93405 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.755255 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.12565 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.753382 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.929723 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.2684 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.58333 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.02733 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.801782 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.35629 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.05031 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.23223 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.937438 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.926258 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.34094 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.77167 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.19566 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.850153 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.803306 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.560455 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.652377 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.26339 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.31969 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.822843 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.37103 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.852258 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.662555 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.62414 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.909524 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.890412 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.742351 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.49553 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.14611 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.93986 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.700957 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.11281 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.09582 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.621764 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.796923 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.12757 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.45795 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.20442 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.837494 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.659301 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.25355 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.72689 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.646617 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.02525 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.06428 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.23624 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.950582 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.923493 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.0466 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.936087 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.15758 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.02884 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.806746 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.725351 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.0198 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.38098 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.14521 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.18428 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.957842 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.42154 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.843691 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.03422 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.853757 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.12183 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.01551 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.40389 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.06596 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.13041 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.1384 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.34827 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.00731 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.855666 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.861125 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.856373 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.984798 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.868531 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.80605 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.06056 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.44191 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.13449 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.776715 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.23481 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.09072 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.26637 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.967283 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.888029 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.36034 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.774441 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.08413 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.935674 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.752852 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.661545 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.874749 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.24214 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.26426 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.08528 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.09487 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.02085 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.701198 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.01682 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.745155 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.908388 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.758603 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.3132 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.01639 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.14635 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.890021 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.42256 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.23409 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.631815 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.856261 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.912272 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.29413 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.993916 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.994085 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.888549 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.48875 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.46533 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.672994 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.21127 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.09052 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.20538 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.878816 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.830031 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.5446 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.716344 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.25205 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.15107 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.722816 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.791962 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.781486 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.93427 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.50106 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.22876 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.72352 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.776831 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.812743 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.63377 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.715339 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.995726 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.71826 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.47051 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.21662 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.52255 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.851058 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.26968 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.33612 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.538376 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.714904 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.04489 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.7785 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.06364 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.694343 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.630445 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.4605 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.91626 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.516554 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.1641 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.3476 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.22112 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.990993 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.918281 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.16879 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.821766 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.02362 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.02415 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.811585 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.810768 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.07579 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.26154 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.22048 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.38356 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.870771 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.43216 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.889791 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.904953 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.75391 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.02083 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.0265 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.21145 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.993445 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.23546 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.17021 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.54852 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.11052 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.939889 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.961885 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.790738 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.07646 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.801717 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.0323 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.13509 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.50751 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.15942 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.820192 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.24581 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.07155 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.1536 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.942922 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.953326 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.18605 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.881657 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.09447 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.972705 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.792357 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.797973 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.966942 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.29344 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.11483 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.09218 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.01401 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.20601 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.915906 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.02408 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.783789 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.962408 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.941336 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.12712 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.949265 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.12515 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.08426 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.32344 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.05138 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.886781 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.936949 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.833442 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.1583 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.867562 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.944567 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.05979 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.28749 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.21394 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.758863 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.07453 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.976236 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.347999846858327 z_k1_s0000
                                 - 0.301999788170179 z_k1_s0001
                                 - 0.253999798017309 z_k1_s0002
                                 - 0.368999781237724 z_k1_s0003
                                 - 0.361001363378489 z_k1_s0004
                                 - 0.2909998381143 z_k1_s0005
                                 - 0.909998246440241 z_k1_s0006
                                 - 0.435001443572471 z_k1_s0007
                                 - 0.44200046357867 z_k1_s0010
                                 - 0.33800017806589 z_k1_s0011
                                 - 0.329000470602557 z_k1_s0012
                                 - 0.681999161306472 z_k1_s0013
                                 - 1.00899903038327 z_k1_s0014
                                 - 1.02000183887452 z_k1_s0015
                                 - 0.927000362649536 z_k1_s0016
                                 - 0.252000307280636 z_k1_s0017
                                 - 0.105000006816453 z_k1_s0020
                                 - 0.153000657018289 z_k1_s0021
                                 - 0.160000677339387 z_k1_s0022
                                 - 0.109000060999939 z_k1_s0023
                                 - 0.271999512476883 z_k1_s0024
                                 - 0.0970000474494867 z_k1_s0025
                                 - 0.325000304908081 z_k1_s0026
                                 - 0.269000142339299 z_k1_s0027
                                 - 0.222999169122143 z_k1_s0030
                                 - 0.268999007645092 z_k1_s0031
                                 - 0.423998896124442 z_k1_s0032
                                 - 0.347000848704989 z_k1_s0033
                                 - 0.210999855320273 z_k1_s0034
                                 - 0.334000173361805 z_k1_s0035
                                 - 0.133000004159664 z_k1_s0036
                                 - 0.270000590286328 z_k1_s0037
                                 - 0.12700000768711 z_k1_s0040
                                 - 0.239999998676424 z_k1_s0041
                                 - 0.303000879016755 z_k1_s0042
                                 - 0.272000669500153 z_k1_s0043
                                 - 0.404999620328849 z_k1_s0044
                                 - 0.1539999823757 z_k1_s0045
                                 - 0.211999160080583 z_k1_s0046
                                 - 0.299999495095375 z_k1_s0047
                                 - 0.628996084514338 z_k1_s0050
                                 - 0.357999566108775 z_k1_s0051
                                 - 0.128999556739276 z_k1_s0052
                                 - 0.38099842407293 z_k1_s0053
                                 - 0.335999184975302 z_k1_s0054
                                 - 0.195999176970377 z_k1_s0055
                                 - 0.390998834314428 z_k1_s0056
                                 - 0.186999329150852 z_k1_s0057
                                 - 0.346999300757011 z_k1_s0060
                                 - 0.0979999718489536 z_k1_s0061
                                 - 0.123000087159401 z_k1_s0062
                                 - 0.335999662796413 z_k1_s0063
                                 - 0.339998956782087 z_k1_s0064
                                 - 0.242000459133128 z_k1_s0065
                                 - 0.168000041998206 z_k1_s0066
                                 - 0.452000248380887 z_k1_s0067
                                 - 1.03500275649989 z_k1_s0070
                                 - 0.618998848932754 z_k1_s0071
                                 - 0.95500157804378 z_k1_s0072
                                 - 0.406000984754344 z_k1_s0073
                                 - 0.445000726599963 z_k1_s0074
                                 - 0.415000839780079 z_k1_s0075
                                 - 1.08300235337124 z_k1_s0076
                                 - 0.593003461844915 z_k1_s0077
                                 - 0.0619998754157661 z_k1_s0100
                                 - 0.0809998142867808 z_k1_s0101
                                 - 0.112999755385228 z_k1_s0102
                                 - 0.0569999003128467 z_k1_s0103
                                 - 0.099999502058388 z_k1_s0104
                                 - 0.115999556307414 z_k1_s0105
                                 - 0.107000153540856 z_k1_s0106
                                 - 0.200999216756846 z_k1_s0107
                                 - 0.0640001317087838 z_k1_s0110
                                 - 0.212000426072829 z_k1_s0111
                                 - 0.0510000870835314 z_k1_s0112
                                 - 0.0650001414261217 z_k1_s0113
                                 - 0.18300014427003 z_k1_s0114
                                 - 0.0819999798203048 z_k1_s0115
                                 - 0.110000251905766 z_k1_s0116
                                 - 0.0500000773661935 z_k1_s0117
                                 - 0.0409999738100195 z_k1_s0120
                                 - 0.0960001830915343 z_k1_s0121
                                 - 0.149000293662894 z_k1_s0122
                                 - 0.142999542032835 z_k1_s0123
                                 - 0.125000020140214 z_k1_s0124
                                 - 0.135999424297088 z_k1_s0125
                                 - 0.0990001034069712 z_k1_s0126
                                 - 0.122000099824777 z_k1_s0127
                                 - 0.0719999114006062 z_k1_s0130
                                 - 0.114999827514899 z_k1_s0131
                                 - 0.231999962290503 z_k1_s0132
                                 - 0.276999070667027 z_k1_s0133
                                 - 0.079999913945101 z_k1_s0134
                                 - 0.136999862387222 z_k1_s0135
                                 - 0.096999849664748 z_k1_s0136
                                 - 0.104999852209243 z_k1_s0137
                                 - 0.0289999905427188 z_k1_s0140
                                 - 0.0529997449505177 z_k1_s0141
                                 - 0.10899991550195 z_k1_s0142
                                 - 0.0839997488272153 z_k1_s0143
                                 - 0.0419999758135942 z_k1_s0144
                                 - 0.0310000038766976 z_k1_s0145
                                 - 0.0749999930242705 z_k1_s0146
                                 - 0.0839997488272153 z_k1_s0147
                                 - 0.141000900027191 z_k1_s0150
                                 - 0.15600082896571 z_k1_s0151
                                 - 0.211000568406945 z_k1_s0152
                                 - 0.0520001456548104 z_k1_s0153
                                 - 0.119000808250058 z_k1_s0154
                                 - 0.134000737188577 z_k1_s0155
                                 - 0.118000224985572 z_k1_s0156
                                 - 0.0490001598671067 z_k1_s0157
                                 - 0.108999834872741 z_k1_s0160
                                 - 0.0919999951127785 z_k1_s0161
                                 - 0.246999881041911 z_k1_s0162
                                 - 0.0730000042634007 z_k1_s0163
                                 - 0.10499996069393 z_k1_s0164
                                 - 0.0869999918892578 z_k1_s0165
                                 - 0.0780000074869214 z_k1_s0166
                                 - 0.279000372472465 z_k1_s0167
                                 - 0.42699873011425 z_k1_s0170
                                 - 0.264999489653512 z_k1_s0171
                                 - 0.105999795861405 z_k1_s0172
                                 - 0.108999701929349 z_k1_s0173
                                 - 0.188999113870861 z_k1_s0174
                                 - 0.131999820381984 z_k1_s0175
                                 - 0.272999718367114 z_k1_s0176
                                 - 0.295999477420436 z_k1_s0177
                                 - 0.093999997540921 z_k1_s1000
                                 - 0.0789999425120927 z_k1_s1001
                                 - 0.2649991271107 z_k1_s1002
                                 - 0.109999969411737 z_k1_s1003
                                 - 0.22600041667466 z_k1_s1004
                                 - 0.299999472577801 z_k1_s1005
                                 - 0.308999049755444 z_k1_s1006
                                 - 0.24499916227218 z_k1_s1007
                                 - 0.104000009790238 z_k1_s1010
                                 - 0.426000945872161 z_k1_s1011
                                 - 0.193000229514719 z_k1_s1012
                                 - 0.113000040829798 z_k1_s1013
                                 - 0.12500002988251 z_k1_s1014
                                 - 0.218000926291684 z_k1_s1015
                                 - 0.234000807028563 z_k1_s1016
                                 - 0.157000105356478 z_k1_s1017
                                 - 0.371999813764104 z_k1_s1020
                                 - 0.206000529450412 z_k1_s1021
                                 - 0.154000023773375 z_k1_s1022
                                 - 0.156000086030211 z_k1_s1023
                                 - 0.10100004346827 z_k1_s1024
                                 - 0.319000310459299 z_k1_s1025
                                 - 0.123000092293067 z_k1_s1026
                                 - 0.159000099915415 z_k1_s1027
                                 - 0.928996363630374 z_k1_s1030
                                 - 0.847996431491173 z_k1_s1031
                                 - 0.26299905705676 z_k1_s1032
                                 - 0.698995693509485 z_k1_s1033
                                 - 0.292999258411293 z_k1_s1034
                                 - 0.333998922077784 z_k1_s1035
                                 - 0.191999192035737 z_k1_s1036
                                 - 0.809999321392008 z_k1_s1037
                                 - 0.244999662068477 z_k1_s1040
                                 - 0.68299921240127 z_k1_s1041
                                 - 0.446999511183513 z_k1_s1042
                                 - 0.302999409773118 z_k1_s1043
                                 - 0.688999914100622 z_k1_s1044
                                 - 0.37599909222896 z_k1_s1045
                                 - 0.18999953741464 z_k1_s1046
                                 - 0.705997656755367 z_k1_s1047
                                 - 0.407001916058162 z_k1_s1050
                                 - 0.18500068340772 z_k1_s1051
                                 - 0.394999823790193 z_k1_s1052
                                 - 0.403000072632234 z_k1_s1053
                                 - 0.823001103805032 z_k1_s1054
                                 - 0.19900060317982 z_k1_s1055
                                 - 0.814003832116324 z_k1_s1056
                                 - 0.2120007496974 z_k1_s1057
                                 - 1.11699644998566 z_k1_s1060
                                 - 0.943000391719304 z_k1_s1061
                                 - 1.23399930355984 z_k1_s1062
                                 - 0.687996845210767 z_k1_s1063
                                 - 0.300000033422921 z_k1_s1064
                                 - 0.493999742316494 z_k1_s1065
                                 - 0.973998753393447 z_k1_s1066
                                 - 0.764997609529113 z_k1_s1067
                                 - 0.633001379039996 z_k1_s1070
                                 - 0.215000234487132 z_k1_s1071
                                 - 0.221999904999634 z_k1_s1072
                                 - 0.60200112168411 z_k1_s1073
                                 - 0.604001138287715 z_k1_s1074
                                 - 0.556000739801181 z_k1_s1075
                                 - 0.203000134865498 z_k1_s1076
                                 - 0.840999229813807 z_k1_s1077
                                 - 0.859001309571034 z_k1_s1100
                                 - 0.779000942711614 z_k1_s1101
                                 - 0.501001534030092 z_k1_s1102
                                 - 1.86600567293832 z_k1_s1103
                                 - 0.751999376867722 z_k1_s1104
                                 - 0.728004017022535 z_k1_s1105
                                 - 0.70700129113994 z_k1_s1106
                                 - 0.594001875678943 z_k1_s1107
                                 - 0.482999405541113 z_k1_s1110
                                 - 0.461999254622172 z_k1_s1111
                                 - 1.27600039360079 z_k1_s1112
                                 - 0.456999250955714 z_k1_s1113
                                 - 0.457999387208823 z_k1_s1114
                                 - 1.86199486043769 z_k1_s1115
                                 - 0.824001146311536 z_k1_s1116
                                 - 0.953998531243092 z_k1_s1117
                                 - 0.605999469860919 z_k1_s1120
                                 - 0.217999883427499 z_k1_s1121
                                 - 0.315000075610747 z_k1_s1122
                                 - 0.560000134419107 z_k1_s1123
                                 - 0.814998506628415 z_k1_s1124
                                 - 0.397000433093424 z_k1_s1125
                                 - 0.650000719021975 z_k1_s1126
                                 - 0.379999753413087 z_k1_s1127
                                 - 0.456000336547491 z_k1_s1130
                                 - 0.374000160203375 z_k1_s1131
                                 - 0.145000118598659 z_k1_s1132
                                 - 0.381001584221163 z_k1_s1133
                                 - 0.58900098487408 z_k1_s1134
                                 - 0.455000133116379 z_k1_s1135
                                 - 0.184000130008634 z_k1_s1136
                                 - 0.717000616045106 z_k1_s1137
                                 - 0.53700238821841 z_k1_s1140
                                 - 0.493000029101837 z_k1_s1141
                                 - 0.854001748884066 z_k1_s1142
                                 - 0.563003981515252 z_k1_s1143
                                 - 0.58500296906601 z_k1_s1144
                                 - 0.3270010526143 z_k1_s1145
                                 - 0.646002154554503 z_k1_s1146
                                 - 0.379001170397444 z_k1_s1147
                                 - 0.752996605665717 z_k1_s1150
                                 - 0.940999137578117 z_k1_s1151
                                 - 0.279999453899491 z_k1_s1152
                                 - 1.0109998418013 z_k1_s1153
                                 - 0.386999249214645 z_k1_s1154
                                 - 0.819998640919454 z_k1_s1155
                                 - 0.652998001770626 z_k1_s1156
                                 - 0.759997236586908 z_k1_s1157
                                 - 0.293000053575857 z_k1_s1160
                                 - 0.642000083405112 z_k1_s1161
                                 - 0.457999176572269 z_k1_s1162
                                 - 0.559000132279544 z_k1_s1163
                                 - 0.600000292715112 z_k1_s1164
                                 - 0.234000037558874 z_k1_s1165
                                 - 0.801998374128687 z_k1_s1166
                                 - 0.242000143595102 z_k1_s1167
                                 - 0.429000493632201 z_k1_s1170
                                 - 0.639001074711485 z_k1_s1171
                                 - 0.735000415774711 z_k1_s1172
                                 - 1.14599901045408 z_k1_s1173
                                 - 0.277000781880796 z_k1_s1174
                                 - 1.29300427121792 z_k1_s1175
                                 - 1.0579990751452 z_k1_s1176
                                 - 0.895001474883796 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.01257 w_k1_ADM_s000
                                 + 1.04405 w_k1_AXP_s000
                                 + 1.16079 w_k1_DIS_s000
                                 + 1.03598 w_k1_FDX_s000 + 1.5221 w_k1_IBM_s000
                                 + 1.07911 w_k1_INTC_s000
                                 + 1.14643 w_k1_KO_s000 + 1.31593 w_k1_MAT_s000
                                 + 1.0223 w_k1_MSFT_s000 + 1.02827 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.38585 w_k1_ADM_s000
                                 + 1.25821 w_k1_AXP_s000 + 1.5143 w_k1_DIS_s000
                                 + 1.38298 w_k1_FDX_s000
                                 + 1.20867 w_k1_IBM_s000
                                 + 1.21799 w_k1_INTC_s000
                                 + 1.22065 w_k1_KO_s000 + 1.05354 w_k1_MAT_s000
                                 + 1.20365 w_k1_MSFT_s000
                                 + 1.33595 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.19621 w_k1_ADM_s000
                                 + 1.31392 w_k1_AXP_s000
                                 + 1.05592 w_k1_DIS_s000 + 1.2011 w_k1_FDX_s000
                                 + 0.826338 w_k1_IBM_s000
                                 + 1.0668 w_k1_INTC_s000 + 1.00452 w_k1_KO_s000
                                 + 0.93494 w_k1_MAT_s000
                                 + 1.00582 w_k1_MSFT_s000
                                 + 1.09903 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.985972 w_k1_ADM_s000
                                 + 0.587395 w_k1_AXP_s000
                                 + 0.704475 w_k1_DIS_s000
                                 + 0.81724 w_k1_FDX_s000
                                 + 0.686169 w_k1_IBM_s000
                                 + 0.732941 w_k1_INTC_s000
                                 + 0.742572 w_k1_KO_s000
                                 + 0.499734 w_k1_MAT_s000
                                 + 0.779205 w_k1_MSFT_s000
                                 + 0.86044 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.755019 w_k1_ADM_s000
                                 + 0.722227 w_k1_AXP_s000
                                 + 0.694877 w_k1_DIS_s000
                                 + 0.733703 w_k1_FDX_s000
                                 + 0.882945 w_k1_IBM_s000
                                 + 0.828077 w_k1_INTC_s000
                                 + 0.838885 w_k1_KO_s000
                                 + 0.73677 w_k1_MAT_s000
                                 + 0.836928 w_k1_MSFT_s000
                                 + 0.839309 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.07609 w_k1_ADM_s000
                                 + 1.06225 w_k1_AXP_s000
                                 + 1.31245 w_k1_DIS_s000
                                 + 1.11989 w_k1_FDX_s000
                                 + 1.11074 w_k1_IBM_s000
                                 + 1.15739 w_k1_INTC_s000
                                 + 1.11829 w_k1_KO_s000 + 1.21839 w_k1_MAT_s000
                                 + 0.976663 w_k1_MSFT_s000
                                 + 1.05969 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.799741 w_k1_ADM_s000
                                 + 0.873987 w_k1_AXP_s000
                                 + 0.959896 w_k1_DIS_s000
                                 + 0.9466 w_k1_FDX_s000
                                 + 0.840064 w_k1_IBM_s000
                                 + 0.917142 w_k1_INTC_s000
                                 + 0.951503 w_k1_KO_s000
                                 + 0.692817 w_k1_MAT_s000
                                 + 0.926629 w_k1_MSFT_s000
                                 + 1.01011 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.973474 w_k1_ADM_s000
                                 + 0.814094 w_k1_AXP_s000
                                 + 0.991287 w_k1_DIS_s000
                                 + 1.00518 w_k1_FDX_s000
                                 + 1.09156 w_k1_IBM_s000
                                 + 0.910407 w_k1_INTC_s000
                                 + 0.987275 w_k1_KO_s000
                                 + 0.965252 w_k1_MAT_s000
                                 + 1.02082 w_k1_MSFT_s000
                                 + 1.00656 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.684883 w_k1_ADM_s001
                                 + 0.909512 w_k1_AXP_s001
                                 + 0.761292 w_k1_DIS_s001
                                 + 0.749469 w_k1_FDX_s001
                                 + 1.34894 w_k1_IBM_s001
                                 + 0.746506 w_k1_INTC_s001
                                 + 1.03316 w_k1_KO_s001 + 1.63261 w_k1_MAT_s001
                                 + 0.811437 w_k1_MSFT_s001
                                 + 0.870107 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.1398 w_k1_ADM_s001
                                 + 1.27018 w_k1_AXP_s001
                                 + 1.20691 w_k1_DIS_s001
                                 + 1.01836 w_k1_FDX_s001
                                 + 1.87974 w_k1_IBM_s001
                                 + 0.945045 w_k1_INTC_s001
                                 + 1.5591 w_k1_KO_s001 + 2.58642 w_k1_MAT_s001
                                 + 0.95382 w_k1_MSFT_s001
                                 + 1.27009 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.10849 w_k1_ADM_s001
                                 + 1.28277 w_k1_AXP_s001
                                 + 1.18078 w_k1_DIS_s001
                                 + 1.05654 w_k1_FDX_s001
                                 + 1.45129 w_k1_IBM_s001
                                 + 0.945692 w_k1_INTC_s001
                                 + 1.22676 w_k1_KO_s001 + 1.91837 w_k1_MAT_s001
                                 + 0.941361 w_k1_MSFT_s001
                                 + 1.07713 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.3971 w_k1_ADM_s001
                                 + 1.24872 w_k1_AXP_s001
                                 + 1.36932 w_k1_DIS_s001
                                 + 1.33123 w_k1_FDX_s001
                                 + 0.901276 w_k1_IBM_s001
                                 + 1.16487 w_k1_INTC_s001
                                 + 1.25334 w_k1_KO_s001 + 1.37748 w_k1_MAT_s001
                                 + 1.17356 w_k1_MSFT_s001 + 1.2801 w_k1_PG_s001
                                 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.04582 w_k1_ADM_s001
                                 + 0.831909 w_k1_AXP_s001
                                 + 0.866309 w_k1_DIS_s001
                                 + 0.89411 w_k1_FDX_s001
                                 + 1.02055 w_k1_IBM_s001
                                 + 0.81274 w_k1_INTC_s001
                                 + 0.933575 w_k1_KO_s001
                                 + 0.821406 w_k1_MAT_s001
                                 + 0.803135 w_k1_MSFT_s001
                                 + 0.957124 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.887963 w_k1_ADM_s001
                                 + 0.784368 w_k1_AXP_s001
                                 + 0.745502 w_k1_DIS_s001
                                 + 0.833373 w_k1_FDX_s001
                                 + 0.685823 w_k1_IBM_s001
                                 + 1.037 w_k1_INTC_s001 + 0.850109 w_k1_KO_s001
                                 + 0.749594 w_k1_MAT_s001
                                 + 0.860479 w_k1_MSFT_s001
                                 + 0.844949 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.12506 w_k1_ADM_s001
                                 + 0.98491 w_k1_AXP_s001
                                 + 1.19164 w_k1_DIS_s001
                                 + 1.07444 w_k1_FDX_s001
                                 + 1.08283 w_k1_IBM_s001
                                 + 0.84694 w_k1_INTC_s001
                                 + 1.19133 w_k1_KO_s001 + 1.59834 w_k1_MAT_s001
                                 + 0.969897 w_k1_MSFT_s001
                                 + 1.11389 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.19514 w_k1_ADM_s001
                                 + 1.36287 w_k1_AXP_s001
                                 + 1.84835 w_k1_DIS_s001
                                 + 1.41462 w_k1_FDX_s001
                                 + 1.33864 w_k1_IBM_s001
                                 + 1.39637 w_k1_INTC_s001
                                 + 1.62462 w_k1_KO_s001 + 2.27461 w_k1_MAT_s001
                                 + 1.47704 w_k1_MSFT_s001
                                 + 1.34328 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.10586 w_k1_ADM_s002
                                 + 1.46098 w_k1_AXP_s002
                                 + 1.22017 w_k1_DIS_s002 + 1.2619 w_k1_FDX_s002
                                 + 1.5172 w_k1_IBM_s002
                                 + 1.18297 w_k1_INTC_s002
                                 + 1.21539 w_k1_KO_s002 + 1.60594 w_k1_MAT_s002
                                 + 1.13616 w_k1_MSFT_s002
                                 + 1.16893 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.843095 w_k1_ADM_s002
                                 + 0.66346 w_k1_AXP_s002 + 0.6534 w_k1_DIS_s002
                                 + 0.778099 w_k1_FDX_s002
                                 + 0.700865 w_k1_IBM_s002
                                 + 0.946974 w_k1_INTC_s002
                                 + 0.779172 w_k1_KO_s002
                                 + 0.615934 w_k1_MAT_s002
                                 + 0.837344 w_k1_MSFT_s002
                                 + 0.786495 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.912683 w_k1_ADM_s002
                                 + 1.11892 w_k1_AXP_s002
                                 + 0.900877 w_k1_DIS_s002
                                 + 0.933413 w_k1_FDX_s002
                                 + 1.45554 w_k1_IBM_s002
                                 + 1.11849 w_k1_INTC_s002
                                 + 1.01054 w_k1_KO_s002 + 1.03461 w_k1_MAT_s002
                                 + 0.937586 w_k1_MSFT_s002
                                 + 1.13149 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.21197 w_k1_ADM_s002
                                 + 1.17562 w_k1_AXP_s002
                                 + 1.55226 w_k1_DIS_s002
                                 + 1.34697 w_k1_FDX_s002
                                 + 1.05838 w_k1_IBM_s002
                                 + 1.27391 w_k1_INTC_s002
                                 + 1.26539 w_k1_KO_s002
                                 + 0.943636 w_k1_MAT_s002
                                 + 1.29918 w_k1_MSFT_s002
                                 + 1.19851 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.844879 w_k1_ADM_s002
                                 + 0.797476 w_k1_AXP_s002
                                 + 0.779401 w_k1_DIS_s002
                                 + 0.812767 w_k1_FDX_s002
                                 + 1.03904 w_k1_IBM_s002
                                 + 1.07507 w_k1_INTC_s002
                                 + 0.831913 w_k1_KO_s002
                                 + 0.717365 w_k1_MAT_s002
                                 + 0.754897 w_k1_MSFT_s002
                                 + 0.910236 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.57956 w_k1_ADM_s002
                                 + 1.32808 w_k1_AXP_s002
                                 + 1.81267 w_k1_DIS_s002
                                 + 1.29133 w_k1_FDX_s002
                                 + 1.43551 w_k1_IBM_s002
                                 + 1.28039 w_k1_INTC_s002
                                 + 1.40922 w_k1_KO_s002 + 1.38443 w_k1_MAT_s002
                                 + 1.22234 w_k1_MSFT_s002
                                 + 1.28853 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.08401 w_k1_ADM_s002
                                 + 1.03343 w_k1_AXP_s002 + 1.2727 w_k1_DIS_s002
                                 + 1.07416 w_k1_FDX_s002
                                 + 1.15721 w_k1_IBM_s002
                                 + 1.30786 w_k1_INTC_s002
                                 + 1.14267 w_k1_KO_s002
                                 + 0.934268 w_k1_MAT_s002
                                 + 1.19264 w_k1_MSFT_s002
                                 + 1.14072 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.1276 w_k1_ADM_s002
                                 + 0.95042 w_k1_AXP_s002
                                 + 1.04639 w_k1_DIS_s002
                                 + 1.02631 w_k1_FDX_s002 + 0.7042 w_k1_IBM_s002
                                 + 1.10799 w_k1_INTC_s002
                                 + 0.883167 w_k1_KO_s002
                                 + 0.79858 w_k1_MAT_s002
                                 + 0.999746 w_k1_MSFT_s002
                                 + 0.963189 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.21511 w_k1_ADM_s003
                                 + 1.53309 w_k1_AXP_s003 + 1.2757 w_k1_DIS_s003
                                 + 1.3216 w_k1_FDX_s003 + 1.03998 w_k1_IBM_s003
                                 + 1.25559 w_k1_INTC_s003
                                 + 1.16116 w_k1_KO_s003 + 1.22471 w_k1_MAT_s003
                                 + 1.34289 w_k1_MSFT_s003 + 1.0367 w_k1_PG_s003
                                 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.872932 w_k1_ADM_s003
                                 + 0.702704 w_k1_AXP_s003
                                 + 0.768775 w_k1_DIS_s003
                                 + 0.708749 w_k1_FDX_s003
                                 + 1.21425 w_k1_IBM_s003
                                 + 0.827828 w_k1_INTC_s003
                                 + 1.06446 w_k1_KO_s003
                                 + 0.807448 w_k1_MAT_s003
                                 + 0.939302 w_k1_MSFT_s003
                                 + 0.89217 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.03994 w_k1_ADM_s003
                                 + 0.981991 w_k1_AXP_s003
                                 + 0.968481 w_k1_DIS_s003
                                 + 1.0034 w_k1_FDX_s003
                                 + 0.737015 w_k1_IBM_s003
                                 + 0.964235 w_k1_INTC_s003
                                 + 0.995485 w_k1_KO_s003
                                 + 0.999765 w_k1_MAT_s003
                                 + 1.05465 w_k1_MSFT_s003
                                 + 1.04045 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.858935 w_k1_ADM_s003
                                 + 0.740676 w_k1_AXP_s003
                                 + 0.614186 w_k1_DIS_s003
                                 + 0.803967 w_k1_FDX_s003
                                 + 0.67075 w_k1_IBM_s003
                                 + 0.873269 w_k1_INTC_s003
                                 + 0.759024 w_k1_KO_s003
                                 + 0.502172 w_k1_MAT_s003
                                 + 0.840538 w_k1_MSFT_s003
                                 + 0.908956 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.17904 w_k1_ADM_s003
                                 + 1.08201 w_k1_AXP_s003 + 1.1093 w_k1_DIS_s003
                                 + 0.968716 w_k1_FDX_s003
                                 + 1.56887 w_k1_IBM_s003
                                 + 1.10506 w_k1_INTC_s003
                                 + 1.40939 w_k1_KO_s003 + 1.51318 w_k1_MAT_s003
                                 + 1.18865 w_k1_MSFT_s003
                                 + 1.04532 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.28742 w_k1_ADM_s003
                                 + 1.10384 w_k1_AXP_s003
                                 + 1.30953 w_k1_DIS_s003
                                 + 1.17397 w_k1_FDX_s003
                                 + 1.08405 w_k1_IBM_s003
                                 + 1.34245 w_k1_INTC_s003
                                 + 1.20014 w_k1_KO_s003
                                 + 0.935059 w_k1_MAT_s003
                                 + 1.19673 w_k1_MSFT_s003
                                 + 1.15532 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.46256 w_k1_ADM_s003
                                 + 1.66176 w_k1_AXP_s003
                                 + 1.81976 w_k1_DIS_s003
                                 + 1.44164 w_k1_FDX_s003 + 1.3848 w_k1_IBM_s003
                                 + 1.41932 w_k1_INTC_s003
                                 + 1.55496 w_k1_KO_s003 + 1.12209 w_k1_MAT_s003
                                 + 1.51809 w_k1_MSFT_s003
                                 + 1.27252 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.916856 w_k1_ADM_s003
                                 + 1.14941 w_k1_AXP_s003
                                 + 1.00023 w_k1_DIS_s003
                                 + 0.936285 w_k1_FDX_s003
                                 + 1.12222 w_k1_IBM_s003
                                 + 1.05054 w_k1_INTC_s003
                                 + 1.06044 w_k1_KO_s003 + 1.1165 w_k1_MAT_s003
                                 + 0.975986 w_k1_MSFT_s003
                                 + 1.21985 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.68515 w_k1_ADM_s004
                                 + 1.50982 w_k1_AXP_s004
                                 + 1.69296 w_k1_DIS_s004
                                 + 1.51308 w_k1_FDX_s004
                                 + 0.936531 w_k1_IBM_s004
                                 + 0.949778 w_k1_INTC_s004
                                 + 1.32892 w_k1_KO_s004
                                 + 0.982016 w_k1_MAT_s004
                                 + 1.33735 w_k1_MSFT_s004
                                 + 1.18594 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.31197 w_k1_ADM_s004
                                 + 1.14129 w_k1_AXP_s004
                                 + 0.872486 w_k1_DIS_s004
                                 + 1.04928 w_k1_FDX_s004
                                 + 1.09753 w_k1_IBM_s004
                                 + 0.97598 w_k1_INTC_s004
                                 + 1.14372 w_k1_KO_s004
                                 + 0.950264 w_k1_MAT_s004
                                 + 0.973825 w_k1_MSFT_s004
                                 + 0.990835 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.10779 w_k1_ADM_s004
                                 + 0.979809 w_k1_AXP_s004
                                 + 1.2047 w_k1_DIS_s004 + 1.1067 w_k1_FDX_s004
                                 + 1.06226 w_k1_IBM_s004
                                 + 0.963977 w_k1_INTC_s004
                                 + 1.41969 w_k1_KO_s004
                                 + 0.863522 w_k1_MAT_s004
                                 + 1.22049 w_k1_MSFT_s004
                                 + 0.985452 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.02795 w_k1_ADM_s004
                                 + 0.944916 w_k1_AXP_s004 + 0.846 w_k1_DIS_s004
                                 + 1.01816 w_k1_FDX_s004
                                 + 0.76645 w_k1_IBM_s004
                                 + 0.87621 w_k1_INTC_s004
                                 + 0.992818 w_k1_KO_s004
                                 + 0.621119 w_k1_MAT_s004
                                 + 1.01298 w_k1_MSFT_s004
                                 + 0.881089 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.852749 w_k1_ADM_s004
                                 + 0.631991 w_k1_AXP_s004 + 0.634 w_k1_DIS_s004
                                 + 0.760087 w_k1_FDX_s004
                                 + 0.823828 w_k1_IBM_s004
                                 + 0.840411 w_k1_INTC_s004
                                 + 0.978744 w_k1_KO_s004
                                 + 0.614117 w_k1_MAT_s004
                                 + 0.864793 w_k1_MSFT_s004
                                 + 0.82961 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.21078 w_k1_ADM_s004
                                 + 0.890595 w_k1_AXP_s004 + 1.602 w_k1_DIS_s004
                                 + 1.19781 w_k1_FDX_s004
                                 + 1.14519 w_k1_IBM_s004
                                 + 0.954042 w_k1_INTC_s004
                                 + 1.62965 w_k1_KO_s004
                                 + 0.903399 w_k1_MAT_s004
                                 + 1.36507 w_k1_MSFT_s004
                                 + 1.02461 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.00251 w_k1_ADM_s004
                                 + 0.719589 w_k1_AXP_s004
                                 + 0.839101 w_k1_DIS_s004
                                 + 0.852471 w_k1_FDX_s004
                                 + 1.21564 w_k1_IBM_s004
                                 + 0.850736 w_k1_INTC_s004
                                 + 1.2157 w_k1_KO_s004 + 0.823109 w_k1_MAT_s004
                                 + 0.943921 w_k1_MSFT_s004
                                 + 0.99649 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.199 w_k1_ADM_s004
                                 + 1.17533 w_k1_AXP_s004 + 1.1737 w_k1_DIS_s004
                                 + 0.978716 w_k1_FDX_s004
                                 + 1.66417 w_k1_IBM_s004
                                 + 1.01688 w_k1_INTC_s004
                                 + 1.50555 w_k1_KO_s004 + 1.77022 w_k1_MAT_s004
                                 + 1.15523 w_k1_MSFT_s004
                                 + 1.05556 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.31492 w_k1_ADM_s005
                                 + 1.05563 w_k1_AXP_s005
                                 + 1.13427 w_k1_DIS_s005
                                 + 1.09957 w_k1_FDX_s005
                                 + 0.950965 w_k1_IBM_s005
                                 + 0.782424 w_k1_INTC_s005
                                 + 0.956606 w_k1_KO_s005
                                 + 1.17949 w_k1_MAT_s005
                                 + 0.879037 w_k1_MSFT_s005
                                 + 0.975033 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.879162 w_k1_ADM_s005
                                 + 0.961745 w_k1_AXP_s005
                                 + 0.925115 w_k1_DIS_s005
                                 + 0.85995 w_k1_FDX_s005
                                 + 1.26316 w_k1_IBM_s005
                                 + 0.970989 w_k1_INTC_s005
                                 + 1.10729 w_k1_KO_s005 + 1.34429 w_k1_MAT_s005
                                 + 0.824479 w_k1_MSFT_s005
                                 + 0.908576 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.53501 w_k1_ADM_s005
                                 + 1.5046 w_k1_AXP_s005 + 1.78587 w_k1_DIS_s005
                                 + 1.30351 w_k1_FDX_s005
                                 + 1.61625 w_k1_IBM_s005
                                 + 1.00699 w_k1_INTC_s005
                                 + 1.29728 w_k1_KO_s005 + 1.37963 w_k1_MAT_s005
                                 + 0.952105 w_k1_MSFT_s005
                                 + 1.22545 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.20602 w_k1_ADM_s005
                                 + 0.921352 w_k1_AXP_s005
                                 + 0.833723 w_k1_DIS_s005
                                 + 0.934967 w_k1_FDX_s005
                                 + 0.645245 w_k1_IBM_s005
                                 + 0.866524 w_k1_INTC_s005
                                 + 0.703629 w_k1_KO_s005
                                 + 0.70451 w_k1_MAT_s005
                                 + 0.885701 w_k1_MSFT_s005
                                 + 0.952948 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.765272 w_k1_ADM_s005
                                 + 0.593345 w_k1_AXP_s005
                                 + 0.743499 w_k1_DIS_s005
                                 + 0.772133 w_k1_FDX_s005
                                 + 0.795945 w_k1_IBM_s005
                                 + 0.866475 w_k1_INTC_s005
                                 + 0.753237 w_k1_KO_s005
                                 + 0.512272 w_k1_MAT_s005
                                 + 0.741468 w_k1_MSFT_s005
                                 + 0.795483 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.42646 w_k1_ADM_s005
                                 + 1.4732 w_k1_AXP_s005 + 1.38042 w_k1_DIS_s005
                                 + 1.14458 w_k1_FDX_s005 + 1.6493 w_k1_IBM_s005
                                 + 1.24583 w_k1_INTC_s005
                                 + 1.39872 w_k1_KO_s005 + 1.50663 w_k1_MAT_s005
                                 + 1.23052 w_k1_MSFT_s005
                                 + 1.24973 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.772618 w_k1_ADM_s005
                                 + 0.716865 w_k1_AXP_s005
                                 + 1.17625 w_k1_DIS_s005 + 0.9881 w_k1_FDX_s005
                                 + 0.879869 w_k1_IBM_s005
                                 + 1.03401 w_k1_INTC_s005
                                 + 1.15433 w_k1_KO_s005
                                 + 0.856638 w_k1_MAT_s005
                                 + 1.0288 w_k1_MSFT_s005
                                 + 0.976543 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.30271 w_k1_ADM_s005
                                 + 1.33409 w_k1_AXP_s005
                                 + 1.60211 w_k1_DIS_s005
                                 + 1.34563 w_k1_FDX_s005
                                 + 1.05945 w_k1_IBM_s005
                                 + 0.935578 w_k1_INTC_s005
                                 + 1.08617 w_k1_KO_s005 + 1.17052 w_k1_MAT_s005
                                 + 1.06126 w_k1_MSFT_s005 + 1.131 w_k1_PG_s005
                                 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.2915 w_k1_ADM_s006
                                 + 1.21056 w_k1_AXP_s006
                                 + 1.60206 w_k1_DIS_s006 + 1.3796 w_k1_FDX_s006
                                 + 1.13332 w_k1_IBM_s006
                                 + 1.33388 w_k1_INTC_s006
                                 + 1.33712 w_k1_KO_s006
                                 + 0.536961 w_k1_MAT_s006
                                 + 1.20375 w_k1_MSFT_s006
                                 + 1.21746 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.594075 w_k1_ADM_s006
                                 + 0.807593 w_k1_AXP_s006
                                 + 0.66318 w_k1_DIS_s006
                                 + 0.699477 w_k1_FDX_s006
                                 + 1.21619 w_k1_IBM_s006
                                 + 0.96345 w_k1_INTC_s006
                                 + 1.00615 w_k1_KO_s006
                                 + 0.851737 w_k1_MAT_s006
                                 + 0.762567 w_k1_MSFT_s006
                                 + 0.970131 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.861924 w_k1_ADM_s006
                                 + 0.897393 w_k1_AXP_s006
                                 + 0.649718 w_k1_DIS_s006
                                 + 0.781293 w_k1_FDX_s006
                                 + 0.712584 w_k1_IBM_s006
                                 + 1.04597 w_k1_INTC_s006
                                 + 0.876688 w_k1_KO_s006
                                 + 0.539207 w_k1_MAT_s006
                                 + 0.733035 w_k1_MSFT_s006
                                 + 0.840592 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.783112 w_k1_ADM_s006
                                 + 0.752278 w_k1_AXP_s006
                                 + 0.884566 w_k1_DIS_s006
                                 + 0.830023 w_k1_FDX_s006
                                 + 1.0332 w_k1_IBM_s006
                                 + 0.83726 w_k1_INTC_s006
                                 + 1.02201 w_k1_KO_s006
                                 + 0.870423 w_k1_MAT_s006
                                 + 0.825178 w_k1_MSFT_s006
                                 + 0.949264 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.16975 w_k1_ADM_s006
                                 + 1.31476 w_k1_AXP_s006
                                 + 1.12247 w_k1_DIS_s006
                                 + 1.17124 w_k1_FDX_s006
                                 + 1.09109 w_k1_IBM_s006
                                 + 1.21932 w_k1_INTC_s006
                                 + 1.07574 w_k1_KO_s006
                                 + 0.697541 w_k1_MAT_s006
                                 + 1.00619 w_k1_MSFT_s006
                                 + 1.11503 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.07485 w_k1_ADM_s006
                                 + 1.199 w_k1_AXP_s006 + 1.29061 w_k1_DIS_s006
                                 + 1.02999 w_k1_FDX_s006 + 1.5754 w_k1_IBM_s006
                                 + 1.22138 w_k1_INTC_s006
                                 + 1.44491 w_k1_KO_s006 + 1.07767 w_k1_MAT_s006
                                 + 1.0055 w_k1_MSFT_s006 + 1.13323 w_k1_PG_s006
                                 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.53211 w_k1_ADM_s006
                                 + 1.00955 w_k1_AXP_s006
                                 + 0.742843 w_k1_DIS_s006
                                 + 0.804506 w_k1_FDX_s006
                                 + 1.1796 w_k1_IBM_s006
                                 + 0.885519 w_k1_INTC_s006
                                 + 0.910699 w_k1_KO_s006
                                 + 1.09271 w_k1_MAT_s006
                                 + 0.720019 w_k1_MSFT_s006
                                 + 0.955665 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.05398 w_k1_ADM_s006
                                 + 0.919993 w_k1_AXP_s006
                                 + 0.86607 w_k1_DIS_s006
                                 + 1.07118 w_k1_FDX_s006
                                 + 0.954882 w_k1_IBM_s006
                                 + 0.96589 w_k1_INTC_s006
                                 + 1.03172 w_k1_KO_s006
                                 + 0.878877 w_k1_MAT_s006
                                 + 0.889519 w_k1_MSFT_s006
                                 + 1.00404 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.04081 w_k1_ADM_s007
                                 + 1.00555 w_k1_AXP_s007
                                 + 0.884021 w_k1_DIS_s007
                                 + 0.987901 w_k1_FDX_s007
                                 + 0.734335 w_k1_IBM_s007
                                 + 0.976835 w_k1_INTC_s007
                                 + 0.888392 w_k1_KO_s007
                                 + 0.712262 w_k1_MAT_s007
                                 + 1.04477 w_k1_MSFT_s007
                                 + 0.934434 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.10905 w_k1_ADM_s007
                                 + 0.98865 w_k1_AXP_s007
                                 + 0.930933 w_k1_DIS_s007
                                 + 0.940192 w_k1_FDX_s007
                                 + 1.23681 w_k1_IBM_s007
                                 + 1.01523 w_k1_INTC_s007
                                 + 1.10098 w_k1_KO_s007 + 1.1085 w_k1_MAT_s007
                                 + 1.04496 w_k1_MSFT_s007
                                 + 1.08188 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.04975 w_k1_ADM_s007
                                 + 0.911673 w_k1_AXP_s007
                                 + 0.948843 w_k1_DIS_s007
                                 + 0.913406 w_k1_FDX_s007
                                 + 0.92775 w_k1_IBM_s007
                                 + 1.07292 w_k1_INTC_s007 + 0.8985 w_k1_KO_s007
                                 + 0.773266 w_k1_MAT_s007
                                 + 1.02127 w_k1_MSFT_s007
                                 + 0.921834 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.0045 w_k1_ADM_s007
                                 + 1.25952 w_k1_AXP_s007
                                 + 1.51812 w_k1_DIS_s007
                                 + 1.10417 w_k1_FDX_s007
                                 + 1.80159 w_k1_IBM_s007
                                 + 1.26392 w_k1_INTC_s007
                                 + 1.38881 w_k1_KO_s007 + 1.62801 w_k1_MAT_s007
                                 + 1.13661 w_k1_MSFT_s007
                                 + 1.01633 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.71469 w_k1_ADM_s007
                                 + 0.687492 w_k1_AXP_s007
                                 + 0.748448 w_k1_DIS_s007
                                 + 0.684653 w_k1_FDX_s007
                                 + 1.22172 w_k1_IBM_s007
                                 + 0.89808 w_k1_INTC_s007
                                 + 0.939564 w_k1_KO_s007
                                 + 0.834715 w_k1_MAT_s007
                                 + 0.797521 w_k1_MSFT_s007
                                 + 0.930519 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.20157 w_k1_ADM_s007
                                 + 1.2392 w_k1_AXP_s007 + 1.11837 w_k1_DIS_s007
                                 + 1.13259 w_k1_FDX_s007
                                 + 0.925001 w_k1_IBM_s007
                                 + 0.86409 w_k1_INTC_s007
                                 + 1.10153 w_k1_KO_s007
                                 + 0.953446 w_k1_MAT_s007
                                 + 1.07428 w_k1_MSFT_s007
                                 + 1.00573 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.43804 w_k1_ADM_s007
                                 + 1.37422 w_k1_AXP_s007
                                 + 1.56198 w_k1_DIS_s007
                                 + 1.25897 w_k1_FDX_s007
                                 + 1.19599 w_k1_IBM_s007
                                 + 1.03927 w_k1_INTC_s007
                                 + 1.25544 w_k1_KO_s007 + 1.32045 w_k1_MAT_s007
                                 + 1.24764 w_k1_MSFT_s007
                                 + 1.07178 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.0154 w_k1_ADM_s007
                                 + 0.547267 w_k1_AXP_s007
                                 + 0.600546 w_k1_DIS_s007
                                 + 0.728009 w_k1_FDX_s007
                                 + 0.685715 w_k1_IBM_s007
                                 + 0.69658 w_k1_INTC_s007
                                 + 0.701617 w_k1_KO_s007
                                 + 0.580572 w_k1_MAT_s007
                                 + 0.784782 w_k1_MSFT_s007
                                 + 0.797938 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.82592 w_k1_ADM_s010
                                 + 0.774962 w_k1_AXP_s010
                                 + 0.940326 w_k1_DIS_s010
                                 + 0.931274 w_k1_FDX_s010
                                 + 0.789414 w_k1_IBM_s010
                                 + 1.12174 w_k1_INTC_s010
                                 + 0.860348 w_k1_KO_s010
                                 + 0.630323 w_k1_MAT_s010
                                 + 0.997541 w_k1_MSFT_s010
                                 + 0.791455 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.01649 w_k1_ADM_s010
                                 + 1.00077 w_k1_AXP_s010
                                 + 0.836792 w_k1_DIS_s010
                                 + 0.802783 w_k1_FDX_s010
                                 + 0.949026 w_k1_IBM_s010
                                 + 0.858136 w_k1_INTC_s010
                                 + 0.927036 w_k1_KO_s010
                                 + 1.36945 w_k1_MAT_s010
                                 + 0.794893 w_k1_MSFT_s010
                                 + 0.972001 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.41867 w_k1_ADM_s010
                                 + 1.63062 w_k1_AXP_s010
                                 + 1.64489 w_k1_DIS_s010
                                 + 1.16615 w_k1_FDX_s010
                                 + 1.49436 w_k1_IBM_s010
                                 + 1.31761 w_k1_INTC_s010
                                 + 1.67554 w_k1_KO_s010 + 2.04213 w_k1_MAT_s010
                                 + 1.26111 w_k1_MSFT_s010
                                 + 1.18793 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.993181 w_k1_ADM_s010
                                 + 0.744966 w_k1_AXP_s010
                                 + 0.914602 w_k1_DIS_s010
                                 + 0.965175 w_k1_FDX_s010
                                 + 0.626306 w_k1_IBM_s010
                                 + 0.990529 w_k1_INTC_s010
                                 + 0.820898 w_k1_KO_s010
                                 + 0.778686 w_k1_MAT_s010
                                 + 1.08518 w_k1_MSFT_s010
                                 + 0.837556 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.11555 w_k1_ADM_s010
                                 + 0.975678 w_k1_AXP_s010
                                 + 1.03554 w_k1_DIS_s010
                                 + 0.906614 w_k1_FDX_s010
                                 + 1.42113 w_k1_IBM_s010
                                 + 0.942407 w_k1_INTC_s010
                                 + 1.02389 w_k1_KO_s010 + 1.2227 w_k1_MAT_s010
                                 + 0.891412 w_k1_MSFT_s010
                                 + 0.985907 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.770474 w_k1_ADM_s010
                                 + 0.651087 w_k1_AXP_s010
                                 + 0.565209 w_k1_DIS_s010
                                 + 0.697317 w_k1_FDX_s010
                                 + 0.754319 w_k1_IBM_s010
                                 + 0.81146 w_k1_INTC_s010
                                 + 0.764079 w_k1_KO_s010
                                 + 0.47257 w_k1_MAT_s010
                                 + 0.716976 w_k1_MSFT_s010
                                 + 0.819818 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.963547 w_k1_ADM_s010
                                 + 1.0382 w_k1_AXP_s010
                                 + 0.793574 w_k1_DIS_s010
                                 + 0.98367 w_k1_FDX_s010
                                 + 0.843668 w_k1_IBM_s010
                                 + 0.820273 w_k1_INTC_s010
                                 + 0.803903 w_k1_KO_s010
                                 + 1.23686 w_k1_MAT_s010
                                 + 0.876084 w_k1_MSFT_s010
                                 + 0.893365 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.35795 w_k1_ADM_s010
                                 + 1.20691 w_k1_AXP_s010
                                 + 1.19526 w_k1_DIS_s010
                                 + 1.19196 w_k1_FDX_s010
                                 + 1.03379 w_k1_IBM_s010
                                 + 1.04178 w_k1_INTC_s010
                                 + 1.10196 w_k1_KO_s010 + 1.19604 w_k1_MAT_s010
                                 + 1.08889 w_k1_MSFT_s010
                                 + 1.05246 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.11662 w_k1_ADM_s011
                                 + 1.31542 w_k1_AXP_s011
                                 + 1.49601 w_k1_DIS_s011
                                 + 1.25509 w_k1_FDX_s011
                                 + 1.39935 w_k1_IBM_s011
                                 + 1.7891 w_k1_INTC_s011 + 1.48631 w_k1_KO_s011
                                 + 1.25845 w_k1_MAT_s011
                                 + 1.48381 w_k1_MSFT_s011
                                 + 1.17112 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.651905 w_k1_ADM_s011
                                 + 0.822755 w_k1_AXP_s011
                                 + 0.755684 w_k1_DIS_s011
                                 + 0.783563 w_k1_FDX_s011
                                 + 1.07813 w_k1_IBM_s011
                                 + 1.00651 w_k1_INTC_s011
                                 + 0.973777 w_k1_KO_s011
                                 + 0.921218 w_k1_MAT_s011
                                 + 0.8216 w_k1_MSFT_s011
                                 + 0.868478 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.836695 w_k1_ADM_s011
                                 + 0.918815 w_k1_AXP_s011
                                 + 1.19709 w_k1_DIS_s011
                                 + 1.08502 w_k1_FDX_s011
                                 + 1.12527 w_k1_IBM_s011
                                 + 1.17245 w_k1_INTC_s011
                                 + 1.20146 w_k1_KO_s011
                                 + 0.776257 w_k1_MAT_s011
                                 + 1.15048 w_k1_MSFT_s011
                                 + 1.03908 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.741192 w_k1_ADM_s011
                                 + 0.646338 w_k1_AXP_s011
                                 + 0.992941 w_k1_DIS_s011
                                 + 0.893891 w_k1_FDX_s011
                                 + 0.793717 w_k1_IBM_s011
                                 + 1.13426 w_k1_INTC_s011
                                 + 0.883524 w_k1_KO_s011
                                 + 0.502212 w_k1_MAT_s011
                                 + 0.985692 w_k1_MSFT_s011
                                 + 0.805144 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.94168 w_k1_ADM_s011
                                 + 0.918417 w_k1_AXP_s011
                                 + 0.863977 w_k1_DIS_s011
                                 + 0.994154 w_k1_FDX_s011
                                 + 0.820899 w_k1_IBM_s011
                                 + 0.906286 w_k1_INTC_s011
                                 + 0.728921 w_k1_KO_s011
                                 + 0.669377 w_k1_MAT_s011
                                 + 0.849226 w_k1_MSFT_s011
                                 + 0.94646 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.833015 w_k1_ADM_s011
                                 + 1.22985 w_k1_AXP_s011
                                 + 1.04258 w_k1_DIS_s011
                                 + 1.08327 w_k1_FDX_s011
                                 + 1.61782 w_k1_IBM_s011
                                 + 1.13865 w_k1_INTC_s011
                                 + 1.19574 w_k1_KO_s011 + 1.30376 w_k1_MAT_s011
                                 + 0.994528 w_k1_MSFT_s011
                                 + 0.990088 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.78421 w_k1_ADM_s011
                                 + 1.15773 w_k1_AXP_s011
                                 + 1.02726 w_k1_DIS_s011
                                 + 1.08331 w_k1_FDX_s011
                                 + 1.12408 w_k1_IBM_s011
                                 + 1.11579 w_k1_INTC_s011 + 1.1483 w_k1_KO_s011
                                 + 1.26047 w_k1_MAT_s011
                                 + 1.05474 w_k1_MSFT_s011
                                 + 0.906524 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.926454 w_k1_ADM_s011
                                 + 1.03635 w_k1_AXP_s011
                                 + 1.23723 w_k1_DIS_s011
                                 + 1.22502 w_k1_FDX_s011
                                 + 1.01537 w_k1_IBM_s011
                                 + 1.29112 w_k1_INTC_s011
                                 + 1.05092 w_k1_KO_s011
                                 + 0.813099 w_k1_MAT_s011
                                 + 1.24394 w_k1_MSFT_s011
                                 + 1.02783 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.26891 w_k1_ADM_s012
                                 + 1.60443 w_k1_AXP_s012
                                 + 1.74237 w_k1_DIS_s012
                                 + 1.41315 w_k1_FDX_s012
                                 + 1.07406 w_k1_IBM_s012
                                 + 1.21435 w_k1_INTC_s012
                                 + 1.59818 w_k1_KO_s012 + 1.77551 w_k1_MAT_s012
                                 + 1.48018 w_k1_MSFT_s012
                                 + 1.27336 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.733245 w_k1_ADM_s012
                                 + 0.62689 w_k1_AXP_s012
                                 + 0.553073 w_k1_DIS_s012
                                 + 0.63204 w_k1_FDX_s012
                                 + 0.786932 w_k1_IBM_s012
                                 + 0.716791 w_k1_INTC_s012
                                 + 0.906453 w_k1_KO_s012
                                 + 0.826646 w_k1_MAT_s012
                                 + 0.813324 w_k1_MSFT_s012
                                 + 0.804102 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.801663 w_k1_ADM_s012
                                 + 0.956622 w_k1_AXP_s012
                                 + 1.07448 w_k1_DIS_s012
                                 + 0.931095 w_k1_FDX_s012
                                 + 1.62191 w_k1_IBM_s012
                                 + 0.923664 w_k1_INTC_s012
                                 + 1.36669 w_k1_KO_s012 + 1.46324 w_k1_MAT_s012
                                 + 1.05868 w_k1_MSFT_s012
                                 + 1.14174 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.08927 w_k1_ADM_s012
                                 + 0.996627 w_k1_AXP_s012
                                 + 1.25482 w_k1_DIS_s012
                                 + 1.03863 w_k1_FDX_s012
                                 + 1.26354 w_k1_IBM_s012
                                 + 1.04843 w_k1_INTC_s012
                                 + 1.26649 w_k1_KO_s012 + 1.08862 w_k1_MAT_s012
                                 + 1.06547 w_k1_MSFT_s012
                                 + 1.08291 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.16267 w_k1_ADM_s012
                                 + 1.20517 w_k1_AXP_s012 + 1.2378 w_k1_DIS_s012
                                 + 1.25751 w_k1_FDX_s012
                                 + 0.791726 w_k1_IBM_s012
                                 + 1.20818 w_k1_INTC_s012
                                 + 1.07877 w_k1_KO_s012
                                 + 0.835877 w_k1_MAT_s012
                                 + 1.27811 w_k1_MSFT_s012
                                 + 1.05028 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.92802 w_k1_ADM_s012
                                 + 1.13081 w_k1_AXP_s012
                                 + 0.886206 w_k1_DIS_s012
                                 + 0.918386 w_k1_FDX_s012
                                 + 1.02393 w_k1_IBM_s012
                                 + 0.881208 w_k1_INTC_s012
                                 + 0.992396 w_k1_KO_s012
                                 + 1.14921 w_k1_MAT_s012
                                 + 0.971421 w_k1_MSFT_s012
                                 + 1.00912 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.76668 w_k1_ADM_s012
                                 + 0.970091 w_k1_AXP_s012
                                 + 0.796982 w_k1_DIS_s012
                                 + 0.953393 w_k1_FDX_s012
                                 + 0.617137 w_k1_IBM_s012
                                 + 1.08171 w_k1_INTC_s012
                                 + 0.868626 w_k1_KO_s012
                                 + 0.836695 w_k1_MAT_s012
                                 + 1.03074 w_k1_MSFT_s012
                                 + 0.968584 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.15468 w_k1_ADM_s012
                                 + 0.78505 w_k1_AXP_s012
                                 + 0.904945 w_k1_DIS_s012
                                 + 0.808091 w_k1_FDX_s012
                                 + 0.889236 w_k1_IBM_s012
                                 + 0.996586 w_k1_INTC_s012
                                 + 1.09657 w_k1_KO_s012
                                 + 0.829934 w_k1_MAT_s012
                                 + 1.02473 w_k1_MSFT_s012
                                 + 0.949421 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.813748 w_k1_ADM_s013
                                 + 0.671359 w_k1_AXP_s013
                                 + 0.540122 w_k1_DIS_s013
                                 + 0.77302 w_k1_FDX_s013
                                 + 0.710259 w_k1_IBM_s013
                                 + 0.564832 w_k1_INTC_s013
                                 + 0.679947 w_k1_KO_s013
                                 + 0.761764 w_k1_MAT_s013
                                 + 0.661441 w_k1_MSFT_s013
                                 + 0.776221 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.957672 w_k1_ADM_s013
                                 + 0.964955 w_k1_AXP_s013
                                 + 1.00609 w_k1_DIS_s013
                                 + 1.05292 w_k1_FDX_s013
                                 + 0.769166 w_k1_IBM_s013
                                 + 1.04969 w_k1_INTC_s013
                                 + 0.969623 w_k1_KO_s013
                                 + 0.984463 w_k1_MAT_s013
                                 + 1.06483 w_k1_MSFT_s013
                                 + 1.00154 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.03989 w_k1_ADM_s013
                                 + 1.20959 w_k1_AXP_s013
                                 + 1.30494 w_k1_DIS_s013 + 1.2372 w_k1_FDX_s013
                                 + 1.09623 w_k1_IBM_s013
                                 + 1.13439 w_k1_INTC_s013
                                 + 1.18417 w_k1_KO_s013 + 1.1534 w_k1_MAT_s013
                                 + 1.14004 w_k1_MSFT_s013
                                 + 1.22905 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.839635 w_k1_ADM_s013
                                 + 0.664656 w_k1_AXP_s013
                                 + 0.774202 w_k1_DIS_s013
                                 + 0.775185 w_k1_FDX_s013
                                 + 1.00516 w_k1_IBM_s013
                                 + 0.750319 w_k1_INTC_s013
                                 + 0.870732 w_k1_KO_s013
                                 + 1.16189 w_k1_MAT_s013
                                 + 0.88286 w_k1_MSFT_s013
                                 + 0.964842 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.03879 w_k1_ADM_s013
                                 + 1.1295 w_k1_AXP_s013 + 1.14873 w_k1_DIS_s013
                                 + 1.25498 w_k1_FDX_s013
                                 + 0.725827 w_k1_IBM_s013
                                 + 1.03813 w_k1_INTC_s013
                                 + 0.886217 w_k1_KO_s013
                                 + 0.957417 w_k1_MAT_s013
                                 + 1.16345 w_k1_MSFT_s013
                                 + 1.02381 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.688391 w_k1_ADM_s013
                                 + 0.882536 w_k1_AXP_s013
                                 + 0.811519 w_k1_DIS_s013
                                 + 0.767485 w_k1_FDX_s013
                                 + 1.35431 w_k1_IBM_s013
                                 + 0.973537 w_k1_INTC_s013
                                 + 1.01707 w_k1_KO_s013 + 1.20836 w_k1_MAT_s013
                                 + 0.841796 w_k1_MSFT_s013
                                 + 0.962716 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.95583 w_k1_ADM_s013
                                 + 1.53627 w_k1_AXP_s013
                                 + 1.20118 w_k1_DIS_s013
                                 + 1.24088 w_k1_FDX_s013
                                 + 1.64543 w_k1_IBM_s013
                                 + 1.02378 w_k1_INTC_s013
                                 + 1.53262 w_k1_KO_s013 + 2.88131 w_k1_MAT_s013
                                 + 1.12883 w_k1_MSFT_s013
                                 + 1.20848 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.642316 w_k1_ADM_s013
                                 + 0.726406 w_k1_AXP_s013
                                 + 0.866736 w_k1_DIS_s013
                                 + 0.935959 w_k1_FDX_s013
                                 + 0.682304 w_k1_IBM_s013
                                 + 1.0823 w_k1_INTC_s013
                                 + 0.840876 w_k1_KO_s013
                                 + 0.694845 w_k1_MAT_s013
                                 + 1.09034 w_k1_MSFT_s013
                                 + 0.896127 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.14225 w_k1_ADM_s014
                                 + 1.27348 w_k1_AXP_s014 + 1.5765 w_k1_DIS_s014
                                 + 1.32769 w_k1_FDX_s014
                                 + 1.01118 w_k1_IBM_s014
                                 + 1.11445 w_k1_INTC_s014 + 1.2368 w_k1_KO_s014
                                 + 0.96686 w_k1_MAT_s014
                                 + 1.26142 w_k1_MSFT_s014
                                 + 1.04766 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.971936 w_k1_ADM_s014
                                 + 1.30933 w_k1_AXP_s014
                                 + 1.51434 w_k1_DIS_s014
                                 + 1.18703 w_k1_FDX_s014
                                 + 1.56183 w_k1_IBM_s014
                                 + 1.1652 w_k1_INTC_s014 + 1.37505 w_k1_KO_s014
                                 + 1.44665 w_k1_MAT_s014
                                 + 1.21294 w_k1_MSFT_s014
                                 + 1.06293 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.827553 w_k1_ADM_s014
                                 + 0.771555 w_k1_AXP_s014
                                 + 0.88353 w_k1_DIS_s014
                                 + 0.766984 w_k1_FDX_s014
                                 + 1.09029 w_k1_IBM_s014
                                 + 0.797735 w_k1_INTC_s014
                                 + 0.968372 w_k1_KO_s014
                                 + 0.810216 w_k1_MAT_s014
                                 + 0.818373 w_k1_MSFT_s014
                                 + 0.864737 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.20074 w_k1_ADM_s014
                                 + 1.13066 w_k1_AXP_s014
                                 + 1.15597 w_k1_DIS_s014
                                 + 1.11642 w_k1_FDX_s014
                                 + 0.966408 w_k1_IBM_s014
                                 + 0.932473 w_k1_INTC_s014
                                 + 1.06289 w_k1_KO_s014
                                 + 0.584332 w_k1_MAT_s014
                                 + 0.934676 w_k1_MSFT_s014
                                 + 1.07035 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.830695 w_k1_ADM_s014
                                 + 0.939679 w_k1_AXP_s014
                                 + 0.967235 w_k1_DIS_s014
                                 + 1.01472 w_k1_FDX_s014
                                 + 1.35149 w_k1_IBM_s014
                                 + 0.960924 w_k1_INTC_s014
                                 + 1.15562 w_k1_KO_s014
                                 + 0.780602 w_k1_MAT_s014
                                 + 1.09251 w_k1_MSFT_s014
                                 + 1.20346 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.840792 w_k1_ADM_s014
                                 + 1.22512 w_k1_AXP_s014 + 1.0304 w_k1_DIS_s014
                                 + 0.98081 w_k1_FDX_s014
                                 + 1.12378 w_k1_IBM_s014
                                 + 0.974691 w_k1_INTC_s014
                                 + 1.02191 w_k1_KO_s014
                                 + 0.939309 w_k1_MAT_s014
                                 + 1.03683 w_k1_MSFT_s014
                                 + 1.04866 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.76681 w_k1_ADM_s014
                                 + 0.643221 w_k1_AXP_s014
                                 + 0.725591 w_k1_DIS_s014
                                 + 0.757856 w_k1_FDX_s014
                                 + 0.763322 w_k1_IBM_s014
                                 + 0.730393 w_k1_INTC_s014
                                 + 0.767682 w_k1_KO_s014
                                 + 0.590288 w_k1_MAT_s014
                                 + 0.680183 w_k1_MSFT_s014
                                 + 0.793989 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.10261 w_k1_ADM_s014
                                 + 0.926394 w_k1_AXP_s014
                                 + 0.812882 w_k1_DIS_s014
                                 + 0.991326 w_k1_FDX_s014
                                 + 0.862629 w_k1_IBM_s014
                                 + 0.868876 w_k1_INTC_s014
                                 + 0.822136 w_k1_KO_s014
                                 + 0.513813 w_k1_MAT_s014
                                 + 0.968382 w_k1_MSFT_s014
                                 + 0.897572 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.68901 w_k1_ADM_s015
                                 + 1.33317 w_k1_AXP_s015
                                 + 1.49675 w_k1_DIS_s015
                                 + 1.29583 w_k1_FDX_s015
                                 + 1.36569 w_k1_IBM_s015
                                 + 1.12125 w_k1_INTC_s015
                                 + 1.62182 w_k1_KO_s015 + 1.53381 w_k1_MAT_s015
                                 + 1.31797 w_k1_MSFT_s015 + 1.389 w_k1_PG_s015
                                 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.25139 w_k1_ADM_s015
                                 + 1.02435 w_k1_AXP_s015
                                 + 1.15102 w_k1_DIS_s015
                                 + 1.19566 w_k1_FDX_s015
                                 + 0.978918 w_k1_IBM_s015
                                 + 1.30381 w_k1_INTC_s015
                                 + 1.25868 w_k1_KO_s015
                                 + 0.936018 w_k1_MAT_s015
                                 + 1.25863 w_k1_MSFT_s015
                                 + 1.01249 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.00313 w_k1_ADM_s015
                                 + 0.677706 w_k1_AXP_s015
                                 + 0.696319 w_k1_DIS_s015
                                 + 0.833035 w_k1_FDX_s015
                                 + 0.65184 w_k1_IBM_s015
                                 + 0.944281 w_k1_INTC_s015
                                 + 0.870558 w_k1_KO_s015
                                 + 0.550755 w_k1_MAT_s015
                                 + 0.936769 w_k1_MSFT_s015
                                 + 0.827793 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.74677 w_k1_ADM_s015
                                 + 1.24294 w_k1_AXP_s015 + 1.1005 w_k1_DIS_s015
                                 + 1.29594 w_k1_FDX_s015
                                 + 0.607751 w_k1_IBM_s015
                                 + 1.13055 w_k1_INTC_s015
                                 + 0.97986 w_k1_KO_s015
                                 + 0.700381 w_k1_MAT_s015
                                 + 1.11455 w_k1_MSFT_s015
                                 + 1.12215 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.0605 w_k1_ADM_s015
                                 + 1.10399 w_k1_AXP_s015
                                 + 0.888944 w_k1_DIS_s015
                                 + 0.977734 w_k1_FDX_s015
                                 + 0.90662 w_k1_IBM_s015
                                 + 0.834399 w_k1_INTC_s015
                                 + 1.17041 w_k1_KO_s015 + 1.02868 w_k1_MAT_s015
                                 + 0.953206 w_k1_MSFT_s015
                                 + 1.1101 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.24208 w_k1_ADM_s015
                                 + 1.46188 w_k1_AXP_s015 + 1.1182 w_k1_DIS_s015
                                 + 1.14635 w_k1_FDX_s015
                                 + 1.19366 w_k1_IBM_s015
                                 + 1.23777 w_k1_INTC_s015
                                 + 1.31452 w_k1_KO_s015 + 1.49332 w_k1_MAT_s015
                                 + 1.1502 w_k1_MSFT_s015 + 1.13543 w_k1_PG_s015
                                 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.889951 w_k1_ADM_s015
                                 + 0.841904 w_k1_AXP_s015
                                 + 1.0493 w_k1_DIS_s015
                                 + 0.957263 w_k1_FDX_s015
                                 + 1.03606 w_k1_IBM_s015
                                 + 1.28348 w_k1_INTC_s015
                                 + 1.23278 w_k1_KO_s015 + 0.91623 w_k1_MAT_s015
                                 + 1.22328 w_k1_MSFT_s015
                                 + 1.08785 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.1586 w_k1_ADM_s015
                                 + 1.02097 w_k1_AXP_s015
                                 + 1.19322 w_k1_DIS_s015
                                 + 0.923277 w_k1_FDX_s015
                                 + 1.39013 w_k1_IBM_s015
                                 + 1.15463 w_k1_INTC_s015
                                 + 1.51686 w_k1_KO_s015 + 1.59546 w_k1_MAT_s015
                                 + 1.05032 w_k1_MSFT_s015
                                 + 1.06332 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.06811 w_k1_ADM_s016
                                 + 0.815164 w_k1_AXP_s016
                                 + 1.10671 w_k1_DIS_s016
                                 + 1.02657 w_k1_FDX_s016
                                 + 0.875215 w_k1_IBM_s016
                                 + 0.867818 w_k1_INTC_s016
                                 + 0.982058 w_k1_KO_s016
                                 + 0.919821 w_k1_MAT_s016
                                 + 1.23384 w_k1_MSFT_s016
                                 + 1.10756 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.15274 w_k1_ADM_s016
                                 + 0.994555 w_k1_AXP_s016
                                 + 1.24121 w_k1_DIS_s016
                                 + 1.16997 w_k1_FDX_s016
                                 + 0.698612 w_k1_IBM_s016
                                 + 1.04301 w_k1_INTC_s016
                                 + 0.945851 w_k1_KO_s016
                                 + 0.661177 w_k1_MAT_s016
                                 + 1.1583 w_k1_MSFT_s016
                                 + 0.851628 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.889054 w_k1_ADM_s016
                                 + 1.27986 w_k1_AXP_s016
                                 + 1.39417 w_k1_DIS_s016
                                 + 1.20472 w_k1_FDX_s016 + 1.0911 w_k1_IBM_s016
                                 + 1.28818 w_k1_INTC_s016
                                 + 1.06342 w_k1_KO_s016 + 1.33208 w_k1_MAT_s016
                                 + 1.19175 w_k1_MSFT_s016
                                 + 1.14839 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.735488 w_k1_ADM_s016
                                 + 0.653433 w_k1_AXP_s016
                                 + 0.691275 w_k1_DIS_s016 + 0.828 w_k1_FDX_s016
                                 + 0.753153 w_k1_IBM_s016
                                 + 0.861731 w_k1_INTC_s016
                                 + 0.744991 w_k1_KO_s016
                                 + 0.645662 w_k1_MAT_s016
                                 + 0.900195 w_k1_MSFT_s016
                                 + 0.91324 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.759472 w_k1_ADM_s016
                                 + 0.820986 w_k1_AXP_s016
                                 + 0.692369 w_k1_DIS_s016
                                 + 0.891711 w_k1_FDX_s016
                                 + 0.554858 w_k1_IBM_s016
                                 + 0.80695 w_k1_INTC_s016
                                 + 0.645668 w_k1_KO_s016
                                 + 0.827038 w_k1_MAT_s016
                                 + 0.758616 w_k1_MSFT_s016
                                 + 0.977729 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.31494 w_k1_ADM_s016
                                 + 1.55841 w_k1_AXP_s016 + 1.5427 w_k1_DIS_s016
                                 + 1.25401 w_k1_FDX_s016
                                 + 1.89694 w_k1_IBM_s016
                                 + 1.03885 w_k1_INTC_s016
                                 + 1.41066 w_k1_KO_s016 + 2.19309 w_k1_MAT_s016
                                 + 1.16374 w_k1_MSFT_s016
                                 + 1.39572 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.853968 w_k1_ADM_s016
                                 + 1.15448 w_k1_AXP_s016
                                 + 0.870817 w_k1_DIS_s016
                                 + 0.990102 w_k1_FDX_s016
                                 + 0.83171 w_k1_IBM_s016
                                 + 1.05337 w_k1_INTC_s016
                                 + 0.853444 w_k1_KO_s016
                                 + 1.02561 w_k1_MAT_s016
                                 + 0.951811 w_k1_MSFT_s016
                                 + 0.935166 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.96866 w_k1_ADM_s016
                                 + 0.724683 w_k1_AXP_s016
                                 + 0.828935 w_k1_DIS_s016
                                 + 0.76259 w_k1_FDX_s016
                                 + 1.10012 w_k1_IBM_s016
                                 + 0.684046 w_k1_INTC_s016
                                 + 0.96006 w_k1_KO_s016
                                 + 0.831426 w_k1_MAT_s016
                                 + 0.849853 w_k1_MSFT_s016
                                 + 1.0165 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.905466 w_k1_ADM_s017
                                 + 0.834811 w_k1_AXP_s017
                                 + 0.835722 w_k1_DIS_s017
                                 + 0.830403 w_k1_FDX_s017
                                 + 0.957389 w_k1_IBM_s017
                                 + 0.815362 w_k1_INTC_s017
                                 + 0.961411 w_k1_KO_s017
                                 + 0.99158 w_k1_MAT_s017
                                 + 0.810029 w_k1_MSFT_s017
                                 + 0.986106 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.12731 w_k1_ADM_s017
                                 + 1.19589 w_k1_AXP_s017
                                 + 0.913238 w_k1_DIS_s017
                                 + 1.09949 w_k1_FDX_s017
                                 + 0.832648 w_k1_IBM_s017
                                 + 1.13559 w_k1_INTC_s017
                                 + 0.768908 w_k1_KO_s017
                                 + 0.711759 w_k1_MAT_s017
                                 + 1.01434 w_k1_MSFT_s017 + 1.0424 w_k1_PG_s017
                                 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.704565 w_k1_ADM_s017
                                 + 0.589827 w_k1_AXP_s017
                                 + 0.616017 w_k1_DIS_s017
                                 + 0.77681 w_k1_FDX_s017
                                 + 0.628201 w_k1_IBM_s017
                                 + 0.715061 w_k1_INTC_s017
                                 + 0.665168 w_k1_KO_s017
                                 + 0.630828 w_k1_MAT_s017
                                 + 0.738203 w_k1_MSFT_s017
                                 + 0.833595 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.30218 w_k1_ADM_s017
                                 + 1.05087 w_k1_AXP_s017
                                 + 1.67886 w_k1_DIS_s017
                                 + 1.30043 w_k1_FDX_s017
                                 + 1.09997 w_k1_IBM_s017
                                 + 0.997314 w_k1_INTC_s017
                                 + 1.2833 w_k1_KO_s017 + 0.887756 w_k1_MAT_s017
                                 + 1.14613 w_k1_MSFT_s017
                                 + 1.24365 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.08976 w_k1_ADM_s017
                                 + 0.958369 w_k1_AXP_s017
                                 + 1.09038 w_k1_DIS_s017
                                 + 1.11864 w_k1_FDX_s017
                                 + 0.785522 w_k1_IBM_s017
                                 + 0.986333 w_k1_INTC_s017
                                 + 0.895586 w_k1_KO_s017
                                 + 0.619428 w_k1_MAT_s017
                                 + 0.978039 w_k1_MSFT_s017
                                 + 0.95989 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.794708 w_k1_ADM_s017
                                 + 0.669056 w_k1_AXP_s017
                                 + 0.615712 w_k1_DIS_s017
                                 + 0.775758 w_k1_FDX_s017
                                 + 1.22946 w_k1_IBM_s017
                                 + 0.829685 w_k1_INTC_s017
                                 + 0.806118 w_k1_KO_s017
                                 + 0.799053 w_k1_MAT_s017
                                 + 0.739778 w_k1_MSFT_s017
                                 + 0.86851 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.2374 w_k1_ADM_s017
                                 + 1.6351 w_k1_AXP_s017 + 1.42534 w_k1_DIS_s017
                                 + 1.34815 w_k1_FDX_s017
                                 + 1.25759 w_k1_IBM_s017
                                 + 1.19897 w_k1_INTC_s017
                                 + 1.25214 w_k1_KO_s017 + 1.6697 w_k1_MAT_s017
                                 + 1.21499 w_k1_MSFT_s017
                                 + 1.22293 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.918887 w_k1_ADM_s017
                                 + 0.966487 w_k1_AXP_s017
                                 + 0.974226 w_k1_DIS_s017
                                 + 0.858097 w_k1_FDX_s017 + 1.387 w_k1_IBM_s017
                                 + 0.964791 w_k1_INTC_s017
                                 + 1.15628 w_k1_KO_s017 + 1.18521 w_k1_MAT_s017
                                 + 0.813915 w_k1_MSFT_s017
                                 + 1.09765 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.28524 w_k1_ADM_s100
                                 + 1.3115 w_k1_AXP_s100 + 1.00339 w_k1_DIS_s100
                                 + 1.16445 w_k1_FDX_s100
                                 + 0.836428 w_k1_IBM_s100
                                 + 1.05878 w_k1_INTC_s100
                                 + 0.93185 w_k1_KO_s100
                                 + 0.767143 w_k1_MAT_s100
                                 + 1.03739 w_k1_MSFT_s100
                                 + 0.982956 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.995899 w_k1_ADM_s100
                                 + 0.677149 w_k1_AXP_s100
                                 + 1.04449 w_k1_DIS_s100
                                 + 0.928601 w_k1_FDX_s100
                                 + 0.743226 w_k1_IBM_s100
                                 + 0.855032 w_k1_INTC_s100
                                 + 0.874536 w_k1_KO_s100
                                 + 0.616937 w_k1_MAT_s100
                                 + 1.15232 w_k1_MSFT_s100
                                 + 0.958071 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.06434 w_k1_ADM_s100
                                 + 0.95325 w_k1_AXP_s100
                                 + 1.31308 w_k1_DIS_s100 + 1.1508 w_k1_FDX_s100
                                 + 0.997718 w_k1_IBM_s100
                                 + 1.21105 w_k1_INTC_s100
                                 + 1.10523 w_k1_KO_s100
                                 + 0.554089 w_k1_MAT_s100
                                 + 1.098 w_k1_MSFT_s100 + 1.03705 w_k1_PG_s100
                                 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.314 w_k1_ADM_s100
                                 + 1.20831 w_k1_AXP_s100
                                 + 1.45687 w_k1_DIS_s100
                                 + 1.21424 w_k1_FDX_s100
                                 + 1.07076 w_k1_IBM_s100
                                 + 1.20319 w_k1_INTC_s100
                                 + 1.14688 w_k1_KO_s100
                                 + 0.826111 w_k1_MAT_s100
                                 + 1.38018 w_k1_MSFT_s100
                                 + 1.15108 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.863394 w_k1_ADM_s100
                                 + 0.798326 w_k1_AXP_s100
                                 + 0.72367 w_k1_DIS_s100
                                 + 0.737985 w_k1_FDX_s100
                                 + 0.886731 w_k1_IBM_s100
                                 + 0.790704 w_k1_INTC_s100
                                 + 0.91351 w_k1_KO_s100
                                 + 0.819674 w_k1_MAT_s100
                                 + 0.82343 w_k1_MSFT_s100
                                 + 0.963991 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.3 w_k1_ADM_s100
                                 + 1.30098 w_k1_AXP_s100
                                 + 1.14501 w_k1_DIS_s100
                                 + 1.13688 w_k1_FDX_s100
                                 + 1.33657 w_k1_IBM_s100
                                 + 1.02028 w_k1_INTC_s100
                                 + 1.11988 w_k1_KO_s100 + 1.19398 w_k1_MAT_s100
                                 + 1.12192 w_k1_MSFT_s100 + 1.1751 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.90346 w_k1_ADM_s100
                                 + 1.05461 w_k1_AXP_s100
                                 + 0.975509 w_k1_DIS_s100
                                 + 1.06975 w_k1_FDX_s100
                                 + 0.974269 w_k1_IBM_s100
                                 + 1.20282 w_k1_INTC_s100
                                 + 0.96201 w_k1_KO_s100
                                 + 0.766444 w_k1_MAT_s100
                                 + 1.07467 w_k1_MSFT_s100
                                 + 1.05865 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.06223 w_k1_ADM_s100
                                 + 0.798625 w_k1_AXP_s100
                                 + 0.899431 w_k1_DIS_s100
                                 + 0.861254 w_k1_FDX_s100
                                 + 0.955915 w_k1_IBM_s100
                                 + 0.947186 w_k1_INTC_s100
                                 + 0.984231 w_k1_KO_s100
                                 + 0.532379 w_k1_MAT_s100
                                 + 0.867224 w_k1_MSFT_s100
                                 + 0.924177 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.21483 w_k1_ADM_s101
                                 + 1.34945 w_k1_AXP_s101
                                 + 0.991016 w_k1_DIS_s101
                                 + 1.12634 w_k1_FDX_s101
                                 + 1.00251 w_k1_IBM_s101
                                 + 0.948259 w_k1_INTC_s101
                                 + 1.09589 w_k1_KO_s101 + 1.51121 w_k1_MAT_s101
                                 + 1.10816 w_k1_MSFT_s101
                                 + 0.975492 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.04602 w_k1_ADM_s101
                                 + 0.928473 w_k1_AXP_s101
                                 + 0.953323 w_k1_DIS_s101
                                 + 0.928337 w_k1_FDX_s101
                                 + 0.801182 w_k1_IBM_s101
                                 + 1.03905 w_k1_INTC_s101
                                 + 0.821196 w_k1_KO_s101
                                 + 0.788152 w_k1_MAT_s101
                                 + 0.953999 w_k1_MSFT_s101
                                 + 0.912871 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.86952 w_k1_ADM_s101
                                 + 0.757823 w_k1_AXP_s101
                                 + 0.852093 w_k1_DIS_s101
                                 + 0.804417 w_k1_FDX_s101
                                 + 1.10331 w_k1_IBM_s101
                                 + 0.90568 w_k1_INTC_s101
                                 + 1.09726 w_k1_KO_s101 + 1.45244 w_k1_MAT_s101
                                 + 0.902104 w_k1_MSFT_s101
                                 + 0.968623 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.730395 w_k1_ADM_s101
                                 + 0.961258 w_k1_AXP_s101
                                 + 0.642103 w_k1_DIS_s101
                                 + 0.747931 w_k1_FDX_s101
                                 + 0.926568 w_k1_IBM_s101
                                 + 0.939099 w_k1_INTC_s101
                                 + 0.863715 w_k1_KO_s101
                                 + 1.11341 w_k1_MAT_s101
                                 + 0.727683 w_k1_MSFT_s101
                                 + 0.812217 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.4314 w_k1_ADM_s101
                                 + 1.36472 w_k1_AXP_s101
                                 + 1.43848 w_k1_DIS_s101
                                 + 1.36866 w_k1_FDX_s101
                                 + 1.23205 w_k1_IBM_s101
                                 + 1.11406 w_k1_INTC_s101
                                 + 1.57728 w_k1_KO_s101 + 2.14747 w_k1_MAT_s101
                                 + 1.31006 w_k1_MSFT_s101 + 1.171 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.970826 w_k1_ADM_s101
                                 + 1.25481 w_k1_AXP_s101
                                 + 1.24741 w_k1_DIS_s101
                                 + 1.05366 w_k1_FDX_s101
                                 + 1.32566 w_k1_IBM_s101
                                 + 0.929401 w_k1_INTC_s101
                                 + 1.14165 w_k1_KO_s101 + 1.81045 w_k1_MAT_s101
                                 + 1.05643 w_k1_MSFT_s101
                                 + 1.06095 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.04813 w_k1_ADM_s101
                                 + 1.03237 w_k1_AXP_s101
                                 + 1.37994 w_k1_DIS_s101
                                 + 1.14681 w_k1_FDX_s101
                                 + 0.893156 w_k1_IBM_s101
                                 + 1.0523 w_k1_INTC_s101 + 1.19262 w_k1_KO_s101
                                 + 1.11694 w_k1_MAT_s101
                                 + 1.33089 w_k1_MSFT_s101
                                 + 1.09061 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.971619 w_k1_ADM_s101
                                 + 0.661634 w_k1_AXP_s101
                                 + 0.618018 w_k1_DIS_s101
                                 + 0.74037 w_k1_FDX_s101
                                 + 0.642891 w_k1_IBM_s101
                                 + 0.758873 w_k1_INTC_s101
                                 + 0.694998 w_k1_KO_s101
                                 + 0.569567 w_k1_MAT_s101
                                 + 0.762374 w_k1_MSFT_s101
                                 + 0.766593 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.09059 w_k1_ADM_s102
                                 + 0.787399 w_k1_AXP_s102
                                 + 0.98734 w_k1_DIS_s102
                                 + 1.05606 w_k1_FDX_s102
                                 + 0.760122 w_k1_IBM_s102
                                 + 0.770977 w_k1_INTC_s102
                                 + 0.824118 w_k1_KO_s102
                                 + 0.811189 w_k1_MAT_s102
                                 + 0.96538 w_k1_MSFT_s102
                                 + 0.949882 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.28709 w_k1_ADM_s102
                                 + 1.50716 w_k1_AXP_s102
                                 + 1.33839 w_k1_DIS_s102
                                 + 1.30948 w_k1_FDX_s102
                                 + 1.32139 w_k1_IBM_s102
                                 + 1.3704 w_k1_INTC_s102 + 1.29438 w_k1_KO_s102
                                 + 1.6952 w_k1_MAT_s102
                                 + 1.25761 w_k1_MSFT_s102
                                 + 1.21676 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.05056 w_k1_ADM_s102
                                 + 1.04517 w_k1_AXP_s102
                                 + 1.23549 w_k1_DIS_s102
                                 + 1.08132 w_k1_FDX_s102
                                 + 1.21185 w_k1_IBM_s102
                                 + 0.861114 w_k1_INTC_s102
                                 + 1.13545 w_k1_KO_s102 + 1.56604 w_k1_MAT_s102
                                 + 1.09406 w_k1_MSFT_s102
                                 + 1.16239 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.700993 w_k1_ADM_s102
                                 + 0.694071 w_k1_AXP_s102
                                 + 0.892375 w_k1_DIS_s102
                                 + 0.908281 w_k1_FDX_s102
                                 + 1.00577 w_k1_IBM_s102
                                 + 1.11188 w_k1_INTC_s102
                                 + 0.937887 w_k1_KO_s102
                                 + 0.879197 w_k1_MAT_s102
                                 + 1.03474 w_k1_MSFT_s102
                                 + 1.00441 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.18276 w_k1_ADM_s102
                                 + 1.1502 w_k1_AXP_s102 + 1.64367 w_k1_DIS_s102
                                 + 1.49811 w_k1_FDX_s102
                                 + 1.00586 w_k1_IBM_s102
                                 + 1.29933 w_k1_INTC_s102
                                 + 1.25101 w_k1_KO_s102 + 1.01952 w_k1_MAT_s102
                                 + 1.44944 w_k1_MSFT_s102
                                 + 1.25052 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.796984 w_k1_ADM_s102
                                 + 0.703712 w_k1_AXP_s102
                                 + 0.689706 w_k1_DIS_s102
                                 + 0.816503 w_k1_FDX_s102
                                 + 0.852783 w_k1_IBM_s102
                                 + 0.837237 w_k1_INTC_s102
                                 + 0.783538 w_k1_KO_s102
                                 + 0.851922 w_k1_MAT_s102
                                 + 0.775922 w_k1_MSFT_s102
                                 + 0.841332 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.655982 w_k1_ADM_s102
                                 + 0.533051 w_k1_AXP_s102
                                 + 0.581735 w_k1_DIS_s102
                                 + 0.692328 w_k1_FDX_s102
                                 + 0.863239 w_k1_IBM_s102
                                 + 0.771574 w_k1_INTC_s102
                                 + 0.762045 w_k1_KO_s102
                                 + 0.920415 w_k1_MAT_s102
                                 + 0.734327 w_k1_MSFT_s102
                                 + 0.84054 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.932205 w_k1_ADM_s102
                                 + 0.827893 w_k1_AXP_s102
                                 + 1.1937 w_k1_DIS_s102 + 0.88161 w_k1_FDX_s102
                                 + 1.54581 w_k1_IBM_s102
                                 + 0.855941 w_k1_INTC_s102
                                 + 1.14815 w_k1_KO_s102 + 1.5811 w_k1_MAT_s102
                                 + 0.938487 w_k1_MSFT_s102
                                 + 1.02753 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.04155 w_k1_ADM_s103
                                 + 1.21343 w_k1_AXP_s103
                                 + 1.35498 w_k1_DIS_s103
                                 + 1.22715 w_k1_FDX_s103
                                 + 1.14804 w_k1_IBM_s103
                                 + 1.19118 w_k1_INTC_s103
                                 + 1.03462 w_k1_KO_s103 + 1.03741 w_k1_MAT_s103
                                 + 1.31038 w_k1_MSFT_s103
                                 + 1.14679 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.99821 w_k1_ADM_s103
                                 + 1.04784 w_k1_AXP_s103
                                 + 0.889258 w_k1_DIS_s103
                                 + 1.00116 w_k1_FDX_s103
                                 + 0.780556 w_k1_IBM_s103
                                 + 0.926738 w_k1_INTC_s103
                                 + 0.787144 w_k1_KO_s103
                                 + 0.589524 w_k1_MAT_s103
                                 + 1.00724 w_k1_MSFT_s103
                                 + 0.924145 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.813564 w_k1_ADM_s103
                                 + 0.729569 w_k1_AXP_s103
                                 + 0.610002 w_k1_DIS_s103
                                 + 0.826499 w_k1_FDX_s103
                                 + 0.582981 w_k1_IBM_s103
                                 + 0.733158 w_k1_INTC_s103
                                 + 0.534043 w_k1_KO_s103
                                 + 0.592581 w_k1_MAT_s103
                                 + 0.680614 w_k1_MSFT_s103
                                 + 0.762591 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.02618 w_k1_ADM_s103
                                 + 0.904271 w_k1_AXP_s103
                                 + 0.75045 w_k1_DIS_s103
                                 + 0.860508 w_k1_FDX_s103
                                 + 1.04634 w_k1_IBM_s103
                                 + 0.930961 w_k1_INTC_s103
                                 + 0.898583 w_k1_KO_s103
                                 + 1.00685 w_k1_MAT_s103
                                 + 0.856126 w_k1_MSFT_s103
                                 + 1.0052 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.827157 w_k1_ADM_s103
                                 + 0.873004 w_k1_AXP_s103
                                 + 1.15926 w_k1_DIS_s103
                                 + 0.921504 w_k1_FDX_s103
                                 + 1.15734 w_k1_IBM_s103
                                 + 0.976326 w_k1_INTC_s103
                                 + 0.989555 w_k1_KO_s103
                                 + 0.845165 w_k1_MAT_s103
                                 + 1.02531 w_k1_MSFT_s103
                                 + 0.93288 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.754646 w_k1_ADM_s103
                                 + 0.664614 w_k1_AXP_s103
                                 + 0.742232 w_k1_DIS_s103
                                 + 0.819645 w_k1_FDX_s103
                                 + 0.826026 w_k1_IBM_s103
                                 + 0.807233 w_k1_INTC_s103
                                 + 0.752223 w_k1_KO_s103
                                 + 0.756395 w_k1_MAT_s103
                                 + 0.946376 w_k1_MSFT_s103
                                 + 0.862475 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.769501 w_k1_ADM_s103
                                 + 0.83593 w_k1_AXP_s103
                                 + 0.588364 w_k1_DIS_s103
                                 + 0.663356 w_k1_FDX_s103
                                 + 0.971143 w_k1_IBM_s103
                                 + 0.674461 w_k1_INTC_s103
                                 + 0.716571 w_k1_KO_s103
                                 + 0.818641 w_k1_MAT_s103
                                 + 0.707681 w_k1_MSFT_s103
                                 + 0.909031 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.01132 w_k1_ADM_s103
                                 + 1.23669 w_k1_AXP_s103
                                 + 1.15738 w_k1_DIS_s103
                                 + 1.04029 w_k1_FDX_s103
                                 + 1.30688 w_k1_IBM_s103
                                 + 1.09841 w_k1_INTC_s103
                                 + 0.985712 w_k1_KO_s103
                                 + 0.974265 w_k1_MAT_s103
                                 + 0.874886 w_k1_MSFT_s103
                                 + 1.0435 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.710206 w_k1_ADM_s104
                                 + 0.605935 w_k1_AXP_s104
                                 + 0.783356 w_k1_DIS_s104
                                 + 0.753999 w_k1_FDX_s104
                                 + 0.8595 w_k1_IBM_s104
                                 + 0.927599 w_k1_INTC_s104
                                 + 1.01415 w_k1_KO_s104
                                 + 0.719833 w_k1_MAT_s104
                                 + 0.88697 w_k1_MSFT_s104
                                 + 0.831892 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.3825 w_k1_ADM_s104
                                 + 1.55171 w_k1_AXP_s104
                                 + 1.57199 w_k1_DIS_s104
                                 + 1.35146 w_k1_FDX_s104
                                 + 0.928448 w_k1_IBM_s104
                                 + 1.31848 w_k1_INTC_s104
                                 + 1.22284 w_k1_KO_s104 + 1.35312 w_k1_MAT_s104
                                 + 1.33177 w_k1_MSFT_s104
                                 + 1.16106 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.03459 w_k1_ADM_s104
                                 + 0.959998 w_k1_AXP_s104
                                 + 1.03013 w_k1_DIS_s104
                                 + 1.03742 w_k1_FDX_s104 + 0.6692 w_k1_IBM_s104
                                 + 0.882779 w_k1_INTC_s104
                                 + 1.04153 w_k1_KO_s104
                                 + 0.863368 w_k1_MAT_s104
                                 + 1.02921 w_k1_MSFT_s104
                                 + 0.989404 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.742661 w_k1_ADM_s104
                                 + 1.01155 w_k1_AXP_s104
                                 + 0.857083 w_k1_DIS_s104
                                 + 0.784765 w_k1_FDX_s104
                                 + 1.38565 w_k1_IBM_s104
                                 + 0.76165 w_k1_INTC_s104
                                 + 1.28511 w_k1_KO_s104 + 1.89778 w_k1_MAT_s104
                                 + 0.808946 w_k1_MSFT_s104
                                 + 0.965316 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.20163 w_k1_ADM_s104
                                 + 1.48347 w_k1_AXP_s104
                                 + 1.21922 w_k1_DIS_s104
                                 + 1.09431 w_k1_FDX_s104
                                 + 1.27777 w_k1_IBM_s104
                                 + 1.2296 w_k1_INTC_s104 + 1.27428 w_k1_KO_s104
                                 + 1.21469 w_k1_MAT_s104
                                 + 0.938244 w_k1_MSFT_s104
                                 + 1.11149 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.644449 w_k1_ADM_s104
                                 + 0.646538 w_k1_AXP_s104
                                 + 0.591536 w_k1_DIS_s104
                                 + 0.59691 w_k1_FDX_s104
                                 + 0.718444 w_k1_IBM_s104
                                 + 0.76986 w_k1_INTC_s104
                                 + 0.836178 w_k1_KO_s104
                                 + 0.538137 w_k1_MAT_s104
                                 + 0.658572 w_k1_MSFT_s104
                                 + 0.717306 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.850222 w_k1_ADM_s104
                                 + 0.924153 w_k1_AXP_s104
                                 + 0.67916 w_k1_DIS_s104
                                 + 0.766619 w_k1_FDX_s104
                                 + 0.863479 w_k1_IBM_s104
                                 + 0.826542 w_k1_INTC_s104
                                 + 0.81886 w_k1_KO_s104 + 1.1975 w_k1_MAT_s104
                                 + 0.65734 w_k1_MSFT_s104
                                 + 0.865007 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.980759 w_k1_ADM_s104
                                 + 1.05076 w_k1_AXP_s104
                                 + 0.93039 w_k1_DIS_s104
                                 + 0.908963 w_k1_FDX_s104
                                 + 0.880446 w_k1_IBM_s104
                                 + 1.02087 w_k1_INTC_s104
                                 + 1.13379 w_k1_KO_s104 + 1.15685 w_k1_MAT_s104
                                 + 0.964638 w_k1_MSFT_s104
                                 + 0.943117 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.21843 w_k1_ADM_s105
                                 + 1.32396 w_k1_AXP_s105
                                 + 1.44336 w_k1_DIS_s105
                                 + 1.27063 w_k1_FDX_s105 + 1.2921 w_k1_IBM_s105
                                 + 1.44537 w_k1_INTC_s105
                                 + 1.34312 w_k1_KO_s105 + 1.68897 w_k1_MAT_s105
                                 + 1.32322 w_k1_MSFT_s105
                                 + 1.14047 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.05294 w_k1_ADM_s105
                                 + 0.848657 w_k1_AXP_s105
                                 + 1.24111 w_k1_DIS_s105
                                 + 1.08283 w_k1_FDX_s105
                                 + 0.947943 w_k1_IBM_s105
                                 + 1.15317 w_k1_INTC_s105
                                 + 1.04174 w_k1_KO_s105
                                 + 0.779405 w_k1_MAT_s105
                                 + 1.33271 w_k1_MSFT_s105
                                 + 0.987367 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.12742 w_k1_ADM_s105
                                 + 1.06254 w_k1_AXP_s105
                                 + 0.838813 w_k1_DIS_s105
                                 + 0.904035 w_k1_FDX_s105
                                 + 1.04058 w_k1_IBM_s105
                                 + 0.998302 w_k1_INTC_s105
                                 + 0.971095 w_k1_KO_s105
                                 + 1.12383 w_k1_MAT_s105
                                 + 0.910078 w_k1_MSFT_s105
                                 + 0.972609 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.32363 w_k1_ADM_s105
                                 + 1.19327 w_k1_AXP_s105
                                 + 1.17993 w_k1_DIS_s105
                                 + 1.11583 w_k1_FDX_s105
                                 + 1.31159 w_k1_IBM_s105
                                 + 1.14648 w_k1_INTC_s105
                                 + 1.08657 w_k1_KO_s105 + 1.37462 w_k1_MAT_s105
                                 + 1.12593 w_k1_MSFT_s105
                                 + 1.08798 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.843035 w_k1_ADM_s105
                                 + 0.801763 w_k1_AXP_s105
                                 + 0.790016 w_k1_DIS_s105
                                 + 0.754397 w_k1_FDX_s105
                                 + 0.998282 w_k1_IBM_s105
                                 + 0.892438 w_k1_INTC_s105
                                 + 0.915408 w_k1_KO_s105
                                 + 1.19762 w_k1_MAT_s105
                                 + 0.817166 w_k1_MSFT_s105
                                 + 0.86095 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.0153 w_k1_ADM_s105
                                 + 1.00264 w_k1_AXP_s105
                                 + 1.25866 w_k1_DIS_s105
                                 + 1.02439 w_k1_FDX_s105
                                 + 1.64477 w_k1_IBM_s105
                                 + 1.09911 w_k1_INTC_s105
                                 + 1.30476 w_k1_KO_s105 + 1.49622 w_k1_MAT_s105
                                 + 1.03383 w_k1_MSFT_s105
                                 + 1.12879 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.874522 w_k1_ADM_s105
                                 + 0.821917 w_k1_AXP_s105
                                 + 0.882655 w_k1_DIS_s105
                                 + 0.89874 w_k1_FDX_s105
                                 + 0.791919 w_k1_IBM_s105
                                 + 0.996824 w_k1_INTC_s105
                                 + 0.883018 w_k1_KO_s105
                                 + 0.858916 w_k1_MAT_s105
                                 + 0.944851 w_k1_MSFT_s105
                                 + 0.901227 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.11364 w_k1_ADM_s105
                                 + 1.23667 w_k1_AXP_s105
                                 + 1.32308 w_k1_DIS_s105
                                 + 1.19221 w_k1_FDX_s105
                                 + 0.90959 w_k1_IBM_s105
                                 + 1.36342 w_k1_INTC_s105 + 1.1417 w_k1_KO_s105
                                 + 1.07573 w_k1_MAT_s105
                                 + 1.1644 w_k1_MSFT_s105 + 1.09688 w_k1_PG_s105
                                 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.749642 w_k1_ADM_s106
                                 + 0.665936 w_k1_AXP_s106
                                 + 0.665804 w_k1_DIS_s106
                                 + 0.723329 w_k1_FDX_s106
                                 + 0.894151 w_k1_IBM_s106
                                 + 0.87439 w_k1_INTC_s106
                                 + 0.785172 w_k1_KO_s106
                                 + 0.825515 w_k1_MAT_s106
                                 + 0.9448 w_k1_MSFT_s106 + 0.93086 w_k1_PG_s106
                                 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.960616 w_k1_ADM_s106
                                 + 0.772566 w_k1_AXP_s106
                                 + 0.752303 w_k1_DIS_s106
                                 + 0.861289 w_k1_FDX_s106
                                 + 0.651573 w_k1_IBM_s106
                                 + 0.768109 w_k1_INTC_s106
                                 + 0.757453 w_k1_KO_s106
                                 + 0.92995 w_k1_MAT_s106
                                 + 0.86887 w_k1_MSFT_s106
                                 + 0.909543 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.07819 w_k1_ADM_s106
                                 + 1.05509 w_k1_AXP_s106
                                 + 1.21669 w_k1_DIS_s106
                                 + 1.06934 w_k1_FDX_s106
                                 + 1.07824 w_k1_IBM_s106
                                 + 0.976892 w_k1_INTC_s106
                                 + 1.00974 w_k1_KO_s106 + 1.20707 w_k1_MAT_s106
                                 + 1.02422 w_k1_MSFT_s106
                                 + 1.14356 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.14538 w_k1_ADM_s106
                                 + 1.52056 w_k1_AXP_s106
                                 + 1.70926 w_k1_DIS_s106
                                 + 1.32944 w_k1_FDX_s106
                                 + 1.36394 w_k1_IBM_s106
                                 + 1.46571 w_k1_INTC_s106 + 1.3377 w_k1_KO_s106
                                 + 1.66743 w_k1_MAT_s106
                                 + 1.31768 w_k1_MSFT_s106
                                 + 1.28292 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.24574 w_k1_ADM_s106
                                 + 1.53148 w_k1_AXP_s106
                                 + 1.53003 w_k1_DIS_s106
                                 + 1.41567 w_k1_FDX_s106
                                 + 0.939902 w_k1_IBM_s106
                                 + 0.989175 w_k1_INTC_s106
                                 + 1.03687 w_k1_KO_s106 + 1.25405 w_k1_MAT_s106
                                 + 1.1482 w_k1_MSFT_s106 + 1.32886 w_k1_PG_s106
                                 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.08966 w_k1_ADM_s106
                                 + 1.06298 w_k1_AXP_s106 + 0.8566 w_k1_DIS_s106
                                 + 0.983961 w_k1_FDX_s106
                                 + 0.906528 w_k1_IBM_s106
                                 + 0.856198 w_k1_INTC_s106
                                 + 0.832524 w_k1_KO_s106
                                 + 0.858952 w_k1_MAT_s106
                                 + 0.916418 w_k1_MSFT_s106
                                 + 0.976761 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.809657 w_k1_ADM_s106
                                 + 1.10072 w_k1_AXP_s106
                                 + 1.07462 w_k1_DIS_s106
                                 + 0.950342 w_k1_FDX_s106
                                 + 1.5649 w_k1_IBM_s106
                                 + 1.04059 w_k1_INTC_s106
                                 + 1.23393 w_k1_KO_s106 + 1.47816 w_k1_MAT_s106
                                 + 1.13535 w_k1_MSFT_s106
                                 + 1.23986 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.00151 w_k1_ADM_s106
                                 + 0.908067 w_k1_AXP_s106
                                 + 1.11812 w_k1_DIS_s106 + 1.1071 w_k1_FDX_s106
                                 + 0.792454 w_k1_IBM_s106
                                 + 0.784811 w_k1_INTC_s106
                                 + 0.924327 w_k1_KO_s106
                                 + 1.05339 w_k1_MAT_s106
                                 + 1.05212 w_k1_MSFT_s106
                                 + 1.05583 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.13662 w_k1_ADM_s107
                                 + 1.0633 w_k1_AXP_s107 + 1.70771 w_k1_DIS_s107
                                 + 1.14196 w_k1_FDX_s107
                                 + 1.20661 w_k1_IBM_s107
                                 + 1.23273 w_k1_INTC_s107
                                 + 1.22365 w_k1_KO_s107
                                 + 0.942115 w_k1_MAT_s107
                                 + 1.26668 w_k1_MSFT_s107
                                 + 1.03918 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.912819 w_k1_ADM_s107
                                 + 0.801453 w_k1_AXP_s107
                                 + 1.14902 w_k1_DIS_s107
                                 + 1.02884 w_k1_FDX_s107
                                 + 0.757746 w_k1_IBM_s107
                                 + 1.07007 w_k1_INTC_s107
                                 + 0.968525 w_k1_KO_s107
                                 + 0.65794 w_k1_MAT_s107
                                 + 1.07928 w_k1_MSFT_s107
                                 + 1.00063 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.914706 w_k1_ADM_s107
                                 + 0.862679 w_k1_AXP_s107
                                 + 0.806018 w_k1_DIS_s107
                                 + 0.783056 w_k1_FDX_s107
                                 + 0.91204 w_k1_IBM_s107
                                 + 0.821704 w_k1_INTC_s107
                                 + 0.854897 w_k1_KO_s107
                                 + 0.956188 w_k1_MAT_s107
                                 + 0.8707 w_k1_MSFT_s107
                                 + 0.874456 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.819061 w_k1_ADM_s107
                                 + 0.778199 w_k1_AXP_s107
                                 + 0.831154 w_k1_DIS_s107
                                 + 0.88118 w_k1_FDX_s107
                                 + 0.765378 w_k1_IBM_s107
                                 + 1.10786 w_k1_INTC_s107
                                 + 0.752839 w_k1_KO_s107
                                 + 0.629668 w_k1_MAT_s107
                                 + 0.822389 w_k1_MSFT_s107
                                 + 0.857024 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.627587 w_k1_ADM_s107
                                 + 0.735628 w_k1_AXP_s107
                                 + 0.907873 w_k1_DIS_s107
                                 + 0.785986 w_k1_FDX_s107
                                 + 1.33023 w_k1_IBM_s107
                                 + 1.05925 w_k1_INTC_s107
                                 + 1.05987 w_k1_KO_s107 + 1.17642 w_k1_MAT_s107
                                 + 0.964035 w_k1_MSFT_s107
                                 + 1.00667 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.07971 w_k1_ADM_s107
                                 + 1.15643 w_k1_AXP_s107
                                 + 1.03262 w_k1_DIS_s107
                                 + 1.07394 w_k1_FDX_s107
                                 + 0.82788 w_k1_IBM_s107
                                 + 0.954922 w_k1_INTC_s107
                                 + 0.926619 w_k1_KO_s107
                                 + 0.908856 w_k1_MAT_s107
                                 + 1.09359 w_k1_MSFT_s107 + 1.0176 w_k1_PG_s107
                                 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.740219 w_k1_ADM_s107
                                 + 0.611239 w_k1_AXP_s107
                                 + 0.665207 w_k1_DIS_s107
                                 + 0.697526 w_k1_FDX_s107
                                 + 0.787183 w_k1_IBM_s107
                                 + 0.923834 w_k1_INTC_s107
                                 + 0.641633 w_k1_KO_s107
                                 + 0.573025 w_k1_MAT_s107
                                 + 0.739049 w_k1_MSFT_s107
                                 + 0.66747 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.910751 w_k1_ADM_s107
                                 + 1.43234 w_k1_AXP_s107
                                 + 1.33175 w_k1_DIS_s107 + 1.1441 w_k1_FDX_s107
                                 + 1.28351 w_k1_IBM_s107
                                 + 1.32931 w_k1_INTC_s107
                                 + 1.19006 w_k1_KO_s107 + 1.45509 w_k1_MAT_s107
                                 + 1.1236 w_k1_MSFT_s107 + 1.20118 w_k1_PG_s107
                                 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.954451 w_k1_ADM_s110
                                 + 1.32047 w_k1_AXP_s110
                                 + 1.18219 w_k1_DIS_s110 + 1.1758 w_k1_FDX_s110
                                 + 0.856861 w_k1_IBM_s110
                                 + 0.946106 w_k1_INTC_s110
                                 + 1.05287 w_k1_KO_s110
                                 + 0.893471 w_k1_MAT_s110
                                 + 1.11156 w_k1_MSFT_s110
                                 + 1.12108 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.05659 w_k1_ADM_s110
                                 + 1.4233 w_k1_AXP_s110 + 1.24378 w_k1_DIS_s110
                                 + 1.08168 w_k1_FDX_s110
                                 + 1.40545 w_k1_IBM_s110
                                 + 1.20035 w_k1_INTC_s110
                                 + 1.36573 w_k1_KO_s110 + 1.59946 w_k1_MAT_s110
                                 + 1.19073 w_k1_MSFT_s110
                                 + 1.26731 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.834336 w_k1_ADM_s110
                                 + 0.822505 w_k1_AXP_s110
                                 + 0.899998 w_k1_DIS_s110
                                 + 0.803188 w_k1_FDX_s110
                                 + 1.52935 w_k1_IBM_s110
                                 + 0.914852 w_k1_INTC_s110
                                 + 1.11805 w_k1_KO_s110 + 1.30896 w_k1_MAT_s110
                                 + 0.966762 w_k1_MSFT_s110
                                 + 1.08368 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.805471 w_k1_ADM_s110
                                 + 0.803995 w_k1_AXP_s110
                                 + 0.797486 w_k1_DIS_s110
                                 + 0.816716 w_k1_FDX_s110
                                 + 0.857078 w_k1_IBM_s110
                                 + 0.712929 w_k1_INTC_s110
                                 + 0.957598 w_k1_KO_s110
                                 + 0.885052 w_k1_MAT_s110
                                 + 0.881657 w_k1_MSFT_s110
                                 + 0.966422 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.18411 w_k1_ADM_s110
                                 + 1.30458 w_k1_AXP_s110
                                 + 1.61909 w_k1_DIS_s110 + 1.3129 w_k1_FDX_s110
                                 + 1.2054 w_k1_IBM_s110
                                 + 1.01084 w_k1_INTC_s110
                                 + 1.34778 w_k1_KO_s110 + 1.08632 w_k1_MAT_s110
                                 + 1.28766 w_k1_MSFT_s110
                                 + 1.35499 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.656638 w_k1_ADM_s110
                                 + 0.64876 w_k1_AXP_s110
                                 + 0.639468 w_k1_DIS_s110
                                 + 0.725509 w_k1_FDX_s110
                                 + 0.633613 w_k1_IBM_s110
                                 + 0.728315 w_k1_INTC_s110
                                 + 0.723038 w_k1_KO_s110
                                 + 0.539057 w_k1_MAT_s110
                                 + 0.766847 w_k1_MSFT_s110
                                 + 0.829627 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.702646 w_k1_ADM_s110
                                 + 1.11621 w_k1_AXP_s110
                                 + 0.827302 w_k1_DIS_s110
                                 + 0.952015 w_k1_FDX_s110
                                 + 1.07776 w_k1_IBM_s110
                                 + 0.976245 w_k1_INTC_s110
                                 + 0.906968 w_k1_KO_s110
                                 + 1.49096 w_k1_MAT_s110
                                 + 0.929574 w_k1_MSFT_s110
                                 + 1.05997 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.15768 w_k1_ADM_s110
                                 + 1.14229 w_k1_AXP_s110 + 1.107 w_k1_DIS_s110
                                 + 1.08906 w_k1_FDX_s110
                                 + 1.05785 w_k1_IBM_s110
                                 + 0.992969 w_k1_INTC_s110
                                 + 1.07849 w_k1_KO_s110 + 1.10795 w_k1_MAT_s110
                                 + 1.08123 w_k1_MSFT_s110
                                 + 1.16444 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.3113 w_k1_ADM_s111
                                 + 1.38019 w_k1_AXP_s111
                                 + 1.75339 w_k1_DIS_s111
                                 + 1.41063 w_k1_FDX_s111
                                 + 1.21599 w_k1_IBM_s111
                                 + 1.48493 w_k1_INTC_s111
                                 + 1.25845 w_k1_KO_s111 + 1.13808 w_k1_MAT_s111
                                 + 1.42482 w_k1_MSFT_s111
                                 + 1.20154 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.754343 w_k1_ADM_s111
                                 + 0.81004 w_k1_AXP_s111
                                 + 0.682816 w_k1_DIS_s111
                                 + 0.747142 w_k1_FDX_s111
                                 + 1.17784 w_k1_IBM_s111
                                 + 0.847954 w_k1_INTC_s111
                                 + 0.889992 w_k1_KO_s111
                                 + 1.28299 w_k1_MAT_s111
                                 + 0.790614 w_k1_MSFT_s111
                                 + 0.844532 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.06893 w_k1_ADM_s111
                                 + 0.94059 w_k1_AXP_s111
                                 + 0.859958 w_k1_DIS_s111
                                 + 0.995457 w_k1_FDX_s111
                                 + 0.667808 w_k1_IBM_s111
                                 + 1.04184 w_k1_INTC_s111
                                 + 0.729082 w_k1_KO_s111
                                 + 0.811548 w_k1_MAT_s111
                                 + 0.921321 w_k1_MSFT_s111
                                 + 0.956792 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.08609 w_k1_ADM_s111
                                 + 1.4886 w_k1_AXP_s111 + 1.01899 w_k1_DIS_s111
                                 + 1.04798 w_k1_FDX_s111
                                 + 1.50584 w_k1_IBM_s111
                                 + 1.0385 w_k1_INTC_s111 + 1.17411 w_k1_KO_s111
                                 + 1.93606 w_k1_MAT_s111
                                 + 0.97131 w_k1_MSFT_s111
                                 + 1.06463 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.10891 w_k1_ADM_s111
                                 + 1.02979 w_k1_AXP_s111
                                 + 1.11419 w_k1_DIS_s111
                                 + 1.05683 w_k1_FDX_s111
                                 + 0.99592 w_k1_IBM_s111
                                 + 1.22315 w_k1_INTC_s111 + 1.0029 w_k1_KO_s111
                                 + 0.958845 w_k1_MAT_s111
                                 + 1.00583 w_k1_MSFT_s111
                                 + 1.10972 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.950782 w_k1_ADM_s111
                                 + 1.06115 w_k1_AXP_s111
                                 + 0.730988 w_k1_DIS_s111
                                 + 0.875046 w_k1_FDX_s111
                                 + 0.975587 w_k1_IBM_s111
                                 + 1.03566 w_k1_INTC_s111
                                 + 0.841081 w_k1_KO_s111 + 1.191 w_k1_MAT_s111
                                 + 0.770735 w_k1_MSFT_s111
                                 + 0.880877 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.31061 w_k1_ADM_s111
                                 + 1.25015 w_k1_AXP_s111
                                 + 1.24449 w_k1_DIS_s111
                                 + 1.27664 w_k1_FDX_s111
                                 + 0.901905 w_k1_IBM_s111
                                 + 1.21914 w_k1_INTC_s111
                                 + 0.965239 w_k1_KO_s111
                                 + 0.937651 w_k1_MAT_s111
                                 + 1.08887 w_k1_MSFT_s111
                                 + 1.00412 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.986941 w_k1_ADM_s111
                                 + 1.25518 w_k1_AXP_s111
                                 + 1.21028 w_k1_DIS_s111
                                 + 1.13735 w_k1_FDX_s111
                                 + 1.27451 w_k1_IBM_s111
                                 + 1.13561 w_k1_INTC_s111
                                 + 1.09942 w_k1_KO_s111 + 1.5769 w_k1_MAT_s111
                                 + 1.06242 w_k1_MSFT_s111
                                 + 1.06353 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.33226 w_k1_ADM_s112
                                 + 1.37483 w_k1_AXP_s112
                                 + 1.41724 w_k1_DIS_s112
                                 + 1.36548 w_k1_FDX_s112
                                 + 1.11233 w_k1_IBM_s112
                                 + 1.14723 w_k1_INTC_s112
                                 + 1.40997 w_k1_KO_s112 + 1.19373 w_k1_MAT_s112
                                 + 1.22025 w_k1_MSFT_s112
                                 + 1.15694 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.90599 w_k1_ADM_s112
                                 + 1.30948 w_k1_AXP_s112
                                 + 1.08252 w_k1_DIS_s112
                                 + 1.07096 w_k1_FDX_s112
                                 + 1.16206 w_k1_IBM_s112
                                 + 1.39103 w_k1_INTC_s112
                                 + 1.08243 w_k1_KO_s112 + 1.19147 w_k1_MAT_s112
                                 + 1.15248 w_k1_MSFT_s112
                                 + 0.977573 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.96542 w_k1_ADM_s112
                                 + 0.877679 w_k1_AXP_s112
                                 + 1.08177 w_k1_DIS_s112
                                 + 1.14248 w_k1_FDX_s112
                                 + 0.725268 w_k1_IBM_s112
                                 + 1.04753 w_k1_INTC_s112
                                 + 1.04533 w_k1_KO_s112
                                 + 0.730675 w_k1_MAT_s112
                                 + 1.1663 w_k1_MSFT_s112
                                 + 0.950492 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.94989 w_k1_ADM_s112
                                 + 1.26562 w_k1_AXP_s112
                                 + 1.07492 w_k1_DIS_s112 + 1.063 w_k1_FDX_s112
                                 + 1.15771 w_k1_IBM_s112
                                 + 0.937053 w_k1_INTC_s112
                                 + 1.15902 w_k1_KO_s112 + 2.10109 w_k1_MAT_s112
                                 + 1.0739 w_k1_MSFT_s112 + 1.1507 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.893083 w_k1_ADM_s112
                                 + 0.866481 w_k1_AXP_s112
                                 + 0.762277 w_k1_DIS_s112
                                 + 0.875515 w_k1_FDX_s112
                                 + 0.871909 w_k1_IBM_s112
                                 + 0.792809 w_k1_INTC_s112
                                 + 0.958307 w_k1_KO_s112
                                 + 1.09504 w_k1_MAT_s112
                                 + 0.835505 w_k1_MSFT_s112
                                 + 0.900892 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.864449 w_k1_ADM_s112
                                 + 1.02407 w_k1_AXP_s112
                                 + 0.969416 w_k1_DIS_s112
                                 + 0.927278 w_k1_FDX_s112
                                 + 1.57498 w_k1_IBM_s112
                                 + 0.822051 w_k1_INTC_s112
                                 + 1.24646 w_k1_KO_s112 + 1.99492 w_k1_MAT_s112
                                 + 0.908785 w_k1_MSFT_s112
                                 + 0.958437 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.802916 w_k1_ADM_s112
                                 + 0.687675 w_k1_AXP_s112
                                 + 0.610436 w_k1_DIS_s112
                                 + 0.780907 w_k1_FDX_s112
                                 + 0.872056 w_k1_IBM_s112
                                 + 0.749847 w_k1_INTC_s112
                                 + 0.765396 w_k1_KO_s112
                                 + 0.871443 w_k1_MAT_s112
                                 + 0.78054 w_k1_MSFT_s112
                                 + 0.827258 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.09071 w_k1_ADM_s112
                                 + 0.880866 w_k1_AXP_s112
                                 + 1.07303 w_k1_DIS_s112
                                 + 1.08415 w_k1_FDX_s112
                                 + 1.02483 w_k1_IBM_s112
                                 + 1.09714 w_k1_INTC_s112
                                 + 1.12617 w_k1_KO_s112
                                 + 0.912255 w_k1_MAT_s112
                                 + 1.23868 w_k1_MSFT_s112
                                 + 1.04375 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.916076 w_k1_ADM_s113
                                 + 0.855513 w_k1_AXP_s113
                                 + 0.752666 w_k1_DIS_s113
                                 + 0.775563 w_k1_FDX_s113
                                 + 0.926501 w_k1_IBM_s113
                                 + 0.748705 w_k1_INTC_s113
                                 + 0.779976 w_k1_KO_s113
                                 + 0.893021 w_k1_MAT_s113
                                 + 0.791146 w_k1_MSFT_s113
                                 + 0.906035 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.969396 w_k1_ADM_s113
                                 + 1.03535 w_k1_AXP_s113
                                 + 1.16055 w_k1_DIS_s113
                                 + 0.995333 w_k1_FDX_s113
                                 + 1.11816 w_k1_IBM_s113
                                 + 0.93633 w_k1_INTC_s113
                                 + 1.16599 w_k1_KO_s113
                                 + 0.913388 w_k1_MAT_s113
                                 + 1.0391 w_k1_MSFT_s113
                                 + 0.976659 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.64415 w_k1_ADM_s113
                                 + 1.57664 w_k1_AXP_s113
                                 + 2.26527 w_k1_DIS_s113
                                 + 1.49668 w_k1_FDX_s113 + 1.2974 w_k1_IBM_s113
                                 + 1.40101 w_k1_INTC_s113
                                 + 1.63888 w_k1_KO_s113 + 1.06134 w_k1_MAT_s113
                                 + 1.5063 w_k1_MSFT_s113 + 1.34399 w_k1_PG_s113
                                 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.785234 w_k1_ADM_s113
                                 + 0.948734 w_k1_AXP_s113
                                 + 1.16635 w_k1_DIS_s113 + 1.0185 w_k1_FDX_s113
                                 + 0.800929 w_k1_IBM_s113
                                 + 0.897254 w_k1_INTC_s113
                                 + 0.968318 w_k1_KO_s113 + 0.8272 w_k1_MAT_s113
                                 + 1.06586 w_k1_MSFT_s113
                                 + 0.999466 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.01436 w_k1_ADM_s113
                                 + 1.18822 w_k1_AXP_s113
                                 + 0.876363 w_k1_DIS_s113
                                 + 0.861725 w_k1_FDX_s113
                                 + 1.15017 w_k1_IBM_s113
                                 + 0.897886 w_k1_INTC_s113
                                 + 1.03892 w_k1_KO_s113 + 1.34391 w_k1_MAT_s113
                                 + 0.846993 w_k1_MSFT_s113
                                 + 1.03409 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.902259 w_k1_ADM_s113
                                 + 0.973208 w_k1_AXP_s113
                                 + 0.701178 w_k1_DIS_s113
                                 + 0.908438 w_k1_FDX_s113
                                 + 0.498625 w_k1_IBM_s113
                                 + 0.766704 w_k1_INTC_s113
                                 + 0.739783 w_k1_KO_s113
                                 + 0.734295 w_k1_MAT_s113
                                 + 0.901274 w_k1_MSFT_s113
                                 + 0.87387 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.693332 w_k1_ADM_s113
                                 + 0.731114 w_k1_AXP_s113
                                 + 0.849179 w_k1_DIS_s113
                                 + 0.801218 w_k1_FDX_s113
                                 + 0.705598 w_k1_IBM_s113
                                 + 0.795706 w_k1_INTC_s113
                                 + 0.878842 w_k1_KO_s113
                                 + 0.607179 w_k1_MAT_s113
                                 + 0.890748 w_k1_MSFT_s113
                                 + 0.953487 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.00614 w_k1_ADM_s113
                                 + 1.53702 w_k1_AXP_s113
                                 + 1.32622 w_k1_DIS_s113
                                 + 1.26139 w_k1_FDX_s113
                                 + 1.16151 w_k1_IBM_s113
                                 + 0.979992 w_k1_INTC_s113
                                 + 1.03533 w_k1_KO_s113 + 1.42605 w_k1_MAT_s113
                                 + 1.07899 w_k1_MSFT_s113
                                 + 1.20074 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.41365 w_k1_ADM_s114
                                 + 1.16415 w_k1_AXP_s114
                                 + 1.20044 w_k1_DIS_s114
                                 + 1.20241 w_k1_FDX_s114
                                 + 1.14732 w_k1_IBM_s114
                                 + 0.850671 w_k1_INTC_s114
                                 + 1.11929 w_k1_KO_s114 + 1.41676 w_k1_MAT_s114
                                 + 1.01916 w_k1_MSFT_s114
                                 + 1.04349 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.12439 w_k1_ADM_s114
                                 + 1.07946 w_k1_AXP_s114
                                 + 1.23951 w_k1_DIS_s114
                                 + 1.27086 w_k1_FDX_s114
                                 + 0.771042 w_k1_IBM_s114
                                 + 1.05627 w_k1_INTC_s114
                                 + 0.93243 w_k1_KO_s114
                                 + 0.755319 w_k1_MAT_s114
                                 + 1.09746 w_k1_MSFT_s114
                                 + 0.972374 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.948812 w_k1_ADM_s114
                                 + 0.896106 w_k1_AXP_s114
                                 + 0.858255 w_k1_DIS_s114
                                 + 1.00497 w_k1_FDX_s114
                                 + 0.92554 w_k1_IBM_s114
                                 + 0.850182 w_k1_INTC_s114
                                 + 0.877994 w_k1_KO_s114
                                 + 0.989926 w_k1_MAT_s114
                                 + 0.856136 w_k1_MSFT_s114
                                 + 0.957663 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.790316 w_k1_ADM_s114
                                 + 0.671785 w_k1_AXP_s114
                                 + 0.592861 w_k1_DIS_s114
                                 + 0.697258 w_k1_FDX_s114
                                 + 0.944827 w_k1_IBM_s114
                                 + 0.810187 w_k1_INTC_s114
                                 + 0.830501 w_k1_KO_s114
                                 + 0.639762 w_k1_MAT_s114
                                 + 0.727697 w_k1_MSFT_s114
                                 + 0.789841 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.83423 w_k1_ADM_s114
                                 + 0.923765 w_k1_AXP_s114
                                 + 0.778177 w_k1_DIS_s114
                                 + 0.894189 w_k1_FDX_s114
                                 + 1.38219 w_k1_IBM_s114
                                 + 0.873082 w_k1_INTC_s114
                                 + 1.06866 w_k1_KO_s114 + 1.09281 w_k1_MAT_s114
                                 + 0.825831 w_k1_MSFT_s114
                                 + 0.856559 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.23739 w_k1_ADM_s114
                                 + 1.0905 w_k1_AXP_s114 + 1.64519 w_k1_DIS_s114
                                 + 1.43539 w_k1_FDX_s114
                                 + 1.09834 w_k1_IBM_s114
                                 + 1.41725 w_k1_INTC_s114
                                 + 1.28043 w_k1_KO_s114
                                 + 0.929929 w_k1_MAT_s114
                                 + 1.227 w_k1_MSFT_s114 + 1.11468 w_k1_PG_s114
                                 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.887906 w_k1_ADM_s114
                                 + 0.737633 w_k1_AXP_s114
                                 + 0.75469 w_k1_DIS_s114
                                 + 0.861776 w_k1_FDX_s114
                                 + 0.686215 w_k1_IBM_s114
                                 + 0.851729 w_k1_INTC_s114
                                 + 0.706274 w_k1_KO_s114
                                 + 0.55093 w_k1_MAT_s114
                                 + 0.837244 w_k1_MSFT_s114
                                 + 0.814744 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.51853 w_k1_ADM_s114
                                 + 1.41789 w_k1_AXP_s114 + 1.7691 w_k1_DIS_s114
                                 + 1.31463 w_k1_FDX_s114
                                 + 1.77315 w_k1_IBM_s114
                                 + 1.14425 w_k1_INTC_s114
                                 + 1.62492 w_k1_KO_s114 + 1.93605 w_k1_MAT_s114
                                 + 1.25493 w_k1_MSFT_s114 + 1.3078 w_k1_PG_s114
                                 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.780982 w_k1_ADM_s115
                                 + 0.635614 w_k1_AXP_s115
                                 + 0.620034 w_k1_DIS_s115
                                 + 0.738335 w_k1_FDX_s115
                                 + 0.783638 w_k1_IBM_s115
                                 + 0.769743 w_k1_INTC_s115
                                 + 0.883048 w_k1_KO_s115
                                 + 0.871703 w_k1_MAT_s115
                                 + 0.722004 w_k1_MSFT_s115
                                 + 0.910239 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.20791 w_k1_ADM_s115
                                 + 1.56426 w_k1_AXP_s115
                                 + 1.38108 w_k1_DIS_s115
                                 + 1.23695 w_k1_FDX_s115
                                 + 1.38202 w_k1_IBM_s115
                                 + 1.11176 w_k1_INTC_s115
                                 + 1.37337 w_k1_KO_s115 + 2.04235 w_k1_MAT_s115
                                 + 1.04605 w_k1_MSFT_s115
                                 + 1.24166 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.988557 w_k1_ADM_s115
                                 + 0.669488 w_k1_AXP_s115
                                 + 0.792358 w_k1_DIS_s115
                                 + 0.711569 w_k1_FDX_s115
                                 + 1.18384 w_k1_IBM_s115
                                 + 0.737122 w_k1_INTC_s115
                                 + 1.0838 w_k1_KO_s115 + 1.22098 w_k1_MAT_s115
                                 + 0.836177 w_k1_MSFT_s115
                                 + 1.06944 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.06515 w_k1_ADM_s115
                                 + 0.967455 w_k1_AXP_s115
                                 + 0.929385 w_k1_DIS_s115
                                 + 0.882078 w_k1_FDX_s115
                                 + 1.39153 w_k1_IBM_s115
                                 + 0.970291 w_k1_INTC_s115
                                 + 1.23542 w_k1_KO_s115 + 1.56988 w_k1_MAT_s115
                                 + 0.989295 w_k1_MSFT_s115
                                 + 1.11476 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.01637 w_k1_ADM_s115
                                 + 0.924866 w_k1_AXP_s115
                                 + 0.617087 w_k1_DIS_s115
                                 + 0.79681 w_k1_FDX_s115
                                 + 1.07678 w_k1_IBM_s115
                                 + 0.72386 w_k1_INTC_s115
                                 + 0.853658 w_k1_KO_s115
                                 + 1.14027 w_k1_MAT_s115
                                 + 0.727039 w_k1_MSFT_s115
                                 + 0.906785 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.13539 w_k1_ADM_s115
                                 + 1.06702 w_k1_AXP_s115
                                 + 1.16303 w_k1_DIS_s115
                                 + 1.13328 w_k1_FDX_s115
                                 + 0.938391 w_k1_IBM_s115
                                 + 1.2147 w_k1_INTC_s115 + 1.15064 w_k1_KO_s115
                                 + 1.10918 w_k1_MAT_s115
                                 + 1.1741 w_k1_MSFT_s115 + 1.22311 w_k1_PG_s115
                                 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.38556 w_k1_ADM_s115
                                 + 1.42779 w_k1_AXP_s115
                                 + 1.46091 w_k1_DIS_s115
                                 + 1.45423 w_k1_FDX_s115
                                 + 0.899189 w_k1_IBM_s115
                                 + 1.16227 w_k1_INTC_s115
                                 + 1.12095 w_k1_KO_s115 + 1.25361 w_k1_MAT_s115
                                 + 1.25167 w_k1_MSFT_s115
                                 + 1.17416 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.16307 w_k1_ADM_s115
                                 + 0.906546 w_k1_AXP_s115
                                 + 0.901575 w_k1_DIS_s115
                                 + 1.00467 w_k1_FDX_s115
                                 + 0.714467 w_k1_IBM_s115
                                 + 1.01183 w_k1_INTC_s115
                                 + 0.868629 w_k1_KO_s115
                                 + 0.974197 w_k1_MAT_s115
                                 + 1.0002 w_k1_MSFT_s115 + 1.00949 w_k1_PG_s115
                                 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.842291 w_k1_ADM_s116
                                 + 0.896649 w_k1_AXP_s116
                                 + 0.973362 w_k1_DIS_s116
                                 + 0.840941 w_k1_FDX_s116
                                 + 0.980016 w_k1_IBM_s116
                                 + 0.793014 w_k1_INTC_s116
                                 + 0.888304 w_k1_KO_s116
                                 + 0.701706 w_k1_MAT_s116
                                 + 0.864902 w_k1_MSFT_s116
                                 + 1.00068 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.947386 w_k1_ADM_s116
                                 + 0.743621 w_k1_AXP_s116
                                 + 0.625014 w_k1_DIS_s116
                                 + 0.729854 w_k1_FDX_s116
                                 + 0.925147 w_k1_IBM_s116
                                 + 0.732591 w_k1_INTC_s116
                                 + 0.763955 w_k1_KO_s116
                                 + 0.732565 w_k1_MAT_s116
                                 + 0.777622 w_k1_MSFT_s116
                                 + 0.911192 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.779948 w_k1_ADM_s116
                                 + 0.722421 w_k1_AXP_s116
                                 + 0.958278 w_k1_DIS_s116
                                 + 0.9093 w_k1_FDX_s116 + 0.73576 w_k1_IBM_s116
                                 + 0.982518 w_k1_INTC_s116
                                 + 0.886798 w_k1_KO_s116
                                 + 0.551186 w_k1_MAT_s116
                                 + 0.988886 w_k1_MSFT_s116
                                 + 0.941303 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.91215 w_k1_ADM_s116
                                 + 1.02738 w_k1_AXP_s116
                                 + 0.995719 w_k1_DIS_s116
                                 + 0.920083 w_k1_FDX_s116
                                 + 1.49058 w_k1_IBM_s116
                                 + 0.929593 w_k1_INTC_s116
                                 + 1.20385 w_k1_KO_s116 + 1.16441 w_k1_MAT_s116
                                 + 1.0847 w_k1_MSFT_s116 + 1.12571 w_k1_PG_s116
                                 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.977564 w_k1_ADM_s116
                                 + 1.02081 w_k1_AXP_s116
                                 + 0.942331 w_k1_DIS_s116
                                 + 0.98356 w_k1_FDX_s116
                                 + 0.949496 w_k1_IBM_s116
                                 + 1.01513 w_k1_INTC_s116
                                 + 0.977795 w_k1_KO_s116
                                 + 1.12376 w_k1_MAT_s116
                                 + 1.09983 w_k1_MSFT_s116
                                 + 1.02952 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.29313 w_k1_ADM_s116
                                 + 1.40238 w_k1_AXP_s116
                                 + 1.75517 w_k1_DIS_s116
                                 + 1.42145 w_k1_FDX_s116
                                 + 1.14309 w_k1_IBM_s116
                                 + 1.26963 w_k1_INTC_s116
                                 + 1.30779 w_k1_KO_s116 + 1.08683 w_k1_MAT_s116
                                 + 1.4822 w_k1_MSFT_s116 + 1.30085 w_k1_PG_s116
                                 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.10628 w_k1_ADM_s116
                                 + 1.34138 w_k1_AXP_s116
                                 + 1.35563 w_k1_DIS_s116
                                 + 1.13545 w_k1_FDX_s116
                                 + 1.09645 w_k1_IBM_s116
                                 + 0.900085 w_k1_INTC_s116
                                 + 1.06667 w_k1_KO_s116 + 1.13007 w_k1_MAT_s116
                                 + 1.16735 w_k1_MSFT_s116
                                 + 1.11948 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.04844 w_k1_ADM_s116
                                 + 1.138 w_k1_AXP_s116 + 1.05252 w_k1_DIS_s116
                                 + 1.21613 w_k1_FDX_s116
                                 + 0.525687 w_k1_IBM_s116
                                 + 0.980431 w_k1_INTC_s116
                                 + 0.720123 w_k1_KO_s116
                                 + 0.74989 w_k1_MAT_s116
                                 + 1.10577 w_k1_MSFT_s116
                                 + 0.973182 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.27374 w_k1_ADM_s117
                                 + 0.972639 w_k1_AXP_s117
                                 + 1.26863 w_k1_DIS_s117
                                 + 1.21626 w_k1_FDX_s117
                                 + 0.787024 w_k1_IBM_s117
                                 + 0.920523 w_k1_INTC_s117
                                 + 0.946949 w_k1_KO_s117
                                 + 0.731156 w_k1_MAT_s117
                                 + 1.23047 w_k1_MSFT_s117
                                 + 1.18811 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.767262 w_k1_ADM_s117
                                 + 0.805333 w_k1_AXP_s117
                                 + 0.561512 w_k1_DIS_s117
                                 + 0.755877 w_k1_FDX_s117
                                 + 0.85551 w_k1_IBM_s117
                                 + 0.75209 w_k1_INTC_s117
                                 + 0.734498 w_k1_KO_s117
                                 + 0.746691 w_k1_MAT_s117
                                 + 0.762848 w_k1_MSFT_s117
                                 + 0.883934 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.45358 w_k1_ADM_s117
                                 + 1.27037 w_k1_AXP_s117
                                 + 1.55069 w_k1_DIS_s117
                                 + 1.33557 w_k1_FDX_s117
                                 + 1.04501 w_k1_IBM_s117
                                 + 0.943551 w_k1_INTC_s117
                                 + 1.22407 w_k1_KO_s117 + 1.20087 w_k1_MAT_s117
                                 + 1.25167 w_k1_MSFT_s117
                                 + 1.41535 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.813642 w_k1_ADM_s117
                                 + 0.887934 w_k1_AXP_s117
                                 + 0.856322 w_k1_DIS_s117
                                 + 0.846378 w_k1_FDX_s117
                                 + 1.1725 w_k1_IBM_s117
                                 + 0.750077 w_k1_INTC_s117
                                 + 0.958099 w_k1_KO_s117
                                 + 1.07024 w_k1_MAT_s117
                                 + 0.814035 w_k1_MSFT_s117
                                 + 0.967177 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.18221 w_k1_ADM_s117
                                 + 1.17192 w_k1_AXP_s117
                                 + 1.40272 w_k1_DIS_s117
                                 + 1.18733 w_k1_FDX_s117
                                 + 1.78345 w_k1_IBM_s117
                                 + 0.936774 w_k1_INTC_s117
                                 + 1.32324 w_k1_KO_s117 + 1.74802 w_k1_MAT_s117
                                 + 1.10105 w_k1_MSFT_s117
                                 + 1.29665 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.932106 w_k1_ADM_s117
                                 + 0.824826 w_k1_AXP_s117
                                 + 0.830651 w_k1_DIS_s117
                                 + 0.916459 w_k1_FDX_s117
                                 + 0.712172 w_k1_IBM_s117
                                 + 0.786441 w_k1_INTC_s117
                                 + 0.795329 w_k1_KO_s117
                                 + 0.607788 w_k1_MAT_s117
                                 + 0.902212 w_k1_MSFT_s117
                                 + 1.03108 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.03816 w_k1_ADM_s117
                                 + 1.25105 w_k1_AXP_s117
                                 + 0.925506 w_k1_DIS_s117
                                 + 1.01799 w_k1_FDX_s117
                                 + 0.913887 w_k1_IBM_s117
                                 + 0.858221 w_k1_INTC_s117
                                 + 0.922751 w_k1_KO_s117
                                 + 0.92837 w_k1_MAT_s117
                                 + 0.942568 w_k1_MSFT_s117
                                 + 1.17266 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.08495 w_k1_ADM_s117
                                 + 1.12424 w_k1_AXP_s117
                                 + 1.31911 w_k1_DIS_s117
                                 + 1.06001 w_k1_FDX_s117
                                 + 1.35173 w_k1_IBM_s117
                                 + 0.989863 w_k1_INTC_s117
                                 + 1.10635 w_k1_KO_s117
                                 + 0.975872 w_k1_MAT_s117
                                 + 1.11004 w_k1_MSFT_s117
                                 + 1.14435 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.031754915691215 w_k1_ADM_s000
                                 + 0.0531054790042999 w_k1_ADM_s001
                                 + 0.0156189631066959 w_k1_ADM_s002
                                 + 0.0236561886804931 w_k1_ADM_s003
                                 + 0.0224820508568449 w_k1_ADM_s004
                                 + 0.0288174086269374 w_k1_ADM_s005
                                 + 0.0209913272259169 w_k1_ADM_s006
                                 + 0.061503093358863 w_k1_ADM_s007
                                 + 0.00927438669299158 w_k1_ADM_s010
                                 + 0.00673737925783388 w_k1_ADM_s011
                                 + 0.00885946498285343 w_k1_ADM_s012
                                 + 0.00980125488456372 w_k1_ADM_s013
                                 + 0.0048678609765911 w_k1_ADM_s014
                                 + 0.0119028909837038 w_k1_ADM_s015
                                 + 0.0102677305804352 w_k1_ADM_s016
                                 + 0.0182265529577632 w_k1_ADM_s017
                                 + 0.0175061811108134 w_k1_ADM_s100
                                 + 0.0161067362696327 w_k1_ADM_s101
                                 + 0.0154458522115313 w_k1_ADM_s102
                                 + 0.0420665606815979 w_k1_ADM_s103
                                 + 0.0372992425839071 w_k1_ADM_s104
                                 + 0.0351327247376361 w_k1_ADM_s105
                                 + 0.0632848371816932 w_k1_ADM_s106
                                 + 0.0350746687984652 w_k1_ADM_s107
                                 + 0.0611689714078768 w_k1_ADM_s110
                                 + 0.0714187614607506 w_k1_ADM_s111
                                 + 0.0384831391582472 w_k1_ADM_s112
                                 + 0.0317482574867099 w_k1_ADM_s113
                                 + 0.0461046458654802 w_k1_ADM_s114
                                 + 0.0619143154612945 w_k1_ADM_s115
                                 + 0.0375220925816947 w_k1_ADM_s116
                                 + 0.0663962546617311 w_k1_ADM_s117
                                 + 0.03013090647018 w_k1_AXP_s000
                                 + 0.0500089148086972 w_k1_AXP_s001
                                 + 0.0149934842207739 w_k1_AXP_s002
                                 + 0.0233262494568941 w_k1_AXP_s003
                                 + 0.0191781592251431 w_k1_AXP_s004
                                 + 0.0257129456079912 w_k1_AXP_s005
                                 + 0.0218496922216262 w_k1_AXP_s006
                                 + 0.0566775394756397 w_k1_AXP_s007
                                 + 0.00882601021847427 w_k1_AXP_s010
                                 + 0.00795569460728018 w_k1_AXP_s011
                                 + 0.0090726837720359 w_k1_AXP_s012
                                 + 0.01060599070819 w_k1_AXP_s013
                                 + 0.00488903303676555 w_k1_AXP_s014
                                 + 0.0103204828591025 w_k1_AXP_s015
                                 + 0.0105819910672811 w_k1_AXP_s016
                                 + 0.0185234718494969 w_k1_AXP_s017
                                 + 0.016545514761139 w_k1_AXP_s100
                                 + 0.0158034719956854 w_k1_AXP_s101
                                 + 0.0141047436827014 w_k1_AXP_s102
                                 + 0.0447977996167569 w_k1_AXP_s103
                                 + 0.0412651941177879 w_k1_AXP_s104
                                 + 0.0338704374477509 w_k1_AXP_s105
                                 + 0.0657182388609494 w_k1_AXP_s106
                                 + 0.0392134879735343 w_k1_AXP_s107
                                 + 0.0707640054568242 w_k1_AXP_s110
                                 + 0.0759641506555188 w_k1_AXP_s111
                                 + 0.0399828412176829 w_k1_AXP_s112
                                 + 0.0374665930350098 w_k1_AXP_s113
                                 + 0.0421171325492695 w_k1_AXP_s114
                                 + 0.0597032949104872 w_k1_AXP_s115
                                 + 0.039371207420787 w_k1_AXP_s116
                                 + 0.0660409772555991 w_k1_AXP_s117
                                 + 0.0332691684374376 w_k1_DIS_s000
                                 + 0.0527173331107329 w_k1_DIS_s001
                                 + 0.0162435856753341 w_k1_DIS_s002
                                 + 0.0229857385485264 w_k1_DIS_s003
                                 + 0.0205301614287836 w_k1_DIS_s004
                                 + 0.0287254786088174 w_k1_DIS_s005
                                 + 0.0220826292134608 w_k1_DIS_s006
                                 + 0.0585865367045143 w_k1_DIS_s007
                                 + 0.00858360823157727 w_k1_DIS_s010
                                 + 0.00800003075788231 w_k1_DIS_s011
                                 + 0.00928621965986506 w_k1_DIS_s012
                                 + 0.0108238590170541 w_k1_DIS_s013
                                 + 0.00514651458219446 w_k1_DIS_s014
                                 + 0.0103266332200478 w_k1_DIS_s015
                                 + 0.0113575593483378 w_k1_DIS_s016
                                 + 0.0181199528514532 w_k1_DIS_s017
                                 + 0.0171389732993121 w_k1_DIS_s100
                                 + 0.0161787652642289 w_k1_DIS_s101
                                 + 0.0161985445840824 w_k1_DIS_s102
                                 + 0.0433585980791691 w_k1_DIS_s103
                                 + 0.038341058567537 w_k1_DIS_s104
                                 + 0.0352353599020796 w_k1_DIS_s105
                                 + 0.0691468972631578 w_k1_DIS_s106
                                 + 0.0438484555852322 w_k1_DIS_s107
                                 + 0.0684897593178991 w_k1_DIS_s110
                                 + 0.067767919891793 w_k1_DIS_s111
                                 + 0.0384819706156348 w_k1_DIS_s112
                                 + 0.0349247180850364 w_k1_DIS_s113
                                 + 0.0447368904123879 w_k1_DIS_s114
                                 + 0.0575960752598036 w_k1_DIS_s115
                                 + 0.0399998351473217 w_k1_DIS_s116
                                 + 0.0664653134229433 w_k1_DIS_s117
                                 + 0.0327423577016041 w_k1_FDX_s000
                                 + 0.0503566046734433 w_k1_FDX_s001
                                 + 0.0151922685386399 w_k1_FDX_s002
                                 + 0.0223082695132431 w_k1_FDX_s003
                                 + 0.0202289475184968 w_k1_FDX_s004
                                 + 0.0264561076943072 w_k1_FDX_s005
                                 + 0.0218906461573609 w_k1_FDX_s006
                                 + 0.0549493251632905 w_k1_FDX_s007
                                 + 0.00825938968021267 w_k1_FDX_s010
                                 + 0.00811054682301931 w_k1_FDX_s011
                                 + 0.00880934042344431 w_k1_FDX_s012
                                 + 0.011026832293517 w_k1_FDX_s013
                                 + 0.00491929134448466 w_k1_FDX_s014
                                 + 0.0104055792600782 w_k1_FDX_s015
                                 + 0.010702622322223 w_k1_FDX_s016
                                 + 0.0180589621845834 w_k1_FDX_s017
                                 + 0.0167075164095405 w_k1_FDX_s100
                                 + 0.0153775474068537 w_k1_FDX_s101
                                 + 0.0160792928806938 w_k1_FDX_s102
                                 + 0.0432161638406197 w_k1_FDX_s103
                                 + 0.0357508529808684 w_k1_FDX_s104
                                 + 0.033332973808323 w_k1_FDX_s105
                                 + 0.0653770543529332 w_k1_FDX_s106
                                 + 0.0382400576615571 w_k1_FDX_s107
                                 + 0.0661449155398711 w_k1_FDX_s110
                                 + 0.0702598379733157 w_k1_FDX_s111
                                 + 0.0401735076981335 w_k1_FDX_s112
                                 + 0.0330371848935997 w_k1_FDX_s113
                                 + 0.045704689485226 w_k1_FDX_s114
                                 + 0.0576176092128425 w_k1_FDX_s115
                                 + 0.0377343558671868 w_k1_FDX_s116
                                 + 0.064959949431464 w_k1_FDX_s117
                                 + 0.0323905099847569 w_k1_IBM_s000
                                 + 0.0539412530476986 w_k1_IBM_s001
                                 + 0.0160217610091494 w_k1_IBM_s002
                                 + 0.0228407378577333 w_k1_IBM_s003
                                 + 0.0217966135537562 w_k1_IBM_s004
                                 + 0.0263753936163761 w_k1_IBM_s005
                                 + 0.0232924731479579 w_k1_IBM_s006
                                 + 0.0577250657664314 w_k1_IBM_s007
                                 + 0.00858052740930652 w_k1_IBM_s010
                                 + 0.00884406693414943 w_k1_IBM_s011
                                 + 0.00949735919636032 w_k1_IBM_s012
                                 + 0.0114720090989951 w_k1_IBM_s013
                                 + 0.005334298449055 w_k1_IBM_s014
                                 + 0.00972629027742627 w_k1_IBM_s015
                                 + 0.0107925344227618 w_k1_IBM_s016
                                 + 0.0188056521427387 w_k1_IBM_s017
                                 + 0.0165613544751784 w_k1_IBM_s100
                                 + 0.0151614121898775 w_k1_IBM_s101
                                 + 0.0162408978398348 w_k1_IBM_s102
                                 + 0.0447316643493986 w_k1_IBM_s103
                                 + 0.0349986080286411 w_k1_IBM_s104
                                 + 0.0362721333311738 w_k1_IBM_s105
                                 + 0.0674236220777614 w_k1_IBM_s106
                                 + 0.0409292407278656 w_k1_IBM_s107
                                 + 0.0695448615611394 w_k1_IBM_s110
                                 + 0.0690349482365461 w_k1_IBM_s111
                                 + 0.0409632201464967 w_k1_IBM_s112
                                 + 0.0320091437208733 w_k1_IBM_s113
                                 + 0.0460165733371434 w_k1_IBM_s114
                                 + 0.0594522150761958 w_k1_IBM_s115
                                 + 0.0389504951065972 w_k1_IBM_s116
                                 + 0.0658762254489197 w_k1_IBM_s117
                                 + 0.0315114808951437 w_k1_INTC_s000
                                 + 0.0476974998750712 w_k1_INTC_s001
                                 + 0.0172663570066863 w_k1_INTC_s002
                                 + 0.0236850211588427 w_k1_INTC_s003
                                 + 0.018579799881127 w_k1_INTC_s004
                                 + 0.0241436652667293 w_k1_INTC_s005
                                 + 0.0226273877686906 w_k1_INTC_s006
                                 + 0.0547409671224354 w_k1_INTC_s007
                                 + 0.00829941790507716 w_k1_INTC_s010
                                 + 0.00907918323737101 w_k1_INTC_s011
                                 + 0.00905690900893212 w_k1_INTC_s012
                                 + 0.0106177071765542 w_k1_INTC_s013
                                 + 0.0045769393286204 w_k1_INTC_s014
                                 + 0.0109270952359998 w_k1_INTC_s015
                                 + 0.0101975722356757 w_k1_INTC_s016
                                 + 0.0174242299590324 w_k1_INTC_s017
                                 + 0.0170886376121708 w_k1_INTC_s100
                                 + 0.0152941990367794 w_k1_INTC_s101
                                 + 0.0150448100524238 w_k1_INTC_s102
                                 + 0.0431091506535849 w_k1_INTC_s103
                                 + 0.0376759890606579 w_k1_INTC_s104
                                 + 0.0371163186499555 w_k1_INTC_s105
                                 + 0.06248531543592 w_k1_INTC_s106
                                 + 0.0433594892046921 w_k1_INTC_s107
                                 + 0.0610685110812033 w_k1_INTC_s110
                                 + 0.074894949726152 w_k1_INTC_s111
                                 + 0.0372999375749761 w_k1_INTC_s112
                                 + 0.0296337014911813 w_k1_INTC_s113
                                 + 0.0411783758334873 w_k1_INTC_s114
                                 + 0.0561727048946237 w_k1_INTC_s115
                                 + 0.0353761458671429 w_k1_INTC_s116
                                 + 0.0549887358845815 w_k1_INTC_s117
                                 + 0.0322034323788958 w_k1_KO_s000
                                 + 0.055648718170051 w_k1_KO_s001
                                 + 0.015183589422908 w_k1_KO_s002
                                 + 0.0242210007877128 w_k1_KO_s003
                                 + 0.0250022876575336 w_k1_KO_s004
                                 + 0.0261523058574145 w_k1_KO_s005
                                 + 0.0234856462017227 w_k1_KO_s006
                                 + 0.056738689891952 w_k1_KO_s007
                                 + 0.00863090083432249 w_k1_KO_s010
                                 + 0.00830571598841613 w_k1_KO_s011
                                 + 0.010268769631311 w_k1_KO_s012
                                 + 0.0112357345075152 w_k1_KO_s013
                                 + 0.00510430034960869 w_k1_KO_s014
                                 + 0.0119489538352401 w_k1_KO_s015
                                 + 0.0103605966605207 w_k1_KO_s016
                                 + 0.0178443302811758 w_k1_KO_s017
                                 + 0.016565369379332 w_k1_KO_s100
                                 + 0.0160740371191302 w_k1_KO_s101
                                 + 0.0154697298719502 w_k1_KO_s102
                                 + 0.0387439962994105 w_k1_KO_s103
                                 + 0.04087033220498 w_k1_KO_s104
                                 + 0.0353469628142682 w_k1_KO_s105
                                 + 0.0638894655227758 w_k1_KO_s106
                                 + 0.0391225687971816 w_k1_KO_s107
                                 + 0.0713710665035263 w_k1_KO_s110
                                 + 0.0635550174435718 w_k1_KO_s111
                                 + 0.0427005727753379 w_k1_KO_s112
                                 + 0.0325088380844298 w_k1_KO_s113
                                 + 0.0439410553963558 w_k1_KO_s114
                                 + 0.0617576695588766 w_k1_KO_s115
                                 + 0.0375227502681254 w_k1_KO_s116
                                 + 0.0623461595963877 w_k1_KO_s117
                                 + 0.028688631389121 w_k1_MAT_s000
                                 + 0.068146589294438 w_k1_MAT_s001
                                 + 0.0137912492446421 w_k1_MAT_s002
                                 + 0.0217074965261958 w_k1_MAT_s003
                                 + 0.0187677621195891 w_k1_MAT_s004
                                 + 0.0269079081532766 w_k1_MAT_s005
                                 + 0.0170736743133854 w_k1_MAT_s006
                                 + 0.0536425860637773 w_k1_MAT_s007
                                 + 0.00974984972595957 w_k1_MAT_s010
                                 + 0.00756784011567571 w_k1_MAT_s011
                                 + 0.00970710844597661 w_k1_MAT_s012
                                 + 0.0135207491289582 w_k1_MAT_s013
                                 + 0.00391443835246761 w_k1_MAT_s014
                                 + 0.0102373057995358 w_k1_MAT_s015
                                 + 0.0112684906923894 w_k1_MAT_s016
                                 + 0.0180484731714616 w_k1_MAT_s017
                                 + 0.0126925762622735 w_k1_MAT_s100
                                 + 0.0191295651268944 w_k1_MAT_s101
                                 + 0.0176864039404745 w_k1_MAT_s102
                                 + 0.0376989665643549 w_k1_MAT_s103
                                 + 0.0414500819430604 w_k1_MAT_s104
                                 + 0.0404009684199471 w_k1_MAT_s105
                                 + 0.0748185459544123 w_k1_MAT_s106
                                 + 0.0388508918501136 w_k1_MAT_s107
                                 + 0.0724235927540225 w_k1_MAT_s110
                                 + 0.0799651800834412 w_k1_MAT_s111
                                 + 0.049874487512596 w_k1_MAT_s112
                                 + 0.0347775245099152 w_k1_MAT_s113
                                 + 0.0437181492043973 w_k1_MAT_s114
                                 + 0.0741706273297847 w_k1_MAT_s115
                                 + 0.0359561714495374 w_k1_MAT_s116
                                 + 0.0602563343406791 w_k1_MAT_s117
                                 + 0.031358967025256 w_k1_MSFT_s000
                                 + 0.0475048401093568 w_k1_MSFT_s001
                                 + 0.0151947491812712 w_k1_MSFT_s002
                                 + 0.0240690827318757 w_k1_MSFT_s003
                                 + 0.0215604270851895 w_k1_MSFT_s004
                                 + 0.0239937382330627 w_k1_MSFT_s005
                                 + 0.0196831589461906 w_k1_MSFT_s006
                                 + 0.0578224813357091 w_k1_MSFT_s007
                                 + 0.00815510690096327 w_k1_MSFT_s010
                                 + 0.00807066803465284 w_k1_MSFT_s011
                                 + 0.00967808948462267 w_k1_MSFT_s012
                                 + 0.011115033114606 w_k1_MSFT_s013
                                 + 0.00478966508414191 w_k1_MSFT_s014
                                 + 0.011011709288506 w_k1_MSFT_s015
                                 + 0.0109337888708407 w_k1_MSFT_s016
                                 + 0.0167296754230462 w_k1_MSFT_s017
                                 + 0.0169854876423467 w_k1_MSFT_s100
                                 + 0.0160317000506503 w_k1_MSFT_s101
                                 + 0.0158154897852158 w_k1_MSFT_s102
                                 + 0.0430993828798665 w_k1_MSFT_s103
                                 + 0.0353207071192218 w_k1_MSFT_s104
                                 + 0.0349256317886886 w_k1_MSFT_s105
                                 + 0.0675299850982241 w_k1_MSFT_s106
                                 + 0.0400751892530289 w_k1_MSFT_s107
                                 + 0.0683798737711919 w_k1_MSFT_s110
                                 + 0.0647949557291864 w_k1_MSFT_s111
                                 + 0.0397925311294032 w_k1_MSFT_s112
                                 + 0.0322038817959588 w_k1_MSFT_s113
                                 + 0.0413000236069329 w_k1_MSFT_s114
                                 + 0.0558391609736256 w_k1_MSFT_s115
                                 + 0.0402244861683189 w_k1_MSFT_s116
                                 + 0.0633047339400423 w_k1_MSFT_s117
                                 + 0.0332636468281935 w_k1_PG_s000
                                 + 0.0523995187523867 w_k1_PG_s001
                                 + 0.0155715228451472 w_k1_PG_s002
                                 + 0.023327790725936 w_k1_PG_s003
                                 + 0.019483685749576 w_k1_PG_s004
                                 + 0.0256526705310314 w_k1_PG_s005
                                 + 0.0220760515916443 w_k1_PG_s006
                                 + 0.0539518883032131 w_k1_PG_s007
                                 + 0.00810600980969599 w_k1_PG_s010
                                 + 0.00769896145080874 w_k1_PG_s011
                                 + 0.00934621028358064 w_k1_PG_s012
                                 + 0.011485769137679 w_k1_PG_s013
                                 + 0.00488881287054513 w_k1_PG_s014
                                 + 0.0105153693543196 w_k1_PG_s015
                                 + 0.0113002831510155 w_k1_PG_s016
                                 + 0.0187604475229508 w_k1_PG_s017
                                 + 0.0169345770777757 w_k1_PG_s100
                                 + 0.01522282022146 w_k1_PG_s101
                                 + 0.0160115687915488 w_k1_PG_s102
                                 + 0.0433339296497756 w_k1_PG_s103
                                 + 0.0359728184831631 w_k1_PG_s104
                                 + 0.0336880775273933 w_k1_PG_s105
                                 + 0.0708777112412661 w_k1_PG_s106
                                 + 0.0390250074953041 w_k1_PG_s107
                                 + 0.0736052227666942 w_k1_PG_s110
                                 + 0.0666835995699597 w_k1_PG_s111
                                 + 0.0390708160941408 w_k1_PG_s112
                                 + 0.0339716377260487 w_k1_PG_s113
                                 + 0.041898417385499 w_k1_PG_s114
                                 + 0.061680800621497 w_k1_PG_s115
                                 + 0.0399401524525205 w_k1_PG_s116
                                 + 0.0718043331690021 w_k1_PG_s117 >= 1.14
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
