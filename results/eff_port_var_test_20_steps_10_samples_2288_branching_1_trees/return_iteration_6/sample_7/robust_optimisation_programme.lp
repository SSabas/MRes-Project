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
 w_k1_ADM_s0_return_constraint:  0.962204 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.04019 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.878359 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.21417 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.899082 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.26507 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.08068 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.49382 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.23507 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.918007 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.870656 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.13562 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.995488 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.18569 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.52462 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.891827 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.810621 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.0322 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.879299 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.890486 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.00545 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.14535 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.16028 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.798989 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.15394 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.990753 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.13523 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.00052 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.12322 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.65013 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.882634 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.27232 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.903005 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.13756 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.41049 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.958646 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.15251 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.972394 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  0.960633 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.11551 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.843938 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.28693 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.864061 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.41361 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.887371 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.49968 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.47163 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.880784 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.669542 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.941653 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.07467 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.1958 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.51738 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.819974 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.67161 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.867564 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.609237 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.810622 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.00395 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.13009 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.911362 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.651027 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.24194 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.00045 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.19179 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.987227 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.44709 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.78263 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.784748 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.64544 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.963943 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.41783 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.22205 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.888157 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.08944 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.772986 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  0.95146 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.20369 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.776227 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.25825 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.851398 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.49146 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.04824 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.61196 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.6198 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.844612 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.600557 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.03028 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.91973 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.32011 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.83058 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.877815 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.644449 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.28602 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.820233 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.734395 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.94342 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.2486 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.15999 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.562671 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.2789 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.852058 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.96288 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.844774 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.34597 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.53579 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.727092 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.63503 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.954494 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.10147 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.50465 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.919824 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.38156 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.767807 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.0028 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  1.0676 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.833148 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.12868 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.883084 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.37763 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.03559 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.57563 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.36619 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.828117 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.73724 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.18092 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.950707 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.17066 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.48104 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.0506 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.751206 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.02695 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.900582 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.797714 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.94577 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.20386 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.10174 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.700898 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.19273 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.03472 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.16981 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.880682 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.17455 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.52436 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.738329 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.25837 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.917627 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.06717 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.30811 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.788593 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.08593 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.902334 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.857816 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.25414 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.864009 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.21239 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.985424 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.205 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.91057 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.883782 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.37521 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.53599 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.779371 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.613727 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.16138 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.13981 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.40511 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.652188 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.735623 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.17652 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.692863 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.0932 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.30857 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.825563 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.690176 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.687843 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.06011 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.773463 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.627717 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.05202 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.72114 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.881882 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.829906 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.89483 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.07246 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.15126 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.907509 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.54817 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.39096 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.633252 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.01836 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.05595 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.825378 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.19422 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.966991 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.23212 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.04026 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.38965 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.17268 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.870089 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.751781 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.12099 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.0324 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.14932 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.12009 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.771525 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.74572 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.02893 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.823904 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.83252 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.914445 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.974058 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.897094 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.630464 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.0841 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.09625 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.927007 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.814354 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.05872 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.03359 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.963479 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.17192 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.985147 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.909568 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.18061 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.853074 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.24966 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.840792 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.909635 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.16413 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.850811 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.26576 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.964135 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.25943 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.974115 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.15663 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.43777 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.06835 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.769979 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.859439 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.00071 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.08177 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.30639 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.735508 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.709974 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.08017 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.872917 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.839188 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.10193 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.97797 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.852746 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.721243 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.24787 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.832148 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.800271 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.946949 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.37843 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.978118 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.762213 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.81128 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.04737 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.09831 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.21251 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.29974 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.26777 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.801176 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.805535 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.33161 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.894676 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.49464 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.862296 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.1697 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.819569 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.20699 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.5545 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.26903 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.699775 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.731356 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.03045 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.19011 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.47924 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.578563 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.710464 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.00718 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.644917 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.972875 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.2658 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.718608 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.629122 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.667318 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.17892 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.695607 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.85193 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.1319 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 2.04165 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.26057 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.754887 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 2.41103 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.08229 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.57896 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.895921 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.34453 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.44681 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.716182 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 0.9671 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.03434 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.893968 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.19363 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.955578 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.34216 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.10463 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.40009 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.20832 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.867243 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.688979 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.09599 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.889983 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.22927 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.35586 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.905492 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.798129 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.12537 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.944497 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.80748 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.880025 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.10449 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.09421 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.648975 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.17803 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.00257 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.968117 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.95058 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.18016 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.30167 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.888255 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.127 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.850904 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.07869 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.27703 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.834895 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.15222 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.927361 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   0.986271 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.06678 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.869782 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.12508 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.967351 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.22548 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.03438 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.21479 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.17617 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.978667 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.80593 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.991773 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.96392 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.08456 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.35327 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.943676 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.790653 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.99056 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.905739 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.844543 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.08142 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.02947 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.964498 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.751532 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.13834 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.918029 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.879684 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.935897 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.15565 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.25187 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.857454 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.31448 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.883349 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.06522 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.08156 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.05142 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.24386 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.811921 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.47499880129992 z_k1_s0000
                                 - 1.57299751989408 z_k1_s0001
                                 - 0.772995305968662 z_k1_s0002
                                 - 0.340998892356181 z_k1_s0003
                                 - 0.589000556790451 z_k1_s0004
                                 - 0.692996171373221 z_k1_s0005
                                 - 0.355998865942464 z_k1_s0006
                                 - 0.633999390752198 z_k1_s0007
                                 - 0.152000035492012 z_k1_s0010
                                 - 0.14400008749771 z_k1_s0011
                                 - 0.240000145829517 z_k1_s0012
                                 - 0.180999825646349 z_k1_s0013
                                 - 0.252000153120993 z_k1_s0014
                                 - 0.114999956143279 z_k1_s0015
                                 - 0.26499987668668 z_k1_s0016
                                 - 0.356999363921172 z_k1_s0017
                                 - 0.183999842697273 z_k1_s0020
                                 - 0.28699883510983 z_k1_s0021
                                 - 0.304998915395381 z_k1_s0022
                                 - 0.175999807014806 z_k1_s0023
                                 - 0.26199872360212 z_k1_s0024
                                 - 0.348999111648951 z_k1_s0025
                                 - 0.418999423870539 z_k1_s0026
                                 - 0.462999620124108 z_k1_s0027
                                 - 0.530000352667071 z_k1_s0030
                                 - 0.619001669963081 z_k1_s0031
                                 - 0.337000678187453 z_k1_s0032
                                 - 0.284000642920746 z_k1_s0033
                                 - 0.607999807956341 z_k1_s0034
                                 - 0.22900061280814 z_k1_s0035
                                 - 0.201000341264776 z_k1_s0036
                                 - 0.628999925688668 z_k1_s0037
                                 - 0.665000172944466 z_k1_s0040
                                 - 0.711001580992968 z_k1_s0041
                                 - 0.404000370519877 z_k1_s0042
                                 - 0.460000350673065 z_k1_s0043
                                 - 1.48899810631126 z_k1_s0044
                                 - 0.321000388260677 z_k1_s0045
                                 - 1.4079987537938 z_k1_s0046
                                 - 1.08000008424047 z_k1_s0047
                                 - 0.198999795431132 z_k1_s0050
                                 - 0.251999776858993 z_k1_s0051
                                 - 0.589000981442644 z_k1_s0052
                                 - 0.582001015999316 z_k1_s0053
                                 - 0.229999739622936 z_k1_s0054
                                 - 0.573000088144381 z_k1_s0055
                                 - 0.562998679083641 z_k1_s0056
                                 - 0.414998923567952 z_k1_s0057
                                 - 0.176000114609259 z_k1_s0060
                                 - 0.610002071074044 z_k1_s0061
                                 - 0.381001704971373 z_k1_s0062
                                 - 0.45600037960137 z_k1_s0063
                                 - 0.201000157753223 z_k1_s0064
                                 - 0.509002449597149 z_k1_s0065
                                 - 0.633999028312234 z_k1_s0066
                                 - 0.670001738178989 z_k1_s0067
                                 - 0.15499991894958 z_k1_s0070
                                 - 0.243000033764798 z_k1_s0071
                                 - 0.483999898283824 z_k1_s0072
                                 - 0.298998907047421 z_k1_s0073
                                 - 0.57900060545924 z_k1_s0074
                                 - 0.162000022509865 z_k1_s0075
                                 - 0.255999178662655 z_k1_s0076
                                 - 0.266000024891509 z_k1_s0077
                                 - 0.365998937154333 z_k1_s0100
                                 - 1.03299818760724 z_k1_s0101
                                 - 0.271999212190646 z_k1_s0102
                                 - 0.528998102431695 z_k1_s0103
                                 - 0.276999193540853 z_k1_s0104
                                 - 0.39899707573041 z_k1_s0105
                                 - 0.323999244972184 z_k1_s0106
                                 - 0.578997915933765 z_k1_s0107
                                 - 0.200000077289716 z_k1_s0110
                                 - 0.627998997633666 z_k1_s0111
                                 - 0.161999967180667 z_k1_s0112
                                 - 0.27400093743894 z_k1_s0113
                                 - 0.215999956240889 z_k1_s0114
                                 - 0.127999940409415 z_k1_s0115
                                 - 0.132000023747212 z_k1_s0116
                                 - 0.532000178326644 z_k1_s0117
                                 - 0.368999467056742 z_k1_s0120
                                 - 0.314999313585609 z_k1_s0121
                                 - 0.481999531358096 z_k1_s0122
                                 - 0.39399825385218 z_k1_s0123
                                 - 0.880998272567518 z_k1_s0124
                                 - 0.542998177134905 z_k1_s0125
                                 - 0.47899923894375 z_k1_s0126
                                 - 0.186999612207915 z_k1_s0127
                                 - 0.218000741908818 z_k1_s0130
                                 - 0.248999622749029 z_k1_s0131
                                 - 0.435999629817585 z_k1_s0132
                                 - 0.0950000749749368 z_k1_s0133
                                 - 0.205999799583537 z_k1_s0134
                                 - 0.34299927929685 z_k1_s0135
                                 - 0.16800002915357 z_k1_s0136
                                 - 0.138999945767535 z_k1_s0137
                                 - 0.417000062524937 z_k1_s0140
                                 - 0.283000012293894 z_k1_s0141
                                 - 0.158000223362017 z_k1_s0142
                                 - 0.155000083895615 z_k1_s0143
                                 - 0.105000011222825 z_k1_s0144
                                 - 0.167000484661183 z_k1_s0145
                                 - 0.191000657792158 z_k1_s0146
                                 - 0.0949999652682584 z_k1_s0147
                                 - 0.521999857903067 z_k1_s0150
                                 - 0.945003099454171 z_k1_s0151
                                 - 0.339001135639179 z_k1_s0152
                                 - 0.640999471669297 z_k1_s0153
                                 - 0.826003485687942 z_k1_s0154
                                 - 0.9820039022579 z_k1_s0155
                                 - 0.230000524573417 z_k1_s0156
                                 - 0.5069988578457 z_k1_s0157
                                 - 0.705002464759555 z_k1_s0160
                                 - 0.555000437146172 z_k1_s0161
                                 - 0.988002977736569 z_k1_s0162
                                 - 0.233999943279353 z_k1_s0163
                                 - 1.01600247599198 z_k1_s0164
                                 - 0.531000205384074 z_k1_s0165
                                 - 0.695001403024891 z_k1_s0166
                                 - 1.06699804613411 z_k1_s0167
                                 - 1.0550016058209 z_k1_s0170
                                 - 1.51699558938978 z_k1_s0171
                                 - 0.306999235240602 z_k1_s0172
                                 - 1.21899524235776 z_k1_s0173
                                 - 0.378999474696443 z_k1_s0174
                                 - 0.795995504227356 z_k1_s0175
                                 - 1.04600131241429 z_k1_s0176
                                 - 0.706997286755421 z_k1_s0177
                                 - 0.134000017338424 z_k1_s1000
                                 - 0.175000510814969 z_k1_s1001
                                 - 0.112999897240644 z_k1_s1002
                                 - 0.235999613670774 z_k1_s1003
                                 - 0.224000007043342 z_k1_s1004
                                 - 0.169999572220515 z_k1_s1005
                                 - 0.298999385964279 z_k1_s1006
                                 - 0.118999994554278 z_k1_s1007
                                 - 0.238000443363381 z_k1_s1010
                                 - 0.883002523251336 z_k1_s1011
                                 - 0.65600334317022 z_k1_s1012
                                 - 0.350000728540468 z_k1_s1013
                                 - 0.50399976534288 z_k1_s1014
                                 - 1.00300329347801 z_k1_s1015
                                 - 0.4840018054775 z_k1_s1016
                                 - 0.21900042983611 z_k1_s1017
                                 - 0.671001757272385 z_k1_s1020
                                 - 0.769996342947807 z_k1_s1021
                                 - 0.601998743664815 z_k1_s1022
                                 - 0.675997196457193 z_k1_s1023
                                 - 0.580998962129618 z_k1_s1024
                                 - 1.04099728622721 z_k1_s1025
                                 - 0.400999248831361 z_k1_s1026
                                 - 1.78799501896731 z_k1_s1027
                                 - 0.38099900162984 z_k1_s1030
                                 - 0.302000774728745 z_k1_s1031
                                 - 0.179999860563392 z_k1_s1032
                                 - 0.212999856309994 z_k1_s1033
                                 - 0.344000614333656 z_k1_s1034
                                 - 0.133999924610499 z_k1_s1035
                                 - 0.418998957986569 z_k1_s1036
                                 - 0.157999863398991 z_k1_s1037
                                 - 0.0999998654268189 z_k1_s1040
                                 - 0.250000424566282 z_k1_s1041
                                 - 0.124999907883447 z_k1_s1042
                                 - 0.223000171721332 z_k1_s1043
                                 - 0.106999840786711 z_k1_s1044
                                 - 0.379998879822524 z_k1_s1045
                                 - 0.158000183093976 z_k1_s1046
                                 - 0.178999195974572 z_k1_s1047
                                 - 1.1540016755373 z_k1_s1050
                                 - 0.594000935605944 z_k1_s1051
                                 - 0.722002913735277 z_k1_s1052
                                 - 1.9079998075486 z_k1_s1053
                                 - 0.781997465507024 z_k1_s1054
                                 - 0.592000882713272 z_k1_s1055
                                 - 0.531000324738035 z_k1_s1056
                                 - 0.75200018962115 z_k1_s1057
                                 - 0.288000609705028 z_k1_s1060
                                 - 0.82300094849764 z_k1_s1061
                                 - 0.446000552625992 z_k1_s1062
                                 - 1.16300387531914 z_k1_s1063
                                 - 0.237000426982159 z_k1_s1064
                                 - 0.841003485532679 z_k1_s1065
                                 - 0.922999095559014 z_k1_s1066
                                 - 0.405000697209973 z_k1_s1067
                                 - 0.158999522575121 z_k1_s1070
                                 - 0.158999522575121 z_k1_s1071
                                 - 0.215000383758465 z_k1_s1072
                                 - 0.237999800030511 z_k1_s1073
                                 - 0.240999791022494 z_k1_s1074
                                 - 0.113999966495266 z_k1_s1075
                                 - 0.237999800030511 z_k1_s1076
                                 - 0.179999459519005 z_k1_s1077
                                 - 0.170999689790906 z_k1_s1100
                                 - 0.0989998005421324 z_k1_s1101
                                 - 0.135999665003423 z_k1_s1102
                                 - 0.0869998819005781 z_k1_s1103
                                 - 0.0889998473742012 z_k1_s1104
                                 - 0.14100020768656 z_k1_s1105
                                 - 0.275999764153355 z_k1_s1106
                                 - 0.258999177028848 z_k1_s1107
                                 - 0.0600001476412271 z_k1_s1110
                                 - 0.0600001476412271 z_k1_s1111
                                 - 0.0770003629499779 z_k1_s1112
                                 - 0.129000236798451 z_k1_s1113
                                 - 0.106000109345816 z_k1_s1114
                                 - 0.168000325435231 z_k1_s1115
                                 - 0.0410000837847988 z_k1_s1116
                                 - 0.0920005098105402 z_k1_s1117
                                 - 0.152999952390466 z_k1_s1120
                                 - 0.454000028389928 z_k1_s1121
                                 - 0.247999944036824 z_k1_s1122
                                 - 0.223000800129474 z_k1_s1123
                                 - 0.230999228260454 z_k1_s1124
                                 - 0.177999907497423 z_k1_s1125
                                 - 0.19099998960899 z_k1_s1126
                                 - 0.350000182697004 z_k1_s1127
                                 - 0.0610000621358618 z_k1_s1130
                                 - 0.0420000748345713 z_k1_s1131
                                 - 0.0740001434894986 z_k1_s1132
                                 - 0.119999951955562 z_k1_s1133
                                 - 0.0410000497765737 z_k1_s1134
                                 - 0.0280000295230199 z_k1_s1135
                                 - 0.133000216609448 z_k1_s1136
                                 - 0.11200011809208 z_k1_s1137
                                 - 0.122000544095618 z_k1_s1140
                                 - 0.110000418853154 z_k1_s1141
                                 - 0.240000681842367 z_k1_s1142
                                 - 0.115000379887169 z_k1_s1143
                                 - 0.260000890579806 z_k1_s1144
                                 - 0.525002289095698 z_k1_s1145
                                 - 0.155000615061357 z_k1_s1146
                                 - 0.296001630908577 z_k1_s1147
                                 - 0.0740001450430481 z_k1_s1150
                                 - 0.0760001307793136 z_k1_s1151
                                 - 0.103999931087031 z_k1_s1152
                                 - 0.0800001863519976 z_k1_s1153
                                 - 0.108000575360785 z_k1_s1154
                                 - 0.119000034359405 z_k1_s1155
                                 - 0.0750000958611044 z_k1_s1156
                                 - 0.205000009719893 z_k1_s1157
                                 - 0.0550000964866811 z_k1_s1160
                                 - 0.157000017363234 z_k1_s1161
                                 - 0.0740001398402837 z_k1_s1162
                                 - 0.0540001485997683 z_k1_s1163
                                 - 0.0770000524011727 z_k1_s1164
                                 - 0.095999957954474 z_k1_s1165
                                 - 0.0330000716720388 z_k1_s1166
                                 - 0.14300019574525 z_k1_s1167
                                 - 0.236999594595699 z_k1_s1170
                                 - 0.206000473112205 z_k1_s1171
                                 - 0.322001456074834 z_k1_s1172
                                 - 0.147000338746576 z_k1_s1173
                                 - 0.484001475545886 z_k1_s1174
                                 - 0.368001664085524 z_k1_s1175
                                 - 0.195000301126804 z_k1_s1176
                                 - 0.383999699041822 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.731266 w_k1_ADM_s000
                                 + 0.836164 w_k1_AXP_s000
                                 + 0.839834 w_k1_DIS_s000
                                 + 0.828711 w_k1_FDX_s000
                                 + 1.23632 w_k1_IBM_s000
                                 + 0.916668 w_k1_INTC_s000 + 1.014 w_k1_KO_s000
                                 + 1.43249 w_k1_MAT_s000
                                 + 0.839541 w_k1_MSFT_s000
                                 + 0.895532 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.981086 w_k1_ADM_s000
                                 + 1.31226 w_k1_AXP_s000
                                 + 1.33758 w_k1_DIS_s000
                                 + 1.15963 w_k1_FDX_s000
                                 + 1.16077 w_k1_IBM_s000
                                 + 1.32787 w_k1_INTC_s000
                                 + 1.27155 w_k1_KO_s000 + 1.64588 w_k1_MAT_s000
                                 + 1.09758 w_k1_MSFT_s000
                                 + 1.11965 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.865859 w_k1_ADM_s000
                                 + 0.808427 w_k1_AXP_s000
                                 + 0.744737 w_k1_DIS_s000
                                 + 0.807358 w_k1_FDX_s000
                                 + 0.972073 w_k1_IBM_s000
                                 + 0.869154 w_k1_INTC_s000
                                 + 0.917352 w_k1_KO_s000
                                 + 0.922505 w_k1_MAT_s000
                                 + 0.758828 w_k1_MSFT_s000
                                 + 0.850334 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.857215 w_k1_ADM_s000
                                 + 1.10577 w_k1_AXP_s000
                                 + 1.14023 w_k1_DIS_s000
                                 + 1.00383 w_k1_FDX_s000
                                 + 1.58446 w_k1_IBM_s000
                                 + 1.16733 w_k1_INTC_s000
                                 + 1.56485 w_k1_KO_s000 + 1.80528 w_k1_MAT_s000
                                 + 1.13533 w_k1_MSFT_s000
                                 + 1.11688 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.983157 w_k1_ADM_s000
                                 + 1.22674 w_k1_AXP_s000
                                 + 1.04718 w_k1_DIS_s000
                                 + 1.22901 w_k1_FDX_s000
                                 + 0.753743 w_k1_IBM_s000
                                 + 1.27023 w_k1_INTC_s000
                                 + 0.909866 w_k1_KO_s000
                                 + 0.81495 w_k1_MAT_s000
                                 + 0.937747 w_k1_MSFT_s000
                                 + 0.996698 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.01355 w_k1_ADM_s000
                                 + 1.10745 w_k1_AXP_s000
                                 + 0.827715 w_k1_DIS_s000
                                 + 1.02433 w_k1_FDX_s000
                                 + 0.932462 w_k1_IBM_s000
                                 + 0.946112 w_k1_INTC_s000
                                 + 1.05309 w_k1_KO_s000 + 1.09525 w_k1_MAT_s000
                                 + 0.900792 w_k1_MSFT_s000
                                 + 0.979624 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.736291 w_k1_ADM_s000
                                 + 0.788156 w_k1_AXP_s000
                                 + 1.03216 w_k1_DIS_s000
                                 + 0.957832 w_k1_FDX_s000
                                 + 1.06044 w_k1_IBM_s000
                                 + 0.952057 w_k1_INTC_s000
                                 + 1.07324 w_k1_KO_s000 + 1.35986 w_k1_MAT_s000
                                 + 0.960483 w_k1_MSFT_s000
                                 + 1.02639 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.799105 w_k1_ADM_s000
                                 + 0.703165 w_k1_AXP_s000
                                 + 0.602005 w_k1_DIS_s000
                                 + 0.77609 w_k1_FDX_s000
                                 + 0.630797 w_k1_IBM_s000
                                 + 0.817415 w_k1_INTC_s000
                                 + 0.613728 w_k1_KO_s000
                                 + 0.685569 w_k1_MAT_s000
                                 + 0.741201 w_k1_MSFT_s000
                                 + 0.785512 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.19692 w_k1_ADM_s001
                                 + 1.09188 w_k1_AXP_s001
                                 + 0.982259 w_k1_DIS_s001
                                 + 1.0468 w_k1_FDX_s001 + 1.07742 w_k1_IBM_s001
                                 + 0.950432 w_k1_INTC_s001
                                 + 0.941152 w_k1_KO_s001
                                 + 0.972937 w_k1_MAT_s001
                                 + 0.934066 w_k1_MSFT_s001
                                 + 1.02723 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.948046 w_k1_ADM_s001
                                 + 0.836761 w_k1_AXP_s001
                                 + 0.769027 w_k1_DIS_s001
                                 + 0.792964 w_k1_FDX_s001
                                 + 1.18954 w_k1_IBM_s001
                                 + 0.999462 w_k1_INTC_s001
                                 + 1.00508 w_k1_KO_s001
                                 + 0.664395 w_k1_MAT_s001
                                 + 0.828809 w_k1_MSFT_s001
                                 + 0.915594 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.841157 w_k1_ADM_s001
                                 + 0.963656 w_k1_AXP_s001
                                 + 0.80577 w_k1_DIS_s001
                                 + 0.966865 w_k1_FDX_s001
                                 + 0.822084 w_k1_IBM_s001
                                 + 0.925577 w_k1_INTC_s001
                                 + 0.785612 w_k1_KO_s001
                                 + 0.582329 w_k1_MAT_s001
                                 + 0.844827 w_k1_MSFT_s001
                                 + 0.938469 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.07824 w_k1_ADM_s001
                                 + 1.03185 w_k1_AXP_s001
                                 + 1.45352 w_k1_DIS_s001
                                 + 1.11301 w_k1_FDX_s001
                                 + 1.72238 w_k1_IBM_s001
                                 + 0.986894 w_k1_INTC_s001
                                 + 1.55822 w_k1_KO_s001 + 1.55237 w_k1_MAT_s001
                                 + 1.14956 w_k1_MSFT_s001
                                 + 1.06897 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.52064 w_k1_ADM_s001
                                 + 1.38659 w_k1_AXP_s001
                                 + 1.49547 w_k1_DIS_s001
                                 + 1.39762 w_k1_FDX_s001
                                 + 1.10258 w_k1_IBM_s001
                                 + 1.10077 w_k1_INTC_s001
                                 + 1.23432 w_k1_KO_s001 + 1.02675 w_k1_MAT_s001
                                 + 1.228 w_k1_MSFT_s001 + 1.35089 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.915172 w_k1_ADM_s001
                                 + 0.67764 w_k1_AXP_s001
                                 + 0.952335 w_k1_DIS_s001
                                 + 0.850092 w_k1_FDX_s001
                                 + 1.09276 w_k1_IBM_s001
                                 + 0.943887 w_k1_INTC_s001
                                 + 1.2596 w_k1_KO_s001 + 0.686346 w_k1_MAT_s001
                                 + 0.911791 w_k1_MSFT_s001
                                 + 0.93028 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.772782 w_k1_ADM_s001
                                 + 0.607284 w_k1_AXP_s001
                                 + 0.728821 w_k1_DIS_s001
                                 + 0.765904 w_k1_FDX_s001
                                 + 0.765545 w_k1_IBM_s001
                                 + 0.806058 w_k1_INTC_s001
                                 + 0.783261 w_k1_KO_s001
                                 + 0.771759 w_k1_MAT_s001
                                 + 0.852137 w_k1_MSFT_s001
                                 + 0.930305 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.834016 w_k1_ADM_s001
                                 + 0.968632 w_k1_AXP_s001
                                 + 1.29306 w_k1_DIS_s001
                                 + 1.11074 w_k1_FDX_s001
                                 + 1.03024 w_k1_IBM_s001
                                 + 1.20792 w_k1_INTC_s001
                                 + 1.19294 w_k1_KO_s001
                                 + 0.574146 w_k1_MAT_s001
                                 + 1.09497 w_k1_MSFT_s001
                                 + 1.09902 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.24558 w_k1_ADM_s002
                                 + 1.74221 w_k1_AXP_s002
                                 + 1.63135 w_k1_DIS_s002 + 1.3688 w_k1_FDX_s002
                                 + 1.23377 w_k1_IBM_s002
                                 + 1.37279 w_k1_INTC_s002
                                 + 1.11555 w_k1_KO_s002 + 1.04857 w_k1_MAT_s002
                                 + 1.21859 w_k1_MSFT_s002
                                 + 1.21038 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.03245 w_k1_ADM_s002
                                 + 1.10212 w_k1_AXP_s002
                                 + 1.00055 w_k1_DIS_s002
                                 + 0.954778 w_k1_FDX_s002
                                 + 1.68129 w_k1_IBM_s002
                                 + 0.910909 w_k1_INTC_s002
                                 + 1.06169 w_k1_KO_s002 + 1.54128 w_k1_MAT_s002
                                 + 0.971287 w_k1_MSFT_s002
                                 + 1.05011 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.12646 w_k1_ADM_s002
                                 + 0.969791 w_k1_AXP_s002
                                 + 1.18243 w_k1_DIS_s002
                                 + 1.08412 w_k1_FDX_s002
                                 + 0.894555 w_k1_IBM_s002
                                 + 1.33703 w_k1_INTC_s002
                                 + 1.01508 w_k1_KO_s002
                                 + 0.790243 w_k1_MAT_s002
                                 + 1.34385 w_k1_MSFT_s002 + 1.0755 w_k1_PG_s002
                                 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.962694 w_k1_ADM_s002
                                 + 0.982315 w_k1_AXP_s002
                                 + 1.04786 w_k1_DIS_s002
                                 + 0.923514 w_k1_FDX_s002
                                 + 1.29216 w_k1_IBM_s002
                                 + 1.06656 w_k1_INTC_s002
                                 + 1.11015 w_k1_KO_s002 + 1.1569 w_k1_MAT_s002
                                 + 1.0646 w_k1_MSFT_s002 + 1.11821 w_k1_PG_s002
                                 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.870673 w_k1_ADM_s002
                                 + 0.82889 w_k1_AXP_s002
                                 + 0.546566 w_k1_DIS_s002
                                 + 0.785965 w_k1_FDX_s002
                                 + 0.76241 w_k1_IBM_s002
                                 + 0.823861 w_k1_INTC_s002
                                 + 0.578354 w_k1_KO_s002
                                 + 0.943273 w_k1_MAT_s002
                                 + 0.781575 w_k1_MSFT_s002
                                 + 0.870513 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.09694 w_k1_ADM_s002
                                 + 1.52436 w_k1_AXP_s002
                                 + 1.17548 w_k1_DIS_s002
                                 + 1.17461 w_k1_FDX_s002
                                 + 1.23996 w_k1_IBM_s002
                                 + 1.16142 w_k1_INTC_s002
                                 + 1.14039 w_k1_KO_s002 + 1.69319 w_k1_MAT_s002
                                 + 1.05295 w_k1_MSFT_s002
                                 + 1.18885 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.03053 w_k1_ADM_s002
                                 + 0.863647 w_k1_AXP_s002
                                 + 0.771675 w_k1_DIS_s002
                                 + 0.833464 w_k1_FDX_s002
                                 + 1.12163 w_k1_IBM_s002
                                 + 0.991678 w_k1_INTC_s002
                                 + 0.9716 w_k1_KO_s002 + 1.16062 w_k1_MAT_s002
                                 + 0.946697 w_k1_MSFT_s002
                                 + 0.947611 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.14593 w_k1_ADM_s002
                                 + 0.980148 w_k1_AXP_s002
                                 + 0.821384 w_k1_DIS_s002
                                 + 0.939896 w_k1_FDX_s002
                                 + 0.818431 w_k1_IBM_s002
                                 + 0.935312 w_k1_INTC_s002
                                 + 0.87984 w_k1_KO_s002 + 1.00344 w_k1_MAT_s002
                                 + 0.85029 w_k1_MSFT_s002
                                 + 0.92779 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.904405 w_k1_ADM_s003
                                 + 1.20447 w_k1_AXP_s003
                                 + 1.24296 w_k1_DIS_s003
                                 + 1.16493 w_k1_FDX_s003
                                 + 0.904124 w_k1_IBM_s003
                                 + 1.228 w_k1_INTC_s003 + 1.1537 w_k1_KO_s003
                                 + 0.857003 w_k1_MAT_s003
                                 + 1.11226 w_k1_MSFT_s003
                                 + 1.12149 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.09425 w_k1_ADM_s003
                                 + 1.03275 w_k1_AXP_s003 + 1.3218 w_k1_DIS_s003
                                 + 1.11241 w_k1_FDX_s003
                                 + 1.16655 w_k1_IBM_s003
                                 + 1.34437 w_k1_INTC_s003
                                 + 1.20498 w_k1_KO_s003 + 1.00778 w_k1_MAT_s003
                                 + 1.08669 w_k1_MSFT_s003
                                 + 1.16257 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.940569 w_k1_ADM_s003
                                 + 0.875336 w_k1_AXP_s003
                                 + 1.05895 w_k1_DIS_s003
                                 + 0.834702 w_k1_FDX_s003
                                 + 1.37028 w_k1_IBM_s003
                                 + 0.869103 w_k1_INTC_s003
                                 + 1.15832 w_k1_KO_s003 + 1.4358 w_k1_MAT_s003
                                 + 0.874554 w_k1_MSFT_s003
                                 + 0.872638 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.18091 w_k1_ADM_s003
                                 + 1.68683 w_k1_AXP_s003
                                 + 1.52212 w_k1_DIS_s003
                                 + 1.23545 w_k1_FDX_s003
                                 + 1.31627 w_k1_IBM_s003
                                 + 1.29938 w_k1_INTC_s003
                                 + 1.44961 w_k1_KO_s003 + 1.90059 w_k1_MAT_s003
                                 + 1.16856 w_k1_MSFT_s003
                                 + 1.33995 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.966104 w_k1_ADM_s003
                                 + 0.897966 w_k1_AXP_s003
                                 + 0.874839 w_k1_DIS_s003
                                 + 0.997918 w_k1_FDX_s003
                                 + 0.731871 w_k1_IBM_s003
                                 + 1.25918 w_k1_INTC_s003
                                 + 0.950062 w_k1_KO_s003
                                 + 0.587672 w_k1_MAT_s003
                                 + 1.01347 w_k1_MSFT_s003
                                 + 0.908682 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.31476 w_k1_ADM_s003
                                 + 1.3715 w_k1_AXP_s003 + 2.1276 w_k1_DIS_s003
                                 + 1.49209 w_k1_FDX_s003
                                 + 1.65173 w_k1_IBM_s003
                                 + 1.39159 w_k1_INTC_s003 + 1.8781 w_k1_KO_s003
                                 + 1.62104 w_k1_MAT_s003
                                 + 1.50239 w_k1_MSFT_s003
                                 + 1.24379 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.580211 w_k1_ADM_s003
                                 + 0.584994 w_k1_AXP_s003
                                 + 0.640769 w_k1_DIS_s003
                                 + 0.676874 w_k1_FDX_s003
                                 + 0.593932 w_k1_IBM_s003
                                 + 0.763409 w_k1_INTC_s003
                                 + 0.687011 w_k1_KO_s003
                                 + 0.453256 w_k1_MAT_s003
                                 + 0.746598 w_k1_MSFT_s003
                                 + 0.79431 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.797381 w_k1_ADM_s003
                                 + 0.647953 w_k1_AXP_s003
                                 + 0.750307 w_k1_DIS_s003
                                 + 0.757419 w_k1_FDX_s003
                                 + 0.915715 w_k1_IBM_s003
                                 + 0.906323 w_k1_INTC_s003
                                 + 0.882421 w_k1_KO_s003
                                 + 0.883498 w_k1_MAT_s003
                                 + 0.738166 w_k1_MSFT_s003
                                 + 0.780356 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.03994 w_k1_ADM_s004
                                 + 1.50327 w_k1_AXP_s004 + 1.387 w_k1_DIS_s004
                                 + 1.33756 w_k1_FDX_s004
                                 + 1.36141 w_k1_IBM_s004
                                 + 1.29002 w_k1_INTC_s004
                                 + 1.18919 w_k1_KO_s004 + 1.82293 w_k1_MAT_s004
                                 + 1.36488 w_k1_MSFT_s004
                                 + 1.25369 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.866992 w_k1_ADM_s004
                                 + 0.975472 w_k1_AXP_s004
                                 + 1.19759 w_k1_DIS_s004
                                 + 0.978066 w_k1_FDX_s004
                                 + 1.48757 w_k1_IBM_s004
                                 + 1.18936 w_k1_INTC_s004
                                 + 1.14483 w_k1_KO_s004 + 1.7129 w_k1_MAT_s004
                                 + 1.20722 w_k1_MSFT_s004
                                 + 1.15879 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.986859 w_k1_ADM_s004
                                 + 1.17472 w_k1_AXP_s004
                                 + 1.11346 w_k1_DIS_s004
                                 + 1.04659 w_k1_FDX_s004
                                 + 1.42397 w_k1_IBM_s004
                                 + 0.990614 w_k1_INTC_s004
                                 + 0.98024 w_k1_KO_s004 + 1.37508 w_k1_MAT_s004
                                 + 1.01764 w_k1_MSFT_s004
                                 + 1.03767 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.0069 w_k1_ADM_s004
                                 + 1.06714 w_k1_AXP_s004
                                 + 1.06807 w_k1_DIS_s004
                                 + 1.19356 w_k1_FDX_s004
                                 + 0.611295 w_k1_IBM_s004
                                 + 1.30475 w_k1_INTC_s004
                                 + 0.746479 w_k1_KO_s004
                                 + 0.605343 w_k1_MAT_s004
                                 + 1.15741 w_k1_MSFT_s004
                                 + 0.926697 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.910764 w_k1_ADM_s004
                                 + 1.02515 w_k1_AXP_s004
                                 + 0.926114 w_k1_DIS_s004
                                 + 0.95045 w_k1_FDX_s004
                                 + 0.969283 w_k1_IBM_s004
                                 + 1.04645 w_k1_INTC_s004
                                 + 0.92344 w_k1_KO_s004
                                 + 0.990912 w_k1_MAT_s004
                                 + 0.991514 w_k1_MSFT_s004
                                 + 0.95314 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.21832 w_k1_ADM_s004
                                 + 1.23316 w_k1_AXP_s004 + 2.0293 w_k1_DIS_s004
                                 + 1.46278 w_k1_FDX_s004
                                 + 1.06335 w_k1_IBM_s004
                                 + 1.19534 w_k1_INTC_s004
                                 + 1.22326 w_k1_KO_s004 + 1.24024 w_k1_MAT_s004
                                 + 1.51322 w_k1_MSFT_s004
                                 + 1.28729 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.694392 w_k1_ADM_s004
                                 + 0.712119 w_k1_AXP_s004
                                 + 0.602041 w_k1_DIS_s004
                                 + 0.735878 w_k1_FDX_s004
                                 + 0.848946 w_k1_IBM_s004
                                 + 0.838923 w_k1_INTC_s004
                                 + 0.708963 w_k1_KO_s004
                                 + 0.753572 w_k1_MAT_s004
                                 + 0.755947 w_k1_MSFT_s004
                                 + 0.826418 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.911771 w_k1_ADM_s004
                                 + 0.855099 w_k1_AXP_s004
                                 + 1.14373 w_k1_DIS_s004
                                 + 1.14173 w_k1_FDX_s004
                                 + 0.735096 w_k1_IBM_s004
                                 + 1.10517 w_k1_INTC_s004
                                 + 0.853321 w_k1_KO_s004
                                 + 0.74268 w_k1_MAT_s004
                                 + 1.14334 w_k1_MSFT_s004
                                 + 1.03893 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.838796 w_k1_ADM_s005
                                 + 0.599483 w_k1_AXP_s005
                                 + 0.673347 w_k1_DIS_s005
                                 + 0.733109 w_k1_FDX_s005
                                 + 0.790015 w_k1_IBM_s005
                                 + 0.752582 w_k1_INTC_s005
                                 + 0.676004 w_k1_KO_s005
                                 + 0.438082 w_k1_MAT_s005
                                 + 0.713078 w_k1_MSFT_s005
                                 + 0.793232 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.26168 w_k1_ADM_s005
                                 + 1.32499 w_k1_AXP_s005
                                 + 0.99643 w_k1_DIS_s005
                                 + 0.974037 w_k1_FDX_s005
                                 + 1.51705 w_k1_IBM_s005
                                 + 0.974689 w_k1_INTC_s005
                                 + 1.2142 w_k1_KO_s005 + 1.34933 w_k1_MAT_s005
                                 + 0.901711 w_k1_MSFT_s005
                                 + 1.13374 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.17419 w_k1_ADM_s005
                                 + 1.11756 w_k1_AXP_s005
                                 + 0.991589 w_k1_DIS_s005
                                 + 1.06713 w_k1_FDX_s005
                                 + 0.832125 w_k1_IBM_s005
                                 + 1.11985 w_k1_INTC_s005
                                 + 0.955912 w_k1_KO_s005
                                 + 0.823657 w_k1_MAT_s005
                                 + 0.954655 w_k1_MSFT_s005
                                 + 0.972617 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.06734 w_k1_ADM_s005
                                 + 1.0796 w_k1_AXP_s005 + 1.27397 w_k1_DIS_s005
                                 + 1.0811 w_k1_FDX_s005 + 1.27467 w_k1_IBM_s005
                                 + 1.09695 w_k1_INTC_s005
                                 + 1.06866 w_k1_KO_s005 + 1.06552 w_k1_MAT_s005
                                 + 1.03372 w_k1_MSFT_s005
                                 + 0.985668 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.951774 w_k1_ADM_s005
                                 + 0.840536 w_k1_AXP_s005
                                 + 0.672466 w_k1_DIS_s005
                                 + 0.869726 w_k1_FDX_s005
                                 + 0.830549 w_k1_IBM_s005
                                 + 0.734382 w_k1_INTC_s005
                                 + 0.77712 w_k1_KO_s005
                                 + 0.744601 w_k1_MAT_s005
                                 + 0.768359 w_k1_MSFT_s005
                                 + 0.879001 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.22385 w_k1_ADM_s005
                                 + 1.30201 w_k1_AXP_s005
                                 + 1.65598 w_k1_DIS_s005
                                 + 1.29628 w_k1_FDX_s005
                                 + 1.36897 w_k1_IBM_s005
                                 + 1.10021 w_k1_INTC_s005 + 1.2623 w_k1_KO_s005
                                 + 0.828047 w_k1_MAT_s005
                                 + 1.20488 w_k1_MSFT_s005
                                 + 1.18538 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.847375 w_k1_ADM_s005
                                 + 0.77424 w_k1_AXP_s005
                                 + 0.941892 w_k1_DIS_s005
                                 + 0.881548 w_k1_FDX_s005
                                 + 0.902667 w_k1_IBM_s005
                                 + 0.921349 w_k1_INTC_s005
                                 + 0.929509 w_k1_KO_s005
                                 + 0.790697 w_k1_MAT_s005
                                 + 0.90368 w_k1_MSFT_s005
                                 + 0.903667 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.794652 w_k1_ADM_s005
                                 + 0.737786 w_k1_AXP_s005
                                 + 0.696218 w_k1_DIS_s005
                                 + 0.731477 w_k1_FDX_s005
                                 + 1.1427 w_k1_IBM_s005
                                 + 0.790242 w_k1_INTC_s005
                                 + 0.93644 w_k1_KO_s005
                                 + 0.798033 w_k1_MAT_s005
                                 + 0.819527 w_k1_MSFT_s005
                                 + 0.925326 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.06527 w_k1_ADM_s006
                                 + 1.28285 w_k1_AXP_s006
                                 + 0.885783 w_k1_DIS_s006
                                 + 1.11843 w_k1_FDX_s006
                                 + 1.06918 w_k1_IBM_s006
                                 + 1.03386 w_k1_INTC_s006 + 0.8914 w_k1_KO_s006
                                 + 1.10595 w_k1_MAT_s006
                                 + 1.08496 w_k1_MSFT_s006
                                 + 0.993684 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.870579 w_k1_ADM_s006
                                 + 0.676166 w_k1_AXP_s006
                                 + 0.690987 w_k1_DIS_s006
                                 + 0.782854 w_k1_FDX_s006
                                 + 0.749905 w_k1_IBM_s006
                                 + 0.728455 w_k1_INTC_s006
                                 + 0.844836 w_k1_KO_s006
                                 + 0.931711 w_k1_MAT_s006
                                 + 0.779144 w_k1_MSFT_s006
                                 + 0.913382 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.19178 w_k1_ADM_s006
                                 + 0.971271 w_k1_AXP_s006
                                 + 1.17847 w_k1_DIS_s006
                                 + 1.14354 w_k1_FDX_s006
                                 + 0.924626 w_k1_IBM_s006
                                 + 1.1598 w_k1_INTC_s006 + 1.04058 w_k1_KO_s006
                                 + 0.758744 w_k1_MAT_s006
                                 + 1.14387 w_k1_MSFT_s006
                                 + 1.07299 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.01211 w_k1_ADM_s006
                                 + 0.758961 w_k1_AXP_s006
                                 + 0.801332 w_k1_DIS_s006
                                 + 0.820919 w_k1_FDX_s006
                                 + 1.09642 w_k1_IBM_s006
                                 + 0.80571 w_k1_INTC_s006
                                 + 0.908325 w_k1_KO_s006 + 1.2045 w_k1_MAT_s006
                                 + 0.839871 w_k1_MSFT_s006
                                 + 0.950368 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.60822 w_k1_ADM_s006
                                 + 0.953066 w_k1_AXP_s006
                                 + 1.43347 w_k1_DIS_s006 + 1.1859 w_k1_FDX_s006
                                 + 1.43426 w_k1_IBM_s006
                                 + 1.06428 w_k1_INTC_s006
                                 + 1.48633 w_k1_KO_s006 + 1.31521 w_k1_MAT_s006
                                 + 1.20047 w_k1_MSFT_s006
                                 + 1.29145 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.28858 w_k1_ADM_s006
                                 + 1.46215 w_k1_AXP_s006
                                 + 1.48061 w_k1_DIS_s006
                                 + 1.38463 w_k1_FDX_s006
                                 + 1.10526 w_k1_IBM_s006
                                 + 1.22778 w_k1_INTC_s006
                                 + 1.31284 w_k1_KO_s006 + 1.59706 w_k1_MAT_s006
                                 + 1.29205 w_k1_MSFT_s006
                                 + 1.34423 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.17559 w_k1_ADM_s006
                                 + 1.02395 w_k1_AXP_s006
                                 + 1.16237 w_k1_DIS_s006
                                 + 1.02729 w_k1_FDX_s006
                                 + 1.32726 w_k1_IBM_s006
                                 + 1.00497 w_k1_INTC_s006
                                 + 1.14105 w_k1_KO_s006 + 1.25974 w_k1_MAT_s006
                                 + 1.05974 w_k1_MSFT_s006
                                 + 1.07844 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.01233 w_k1_ADM_s006
                                 + 0.977075 w_k1_AXP_s006
                                 + 0.904695 w_k1_DIS_s006
                                 + 1.00656 w_k1_FDX_s006
                                 + 0.798893 w_k1_IBM_s006
                                 + 0.89827 w_k1_INTC_s006
                                 + 0.915097 w_k1_KO_s006 + 1.079 w_k1_MAT_s006
                                 + 1.09811 w_k1_MSFT_s006
                                 + 0.989054 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.902488 w_k1_ADM_s007
                                 + 0.627076 w_k1_AXP_s007
                                 + 1.0303 w_k1_DIS_s007
                                 + 0.870393 w_k1_FDX_s007
                                 + 0.832767 w_k1_IBM_s007
                                 + 0.966119 w_k1_INTC_s007
                                 + 1.13829 w_k1_KO_s007
                                 + 0.598909 w_k1_MAT_s007
                                 + 0.962 w_k1_MSFT_s007 + 0.921756 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.21786 w_k1_ADM_s007
                                 + 1.25387 w_k1_AXP_s007
                                 + 1.61569 w_k1_DIS_s007
                                 + 1.45977 w_k1_FDX_s007
                                 + 0.954732 w_k1_IBM_s007
                                 + 1.41534 w_k1_INTC_s007
                                 + 1.27431 w_k1_KO_s007
                                 + 0.721529 w_k1_MAT_s007
                                 + 1.18695 w_k1_MSFT_s007
                                 + 1.17015 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.877327 w_k1_ADM_s007
                                 + 1.10936 w_k1_AXP_s007
                                 + 0.978427 w_k1_DIS_s007
                                 + 1.02916 w_k1_FDX_s007
                                 + 0.93336 w_k1_IBM_s007
                                 + 0.97399 w_k1_INTC_s007
                                 + 1.05169 w_k1_KO_s007
                                 + 0.975221 w_k1_MAT_s007
                                 + 0.932488 w_k1_MSFT_s007
                                 + 1.07877 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.1929 w_k1_ADM_s007
                                 + 1.49175 w_k1_AXP_s007
                                 + 1.38773 w_k1_DIS_s007
                                 + 1.22654 w_k1_FDX_s007
                                 + 1.33538 w_k1_IBM_s007
                                 + 1.16373 w_k1_INTC_s007
                                 + 1.40539 w_k1_KO_s007 + 1.35171 w_k1_MAT_s007
                                 + 0.931831 w_k1_MSFT_s007
                                 + 1.16965 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.857129 w_k1_ADM_s007
                                 + 0.795313 w_k1_AXP_s007
                                 + 0.951309 w_k1_DIS_s007
                                 + 0.910259 w_k1_FDX_s007
                                 + 1.18734 w_k1_IBM_s007
                                 + 1.09572 w_k1_INTC_s007
                                 + 1.19213 w_k1_KO_s007 + 0.81739 w_k1_MAT_s007
                                 + 0.917423 w_k1_MSFT_s007
                                 + 1.12163 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.863524 w_k1_ADM_s007
                                 + 0.628602 w_k1_AXP_s007
                                 + 0.616604 w_k1_DIS_s007
                                 + 0.740695 w_k1_FDX_s007
                                 + 0.805687 w_k1_IBM_s007
                                 + 0.959928 w_k1_INTC_s007
                                 + 0.867783 w_k1_KO_s007
                                 + 0.571867 w_k1_MAT_s007
                                 + 0.710897 w_k1_MSFT_s007
                                 + 0.804532 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.09314 w_k1_ADM_s007
                                 + 0.781738 w_k1_AXP_s007
                                 + 0.812829 w_k1_DIS_s007
                                 + 1.07356 w_k1_FDX_s007
                                 + 0.722359 w_k1_IBM_s007
                                 + 0.891047 w_k1_INTC_s007
                                 + 0.807872 w_k1_KO_s007
                                 + 0.692881 w_k1_MAT_s007
                                 + 0.82732 w_k1_MSFT_s007
                                 + 0.922599 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.846273 w_k1_ADM_s007
                                 + 1.06287 w_k1_AXP_s007
                                 + 1.17031 w_k1_DIS_s007
                                 + 1.02115 w_k1_FDX_s007
                                 + 1.41154 w_k1_IBM_s007
                                 + 1.09805 w_k1_INTC_s007
                                 + 1.34072 w_k1_KO_s007 + 1.62942 w_k1_MAT_s007
                                 + 0.871382 w_k1_MSFT_s007
                                 + 1.13528 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.06296 w_k1_ADM_s010
                                 + 1.11517 w_k1_AXP_s010
                                 + 0.90035 w_k1_DIS_s010 + 1.1074 w_k1_FDX_s010
                                 + 0.586763 w_k1_IBM_s010
                                 + 1.00533 w_k1_INTC_s010
                                 + 0.806154 w_k1_KO_s010
                                 + 0.580239 w_k1_MAT_s010
                                 + 0.953459 w_k1_MSFT_s010
                                 + 0.899918 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.680853 w_k1_ADM_s010
                                 + 0.909061 w_k1_AXP_s010
                                 + 0.818979 w_k1_DIS_s010
                                 + 0.872523 w_k1_FDX_s010
                                 + 0.95786 w_k1_IBM_s010
                                 + 1.10251 w_k1_INTC_s010
                                 + 1.00402 w_k1_KO_s010
                                 + 0.982802 w_k1_MAT_s010
                                 + 0.860812 w_k1_MSFT_s010
                                 + 0.867568 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.680523 w_k1_ADM_s010
                                 + 0.657174 w_k1_AXP_s010
                                 + 0.531335 w_k1_DIS_s010
                                 + 0.733898 w_k1_FDX_s010
                                 + 0.781696 w_k1_IBM_s010
                                 + 0.827991 w_k1_INTC_s010
                                 + 0.707768 w_k1_KO_s010
                                 + 0.748411 w_k1_MAT_s010
                                 + 0.703368 w_k1_MSFT_s010
                                 + 0.71104 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.854207 w_k1_ADM_s010
                                 + 1.19878 w_k1_AXP_s010
                                 + 0.949047 w_k1_DIS_s010
                                 + 0.918125 w_k1_FDX_s010
                                 + 1.45877 w_k1_IBM_s010
                                 + 1.04307 w_k1_INTC_s010
                                 + 1.22293 w_k1_KO_s010 + 1.66029 w_k1_MAT_s010
                                 + 0.836494 w_k1_MSFT_s010
                                 + 1.04784 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.17672 w_k1_ADM_s010
                                 + 1.15118 w_k1_AXP_s010
                                 + 1.46467 w_k1_DIS_s010
                                 + 1.41485 w_k1_FDX_s010
                                 + 0.638095 w_k1_IBM_s010
                                 + 1.27729 w_k1_INTC_s010
                                 + 1.11146 w_k1_KO_s010
                                 + 0.562437 w_k1_MAT_s010
                                 + 1.37026 w_k1_MSFT_s010
                                 + 1.10845 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.908541 w_k1_ADM_s010
                                 + 1.32155 w_k1_AXP_s010
                                 + 1.59526 w_k1_DIS_s010
                                 + 1.28223 w_k1_FDX_s010
                                 + 1.02328 w_k1_IBM_s010
                                 + 1.32975 w_k1_INTC_s010
                                 + 1.32094 w_k1_KO_s010
                                 + 0.970712 w_k1_MAT_s010
                                 + 1.24025 w_k1_MSFT_s010 + 1.1707 w_k1_PG_s010
                                 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.24429 w_k1_ADM_s010
                                 + 1.47157 w_k1_AXP_s010
                                 + 1.75632 w_k1_DIS_s010
                                 + 1.25839 w_k1_FDX_s010
                                 + 1.83119 w_k1_IBM_s010
                                 + 1.29576 w_k1_INTC_s010
                                 + 1.66344 w_k1_KO_s010 + 1.76564 w_k1_MAT_s010
                                 + 1.29335 w_k1_MSFT_s010
                                 + 1.11927 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.837036 w_k1_ADM_s010
                                 + 0.958813 w_k1_AXP_s010
                                 + 1.08454 w_k1_DIS_s010
                                 + 1.02637 w_k1_FDX_s010 + 1.1033 w_k1_IBM_s010
                                 + 1.24742 w_k1_INTC_s010
                                 + 1.09652 w_k1_KO_s010
                                 + 0.894282 w_k1_MAT_s010
                                 + 1.07206 w_k1_MSFT_s010
                                 + 1.01092 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.864828 w_k1_ADM_s011
                                 + 0.668852 w_k1_AXP_s011
                                 + 1.09834 w_k1_DIS_s011
                                 + 0.921099 w_k1_FDX_s011
                                 + 0.958681 w_k1_IBM_s011
                                 + 1.09001 w_k1_INTC_s011
                                 + 1.02088 w_k1_KO_s011 + 1.01603 w_k1_MAT_s011
                                 + 1.07469 w_k1_MSFT_s011
                                 + 0.89384 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.20048 w_k1_ADM_s011
                                 + 1.09028 w_k1_AXP_s011
                                 + 0.981887 w_k1_DIS_s011
                                 + 0.964405 w_k1_FDX_s011
                                 + 1.08648 w_k1_IBM_s011
                                 + 1.04394 w_k1_INTC_s011 + 1.094 w_k1_KO_s011
                                 + 1.28377 w_k1_MAT_s011
                                 + 0.981516 w_k1_MSFT_s011
                                 + 0.89294 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.32044 w_k1_ADM_s011
                                 + 1.59352 w_k1_AXP_s011
                                 + 1.72021 w_k1_DIS_s011
                                 + 1.43496 w_k1_FDX_s011
                                 + 0.947836 w_k1_IBM_s011
                                 + 1.65348 w_k1_INTC_s011
                                 + 1.20694 w_k1_KO_s011 + 1.02529 w_k1_MAT_s011
                                 + 1.36699 w_k1_MSFT_s011
                                 + 1.16204 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.20948 w_k1_ADM_s011
                                 + 1.31603 w_k1_AXP_s011
                                 + 1.18137 w_k1_DIS_s011
                                 + 1.05306 w_k1_FDX_s011 + 1.5728 w_k1_IBM_s011
                                 + 1.02167 w_k1_INTC_s011
                                 + 1.16237 w_k1_KO_s011 + 1.25871 w_k1_MAT_s011
                                 + 1.04114 w_k1_MSFT_s011
                                 + 0.991731 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.928849 w_k1_ADM_s011
                                 + 0.638389 w_k1_AXP_s011
                                 + 0.646874 w_k1_DIS_s011
                                 + 0.756387 w_k1_FDX_s011
                                 + 0.735666 w_k1_IBM_s011
                                 + 0.9762 w_k1_INTC_s011
                                 + 0.693777 w_k1_KO_s011
                                 + 0.687842 w_k1_MAT_s011
                                 + 0.839563 w_k1_MSFT_s011
                                 + 0.791304 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.27231 w_k1_ADM_s011
                                 + 1.56137 w_k1_AXP_s011
                                 + 0.894808 w_k1_DIS_s011
                                 + 1.11693 w_k1_FDX_s011
                                 + 0.889147 w_k1_IBM_s011
                                 + 1.06828 w_k1_INTC_s011
                                 + 0.965135 w_k1_KO_s011
                                 + 1.46625 w_k1_MAT_s011
                                 + 1.18847 w_k1_MSFT_s011
                                 + 1.05224 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.931136 w_k1_ADM_s011
                                 + 0.757241 w_k1_AXP_s011
                                 + 0.759817 w_k1_DIS_s011
                                 + 0.949923 w_k1_FDX_s011
                                 + 0.705861 w_k1_IBM_s011
                                 + 1.07885 w_k1_INTC_s011
                                 + 0.761541 w_k1_KO_s011
                                 + 0.537763 w_k1_MAT_s011
                                 + 0.992354 w_k1_MSFT_s011
                                 + 0.783918 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.405 w_k1_ADM_s011
                                 + 1.04518 w_k1_AXP_s011
                                 + 1.14101 w_k1_DIS_s011
                                 + 1.06952 w_k1_FDX_s011
                                 + 0.818084 w_k1_IBM_s011
                                 + 1.05943 w_k1_INTC_s011
                                 + 1.03877 w_k1_KO_s011
                                 + 0.920738 w_k1_MAT_s011
                                 + 1.15871 w_k1_MSFT_s011
                                 + 0.980865 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.31182 w_k1_ADM_s012
                                 + 1.30041 w_k1_AXP_s012
                                 + 1.58779 w_k1_DIS_s012
                                 + 1.28656 w_k1_FDX_s012
                                 + 1.38514 w_k1_IBM_s012
                                 + 1.28333 w_k1_INTC_s012
                                 + 1.32152 w_k1_KO_s012 + 1.00063 w_k1_MAT_s012
                                 + 1.27948 w_k1_MSFT_s012
                                 + 1.27182 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.870349 w_k1_ADM_s012
                                 + 1.02164 w_k1_AXP_s012
                                 + 1.02841 w_k1_DIS_s012
                                 + 0.954187 w_k1_FDX_s012
                                 + 1.55636 w_k1_IBM_s012
                                 + 0.989903 w_k1_INTC_s012
                                 + 1.09587 w_k1_KO_s012 + 1.73609 w_k1_MAT_s012
                                 + 0.945687 w_k1_MSFT_s012
                                 + 1.15541 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.04645 w_k1_ADM_s012
                                 + 0.970752 w_k1_AXP_s012
                                 + 0.986768 w_k1_DIS_s012
                                 + 0.976798 w_k1_FDX_s012
                                 + 1.08776 w_k1_IBM_s012
                                 + 0.82638 w_k1_INTC_s012
                                 + 0.877658 w_k1_KO_s012
                                 + 0.909662 w_k1_MAT_s012
                                 + 0.884347 w_k1_MSFT_s012
                                 + 1.03076 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.01357 w_k1_ADM_s012
                                 + 0.997117 w_k1_AXP_s012
                                 + 0.776997 w_k1_DIS_s012
                                 + 0.996928 w_k1_FDX_s012
                                 + 0.658372 w_k1_IBM_s012
                                 + 0.950199 w_k1_INTC_s012
                                 + 0.733919 w_k1_KO_s012
                                 + 0.570134 w_k1_MAT_s012
                                 + 0.816558 w_k1_MSFT_s012
                                 + 0.871742 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.81029 w_k1_ADM_s012
                                 + 0.819583 w_k1_AXP_s012
                                 + 0.707502 w_k1_DIS_s012
                                 + 0.780101 w_k1_FDX_s012
                                 + 0.99233 w_k1_IBM_s012
                                 + 0.668168 w_k1_INTC_s012
                                 + 0.871745 w_k1_KO_s012
                                 + 1.00278 w_k1_MAT_s012
                                 + 0.741837 w_k1_MSFT_s012
                                 + 0.877349 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.10386 w_k1_ADM_s012
                                 + 1.30379 w_k1_AXP_s012
                                 + 1.39005 w_k1_DIS_s012
                                 + 1.25279 w_k1_FDX_s012
                                 + 1.12989 w_k1_IBM_s012
                                 + 1.34427 w_k1_INTC_s012
                                 + 1.22929 w_k1_KO_s012
                                 + 0.770953 w_k1_MAT_s012
                                 + 1.04739 w_k1_MSFT_s012
                                 + 1.15654 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.06617 w_k1_ADM_s012
                                 + 1.08361 w_k1_AXP_s012
                                 + 1.05341 w_k1_DIS_s012
                                 + 1.22044 w_k1_FDX_s012
                                 + 0.833055 w_k1_IBM_s012
                                 + 1.01015 w_k1_INTC_s012
                                 + 0.927344 w_k1_KO_s012
                                 + 0.697895 w_k1_MAT_s012
                                 + 1.08189 w_k1_MSFT_s012
                                 + 1.02768 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.760956 w_k1_ADM_s012
                                 + 0.561011 w_k1_AXP_s012
                                 + 0.578368 w_k1_DIS_s012
                                 + 0.797681 w_k1_FDX_s012
                                 + 0.629235 w_k1_IBM_s012
                                 + 0.766469 w_k1_INTC_s012
                                 + 0.683284 w_k1_KO_s012
                                 + 0.483785 w_k1_MAT_s012
                                 + 0.798968 w_k1_MSFT_s012
                                 + 0.866506 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.829639 w_k1_ADM_s013
                                 + 0.913798 w_k1_AXP_s013
                                 + 0.907114 w_k1_DIS_s013
                                 + 0.877647 w_k1_FDX_s013
                                 + 1.34595 w_k1_IBM_s013
                                 + 1.06734 w_k1_INTC_s013
                                 + 1.02515 w_k1_KO_s013
                                 + 0.938944 w_k1_MAT_s013
                                 + 0.912361 w_k1_MSFT_s013
                                 + 0.988116 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.975872 w_k1_ADM_s013
                                 + 1.24708 w_k1_AXP_s013
                                 + 1.18546 w_k1_DIS_s013
                                 + 1.18476 w_k1_FDX_s013
                                 + 1.13632 w_k1_IBM_s013
                                 + 0.968452 w_k1_INTC_s013
                                 + 1.13776 w_k1_KO_s013 + 1.06604 w_k1_MAT_s013
                                 + 0.912335 w_k1_MSFT_s013
                                 + 1.09564 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.844445 w_k1_ADM_s013
                                 + 0.936591 w_k1_AXP_s013
                                 + 0.815965 w_k1_DIS_s013
                                 + 0.837173 w_k1_FDX_s013
                                 + 0.950799 w_k1_IBM_s013
                                 + 0.906076 w_k1_INTC_s013
                                 + 0.931445 w_k1_KO_s013
                                 + 0.730309 w_k1_MAT_s013
                                 + 0.896304 w_k1_MSFT_s013
                                 + 0.982889 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.63205 w_k1_ADM_s013
                                 + 0.539846 w_k1_AXP_s013
                                 + 0.64989 w_k1_DIS_s013
                                 + 0.68169 w_k1_FDX_s013
                                 + 0.728589 w_k1_IBM_s013
                                 + 0.665452 w_k1_INTC_s013
                                 + 0.825452 w_k1_KO_s013
                                 + 0.567418 w_k1_MAT_s013
                                 + 0.689724 w_k1_MSFT_s013
                                 + 0.761803 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.852703 w_k1_ADM_s013
                                 + 1.39127 w_k1_AXP_s013
                                 + 1.27514 w_k1_DIS_s013 + 1.0589 w_k1_FDX_s013
                                 + 1.32134 w_k1_IBM_s013
                                 + 0.918184 w_k1_INTC_s013
                                 + 1.16569 w_k1_KO_s013 + 1.01571 w_k1_MAT_s013
                                 + 0.955145 w_k1_MSFT_s013
                                 + 1.14031 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.89675 w_k1_ADM_s013
                                 + 0.799526 w_k1_AXP_s013
                                 + 0.807421 w_k1_DIS_s013
                                 + 0.930273 w_k1_FDX_s013
                                 + 0.645742 w_k1_IBM_s013
                                 + 1.02338 w_k1_INTC_s013
                                 + 0.798195 w_k1_KO_s013
                                 + 0.403227 w_k1_MAT_s013
                                 + 0.873699 w_k1_MSFT_s013
                                 + 0.877031 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.98424 w_k1_ADM_s013
                                 + 1.37325 w_k1_AXP_s013
                                 + 1.66063 w_k1_DIS_s013
                                 + 1.32043 w_k1_FDX_s013
                                 + 1.76315 w_k1_IBM_s013
                                 + 1.14516 w_k1_INTC_s013
                                 + 1.60819 w_k1_KO_s013 + 1.66451 w_k1_MAT_s013
                                 + 1.28899 w_k1_MSFT_s013
                                 + 1.35744 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.990484 w_k1_ADM_s013
                                 + 1.194 w_k1_AXP_s013 + 1.22163 w_k1_DIS_s013
                                 + 1.31924 w_k1_FDX_s013
                                 + 0.809315 w_k1_IBM_s013
                                 + 1.13092 w_k1_INTC_s013
                                 + 0.991667 w_k1_KO_s013
                                 + 0.676309 w_k1_MAT_s013
                                 + 1.13018 w_k1_MSFT_s013
                                 + 1.12373 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.719873 w_k1_ADM_s014
                                 + 1.13989 w_k1_AXP_s014
                                 + 0.924454 w_k1_DIS_s014
                                 + 0.90945 w_k1_FDX_s014
                                 + 1.11171 w_k1_IBM_s014
                                 + 0.918085 w_k1_INTC_s014
                                 + 0.833489 w_k1_KO_s014
                                 + 1.00782 w_k1_MAT_s014
                                 + 0.840179 w_k1_MSFT_s014
                                 + 0.89816 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.943618 w_k1_ADM_s014
                                 + 1.06191 w_k1_AXP_s014
                                 + 1.08868 w_k1_DIS_s014
                                 + 1.04369 w_k1_FDX_s014
                                 + 0.802481 w_k1_IBM_s014
                                 + 1.04215 w_k1_INTC_s014
                                 + 0.811581 w_k1_KO_s014
                                 + 0.606217 w_k1_MAT_s014
                                 + 1.00604 w_k1_MSFT_s014
                                 + 0.927544 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.04293 w_k1_ADM_s014
                                 + 1.21214 w_k1_AXP_s014
                                 + 1.37364 w_k1_DIS_s014 + 1.1205 w_k1_FDX_s014
                                 + 1.09358 w_k1_IBM_s014
                                 + 1.01604 w_k1_INTC_s014
                                 + 1.05103 w_k1_KO_s014
                                 + 0.861141 w_k1_MAT_s014
                                 + 1.1891 w_k1_MSFT_s014 + 1.03239 w_k1_PG_s014
                                 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.548064 w_k1_ADM_s014
                                 + 0.662808 w_k1_AXP_s014
                                 + 0.698235 w_k1_DIS_s014
                                 + 0.6937 w_k1_FDX_s014
                                 + 0.952099 w_k1_IBM_s014
                                 + 0.77172 w_k1_INTC_s014
                                 + 0.739035 w_k1_KO_s014
                                 + 0.706916 w_k1_MAT_s014
                                 + 0.728365 w_k1_MSFT_s014
                                 + 0.770735 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.938126 w_k1_ADM_s014
                                 + 1.33208 w_k1_AXP_s014
                                 + 1.26907 w_k1_DIS_s014
                                 + 1.09832 w_k1_FDX_s014
                                 + 1.49615 w_k1_IBM_s014
                                 + 1.10277 w_k1_INTC_s014
                                 + 1.22561 w_k1_KO_s014 + 1.83418 w_k1_MAT_s014
                                 + 1.04356 w_k1_MSFT_s014
                                 + 1.21163 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.821631 w_k1_ADM_s014
                                 + 1.51706 w_k1_AXP_s014
                                 + 1.30473 w_k1_DIS_s014
                                 + 1.21094 w_k1_FDX_s014
                                 + 1.13367 w_k1_IBM_s014
                                 + 1.18014 w_k1_INTC_s014
                                 + 1.03041 w_k1_KO_s014 + 1.68738 w_k1_MAT_s014
                                 + 1.02935 w_k1_MSFT_s014 + 1.0378 w_k1_PG_s014
                                 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.901995 w_k1_ADM_s014
                                 + 0.837506 w_k1_AXP_s014
                                 + 0.780158 w_k1_DIS_s014
                                 + 0.825928 w_k1_FDX_s014
                                 + 0.618917 w_k1_IBM_s014
                                 + 0.734347 w_k1_INTC_s014
                                 + 0.61299 w_k1_KO_s014
                                 + 0.401298 w_k1_MAT_s014
                                 + 0.730482 w_k1_MSFT_s014
                                 + 0.769451 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.925621 w_k1_ADM_s014
                                 + 1.7061 w_k1_AXP_s014 + 1.71814 w_k1_DIS_s014
                                 + 1.29423 w_k1_FDX_s014
                                 + 1.38108 w_k1_IBM_s014
                                 + 1.08369 w_k1_INTC_s014
                                 + 1.27845 w_k1_KO_s014 + 1.57401 w_k1_MAT_s014
                                 + 1.21049 w_k1_MSFT_s014
                                 + 1.23773 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.864157 w_k1_ADM_s015
                                 + 1.04451 w_k1_AXP_s015
                                 + 0.968973 w_k1_DIS_s015
                                 + 0.809912 w_k1_FDX_s015
                                 + 1.57619 w_k1_IBM_s015
                                 + 0.700105 w_k1_INTC_s015
                                 + 1.14693 w_k1_KO_s015 + 1.40864 w_k1_MAT_s015
                                 + 0.916238 w_k1_MSFT_s015
                                 + 1.00001 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.854705 w_k1_ADM_s015
                                 + 0.790496 w_k1_AXP_s015
                                 + 0.791239 w_k1_DIS_s015
                                 + 0.809958 w_k1_FDX_s015
                                 + 1.08499 w_k1_IBM_s015
                                 + 0.699577 w_k1_INTC_s015
                                 + 0.909688 w_k1_KO_s015
                                 + 1.13564 w_k1_MAT_s015
                                 + 0.870474 w_k1_MSFT_s015
                                 + 1.02664 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.01394 w_k1_ADM_s015
                                 + 1.47652 w_k1_AXP_s015
                                 + 1.27727 w_k1_DIS_s015
                                 + 1.12407 w_k1_FDX_s015
                                 + 1.51788 w_k1_IBM_s015
                                 + 1.25209 w_k1_INTC_s015
                                 + 1.40324 w_k1_KO_s015 + 1.56053 w_k1_MAT_s015
                                 + 1.00832 w_k1_MSFT_s015
                                 + 1.33781 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.09895 w_k1_ADM_s015
                                 + 1.21793 w_k1_AXP_s015
                                 + 1.08037 w_k1_DIS_s015
                                 + 1.11399 w_k1_FDX_s015
                                 + 1.09204 w_k1_IBM_s015
                                 + 0.92341 w_k1_INTC_s015
                                 + 1.02615 w_k1_KO_s015
                                 + 0.963136 w_k1_MAT_s015
                                 + 0.970246 w_k1_MSFT_s015
                                 + 1.13496 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.870431 w_k1_ADM_s015
                                 + 0.830858 w_k1_AXP_s015
                                 + 0.780998 w_k1_DIS_s015
                                 + 0.887472 w_k1_FDX_s015
                                 + 0.715846 w_k1_IBM_s015
                                 + 0.874953 w_k1_INTC_s015
                                 + 0.808181 w_k1_KO_s015
                                 + 0.734937 w_k1_MAT_s015
                                 + 0.898427 w_k1_MSFT_s015
                                 + 0.924415 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.787988 w_k1_ADM_s015
                                 + 1.00372 w_k1_AXP_s015
                                 + 1.14703 w_k1_DIS_s015
                                 + 1.03984 w_k1_FDX_s015
                                 + 0.926399 w_k1_IBM_s015
                                 + 1.03748 w_k1_INTC_s015
                                 + 0.954798 w_k1_KO_s015
                                 + 0.779353 w_k1_MAT_s015
                                 + 0.972187 w_k1_MSFT_s015
                                 + 1.05282 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.724174 w_k1_ADM_s015
                                 + 0.535147 w_k1_AXP_s015
                                 + 0.533109 w_k1_DIS_s015
                                 + 0.599893 w_k1_FDX_s015
                                 + 0.941086 w_k1_IBM_s015
                                 + 0.634563 w_k1_INTC_s015
                                 + 0.780561 w_k1_KO_s015
                                 + 0.962044 w_k1_MAT_s015
                                 + 0.662947 w_k1_MSFT_s015
                                 + 0.781055 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.16328 w_k1_ADM_s015
                                 + 1.35215 w_k1_AXP_s015
                                 + 1.55734 w_k1_DIS_s015 + 1.4434 w_k1_FDX_s015
                                 + 1.08086 w_k1_IBM_s015
                                 + 1.31687 w_k1_INTC_s015
                                 + 1.15985 w_k1_KO_s015 + 1.14131 w_k1_MAT_s015
                                 + 1.37713 w_k1_MSFT_s015
                                 + 1.22525 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.803629 w_k1_ADM_s016
                                 + 0.963418 w_k1_AXP_s016
                                 + 1.09668 w_k1_DIS_s016
                                 + 0.947437 w_k1_FDX_s016
                                 + 1.20998 w_k1_IBM_s016
                                 + 1.04552 w_k1_INTC_s016
                                 + 0.974081 w_k1_KO_s016
                                 + 0.890147 w_k1_MAT_s016
                                 + 0.987777 w_k1_MSFT_s016
                                 + 0.930271 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.36033 w_k1_ADM_s016
                                 + 1.35274 w_k1_AXP_s016
                                 + 1.73032 w_k1_DIS_s016
                                 + 1.21227 w_k1_FDX_s016
                                 + 1.73142 w_k1_IBM_s016
                                 + 1.14496 w_k1_INTC_s016
                                 + 1.52384 w_k1_KO_s016 + 1.38363 w_k1_MAT_s016
                                 + 1.24313 w_k1_MSFT_s016
                                 + 1.15059 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.918726 w_k1_ADM_s016
                                 + 0.924657 w_k1_AXP_s016
                                 + 1.29025 w_k1_DIS_s016
                                 + 1.10309 w_k1_FDX_s016
                                 + 0.815279 w_k1_IBM_s016
                                 + 1.0969 w_k1_INTC_s016
                                 + 0.913243 w_k1_KO_s016
                                 + 0.624844 w_k1_MAT_s016
                                 + 1.10063 w_k1_MSFT_s016
                                 + 0.909972 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.00069 w_k1_ADM_s016
                                 + 0.783611 w_k1_AXP_s016
                                 + 0.768893 w_k1_DIS_s016
                                 + 0.848683 w_k1_FDX_s016
                                 + 1.04833 w_k1_IBM_s016
                                 + 0.877227 w_k1_INTC_s016
                                 + 0.856054 w_k1_KO_s016
                                 + 1.14054 w_k1_MAT_s016
                                 + 0.882853 w_k1_MSFT_s016
                                 + 0.939187 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.07347 w_k1_ADM_s016
                                 + 1.35932 w_k1_AXP_s016
                                 + 1.34795 w_k1_DIS_s016
                                 + 1.24705 w_k1_FDX_s016
                                 + 0.925404 w_k1_IBM_s016
                                 + 1.14319 w_k1_INTC_s016 + 1.0746 w_k1_KO_s016
                                 + 0.975605 w_k1_MAT_s016
                                 + 1.17637 w_k1_MSFT_s016 + 0.9972 w_k1_PG_s016
                                 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.902357 w_k1_ADM_s016
                                 + 0.70144 w_k1_AXP_s016
                                 + 0.802538 w_k1_DIS_s016
                                 + 0.811937 w_k1_FDX_s016
                                 + 0.936577 w_k1_IBM_s016
                                 + 0.765104 w_k1_INTC_s016
                                 + 0.832698 w_k1_KO_s016
                                 + 0.763639 w_k1_MAT_s016
                                 + 1.03241 w_k1_MSFT_s016
                                 + 0.792452 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.825695 w_k1_ADM_s016
                                 + 1.12138 w_k1_AXP_s016
                                 + 1.14402 w_k1_DIS_s016
                                 + 0.940528 w_k1_FDX_s016
                                 + 1.4347 w_k1_IBM_s016 + 1.04 w_k1_INTC_s016
                                 + 1.16341 w_k1_KO_s016 + 1.15249 w_k1_MAT_s016
                                 + 1.06248 w_k1_MSFT_s016
                                 + 0.924363 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.927214 w_k1_ADM_s016
                                 + 0.756927 w_k1_AXP_s016
                                 + 0.803902 w_k1_DIS_s016
                                 + 0.882739 w_k1_FDX_s016
                                 + 0.701465 w_k1_IBM_s016
                                 + 0.975338 w_k1_INTC_s016
                                 + 0.808677 w_k1_KO_s016
                                 + 0.618524 w_k1_MAT_s016
                                 + 0.96527 w_k1_MSFT_s016
                                 + 0.848656 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.846859 w_k1_ADM_s017
                                 + 0.765006 w_k1_AXP_s017
                                 + 0.799243 w_k1_DIS_s017
                                 + 0.896701 w_k1_FDX_s017
                                 + 0.790201 w_k1_IBM_s017
                                 + 0.986804 w_k1_INTC_s017
                                 + 0.946231 w_k1_KO_s017
                                 + 0.980065 w_k1_MAT_s017
                                 + 0.931666 w_k1_MSFT_s017
                                 + 0.910752 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.873897 w_k1_ADM_s017
                                 + 0.836089 w_k1_AXP_s017
                                 + 0.718507 w_k1_DIS_s017
                                 + 0.817002 w_k1_FDX_s017
                                 + 1.05813 w_k1_IBM_s017
                                 + 0.835623 w_k1_INTC_s017
                                 + 0.841853 w_k1_KO_s017
                                 + 0.986977 w_k1_MAT_s017
                                 + 0.72691 w_k1_MSFT_s017
                                 + 0.848938 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.27135 w_k1_ADM_s017
                                 + 1.56836 w_k1_AXP_s017
                                 + 2.03167 w_k1_DIS_s017
                                 + 1.48781 w_k1_FDX_s017 + 1.7549 w_k1_IBM_s017
                                 + 1.32342 w_k1_INTC_s017
                                 + 1.62821 w_k1_KO_s017 + 2.03379 w_k1_MAT_s017
                                 + 1.46141 w_k1_MSFT_s017
                                 + 1.38631 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.34105 w_k1_ADM_s017
                                 + 1.22661 w_k1_AXP_s017
                                 + 1.10883 w_k1_DIS_s017
                                 + 1.12647 w_k1_FDX_s017
                                 + 1.37994 w_k1_IBM_s017
                                 + 1.22367 w_k1_INTC_s017
                                 + 1.13872 w_k1_KO_s017 + 1.57403 w_k1_MAT_s017
                                 + 1.06895 w_k1_MSFT_s017
                                 + 1.19688 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.36103 w_k1_ADM_s017
                                 + 1.13132 w_k1_AXP_s017
                                 + 1.72128 w_k1_DIS_s017
                                 + 1.42023 w_k1_FDX_s017
                                 + 1.02954 w_k1_IBM_s017
                                 + 1.34532 w_k1_INTC_s017 + 1.3039 w_k1_KO_s017
                                 + 1.12703 w_k1_MAT_s017
                                 + 1.34539 w_k1_MSFT_s017
                                 + 1.18117 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.02379 w_k1_ADM_s017
                                 + 1.30954 w_k1_AXP_s017
                                 + 0.981644 w_k1_DIS_s017
                                 + 1.13151 w_k1_FDX_s017
                                 + 0.88882 w_k1_IBM_s017
                                 + 1.13574 w_k1_INTC_s017
                                 + 0.97322 w_k1_KO_s017 + 1.39896 w_k1_MAT_s017
                                 + 0.950534 w_k1_MSFT_s017
                                 + 0.995291 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.18335 w_k1_ADM_s017
                                 + 0.905954 w_k1_AXP_s017
                                 + 1.15161 w_k1_DIS_s017
                                 + 1.18854 w_k1_FDX_s017
                                 + 0.87381 w_k1_IBM_s017
                                 + 0.927996 w_k1_INTC_s017
                                 + 0.986495 w_k1_KO_s017 + 1.1581 w_k1_MAT_s017
                                 + 1.06038 w_k1_MSFT_s017
                                 + 1.01195 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.0779 w_k1_ADM_s017
                                 + 0.817269 w_k1_AXP_s017
                                 + 1.16485 w_k1_DIS_s017
                                 + 0.982915 w_k1_FDX_s017
                                 + 1.32959 w_k1_IBM_s017
                                 + 0.975197 w_k1_INTC_s017
                                 + 1.29575 w_k1_KO_s017 + 1.71679 w_k1_MAT_s017
                                 + 1.08772 w_k1_MSFT_s017
                                 + 1.16171 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.920736 w_k1_ADM_s100
                                 + 1.31581 w_k1_AXP_s100
                                 + 1.35326 w_k1_DIS_s100
                                 + 1.20149 w_k1_FDX_s100
                                 + 1.31436 w_k1_IBM_s100
                                 + 1.21577 w_k1_INTC_s100
                                 + 1.22873 w_k1_KO_s100 + 1.65217 w_k1_MAT_s100
                                 + 1.21895 w_k1_MSFT_s100
                                 + 1.23957 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.908536 w_k1_ADM_s100
                                 + 0.859879 w_k1_AXP_s100
                                 + 0.677013 w_k1_DIS_s100
                                 + 0.928526 w_k1_FDX_s100
                                 + 0.677104 w_k1_IBM_s100
                                 + 1.23442 w_k1_INTC_s100
                                 + 0.879252 w_k1_KO_s100
                                 + 0.738132 w_k1_MAT_s100
                                 + 0.986545 w_k1_MSFT_s100
                                 + 0.919934 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.987232 w_k1_ADM_s100
                                 + 0.91812 w_k1_AXP_s100
                                 + 1.03749 w_k1_DIS_s100
                                 + 1.00328 w_k1_FDX_s100
                                 + 0.876982 w_k1_IBM_s100
                                 + 1.08889 w_k1_INTC_s100
                                 + 1.04619 w_k1_KO_s100
                                 + 0.933569 w_k1_MAT_s100
                                 + 1.10048 w_k1_MSFT_s100
                                 + 1.01749 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.901836 w_k1_ADM_s100
                                 + 1.05473 w_k1_AXP_s100
                                 + 0.903859 w_k1_DIS_s100
                                 + 0.936188 w_k1_FDX_s100
                                 + 1.20619 w_k1_IBM_s100
                                 + 0.994911 w_k1_INTC_s100
                                 + 1.05466 w_k1_KO_s100 + 1.60196 w_k1_MAT_s100
                                 + 0.875215 w_k1_MSFT_s100
                                 + 0.987518 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.05345 w_k1_ADM_s100
                                 + 1.16464 w_k1_AXP_s100
                                 + 1.12661 w_k1_DIS_s100
                                 + 1.16049 w_k1_FDX_s100
                                 + 1.00615 w_k1_IBM_s100
                                 + 1.44816 w_k1_INTC_s100 + 1.0348 w_k1_KO_s100
                                 + 1.14066 w_k1_MAT_s100
                                 + 1.21544 w_k1_MSFT_s100
                                 + 1.06456 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.07085 w_k1_ADM_s100
                                 + 1.67676 w_k1_AXP_s100
                                 + 1.35533 w_k1_DIS_s100
                                 + 1.40762 w_k1_FDX_s100
                                 + 1.18897 w_k1_IBM_s100
                                 + 1.41037 w_k1_INTC_s100
                                 + 1.31154 w_k1_KO_s100 + 2.03629 w_k1_MAT_s100
                                 + 1.15067 w_k1_MSFT_s100
                                 + 1.19666 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.830156 w_k1_ADM_s100
                                 + 0.961501 w_k1_AXP_s100
                                 + 0.840474 w_k1_DIS_s100
                                 + 0.94527 w_k1_FDX_s100
                                 + 1.02255 w_k1_IBM_s100
                                 + 1.36758 w_k1_INTC_s100
                                 + 0.939587 w_k1_KO_s100
                                 + 1.08075 w_k1_MAT_s100
                                 + 0.986359 w_k1_MSFT_s100
                                 + 0.961616 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.574689 w_k1_ADM_s100
                                 + 0.686837 w_k1_AXP_s100
                                 + 0.604842 w_k1_DIS_s100
                                 + 0.716807 w_k1_FDX_s100
                                 + 0.935257 w_k1_IBM_s100
                                 + 1.11658 w_k1_INTC_s100
                                 + 0.807077 w_k1_KO_s100
                                 + 0.693813 w_k1_MAT_s100
                                 + 0.764874 w_k1_MSFT_s100
                                 + 0.883774 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.12033 w_k1_ADM_s101
                                 + 1.27758 w_k1_AXP_s101
                                 + 1.38518 w_k1_DIS_s101
                                 + 1.21602 w_k1_FDX_s101
                                 + 1.00238 w_k1_IBM_s101
                                 + 1.33208 w_k1_INTC_s101
                                 + 0.982418 w_k1_KO_s101
                                 + 0.988416 w_k1_MAT_s101
                                 + 1.48453 w_k1_MSFT_s101
                                 + 1.07485 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.736802 w_k1_ADM_s101
                                 + 0.804433 w_k1_AXP_s101
                                 + 0.788907 w_k1_DIS_s101
                                 + 0.822564 w_k1_FDX_s101
                                 + 0.900595 w_k1_IBM_s101
                                 + 1.00337 w_k1_INTC_s101
                                 + 0.821252 w_k1_KO_s101
                                 + 0.793244 w_k1_MAT_s101
                                 + 1.0238 w_k1_MSFT_s101
                                 + 0.935711 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.19001 w_k1_ADM_s101
                                 + 1.1362 w_k1_AXP_s101 + 1.34742 w_k1_DIS_s101
                                 + 1.32223 w_k1_FDX_s101
                                 + 0.768259 w_k1_IBM_s101
                                 + 1.54585 w_k1_INTC_s101
                                 + 1.01283 w_k1_KO_s101
                                 + 0.865341 w_k1_MAT_s101
                                 + 1.54563 w_k1_MSFT_s101
                                 + 1.21248 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.684734 w_k1_ADM_s101
                                 + 0.565998 w_k1_AXP_s101
                                 + 0.631793 w_k1_DIS_s101
                                 + 0.697544 w_k1_FDX_s101
                                 + 0.615311 w_k1_IBM_s101
                                 + 0.845037 w_k1_INTC_s101
                                 + 0.627964 w_k1_KO_s101
                                 + 0.609692 w_k1_MAT_s101
                                 + 0.856005 w_k1_MSFT_s101
                                 + 0.815984 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.18411 w_k1_ADM_s101
                                 + 1.42753 w_k1_AXP_s101
                                 + 1.23381 w_k1_DIS_s101
                                 + 1.16059 w_k1_FDX_s101
                                 + 1.49591 w_k1_IBM_s101
                                 + 1.31979 w_k1_INTC_s101
                                 + 1.14037 w_k1_KO_s101 + 1.4564 w_k1_MAT_s101
                                 + 1.17417 w_k1_MSFT_s101
                                 + 1.34369 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.23094 w_k1_ADM_s101
                                 + 1.00584 w_k1_AXP_s101
                                 + 0.942348 w_k1_DIS_s101
                                 + 1.03301 w_k1_FDX_s101
                                 + 0.867495 w_k1_IBM_s101
                                 + 1.0937 w_k1_INTC_s101
                                 + 0.951555 w_k1_KO_s101
                                 + 0.842959 w_k1_MAT_s101
                                 + 1.19352 w_k1_MSFT_s101
                                 + 1.02055 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.949194 w_k1_ADM_s101
                                 + 0.934705 w_k1_AXP_s101
                                 + 0.863878 w_k1_DIS_s101
                                 + 0.866746 w_k1_FDX_s101
                                 + 1.47714 w_k1_IBM_s101
                                 + 1.08311 w_k1_INTC_s101
                                 + 0.972881 w_k1_KO_s101
                                 + 1.06308 w_k1_MAT_s101
                                 + 0.993691 w_k1_MSFT_s101
                                 + 1.06419 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.56038 w_k1_ADM_s101
                                 + 1.67369 w_k1_AXP_s101
                                 + 1.80241 w_k1_DIS_s101
                                 + 1.53208 w_k1_FDX_s101
                                 + 1.26099 w_k1_IBM_s101
                                 + 1.3506 w_k1_INTC_s101 + 1.0909 w_k1_KO_s101
                                 + 1.43799 w_k1_MAT_s101
                                 + 1.37423 w_k1_MSFT_s101 + 1.3047 w_k1_PG_s101
                                 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.01889 w_k1_ADM_s102
                                 + 0.956427 w_k1_AXP_s102
                                 + 0.96007 w_k1_DIS_s102
                                 + 0.881573 w_k1_FDX_s102
                                 + 1.39629 w_k1_IBM_s102
                                 + 0.992529 w_k1_INTC_s102
                                 + 0.937924 w_k1_KO_s102
                                 + 1.22564 w_k1_MAT_s102
                                 + 0.951905 w_k1_MSFT_s102
                                 + 0.904826 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.75368 w_k1_ADM_s102
                                 + 0.728036 w_k1_AXP_s102
                                 + 0.764673 w_k1_DIS_s102
                                 + 0.851168 w_k1_FDX_s102
                                 + 0.687733 w_k1_IBM_s102
                                 + 1.00954 w_k1_INTC_s102
                                 + 0.763318 w_k1_KO_s102 + 0.6005 w_k1_MAT_s102
                                 + 0.90202 w_k1_MSFT_s102
                                 + 0.843635 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.76776 w_k1_ADM_s102
                                 + 0.608389 w_k1_AXP_s102
                                 + 0.712622 w_k1_DIS_s102
                                 + 0.699649 w_k1_FDX_s102
                                 + 0.888601 w_k1_IBM_s102
                                 + 0.680329 w_k1_INTC_s102
                                 + 0.848004 w_k1_KO_s102
                                 + 0.739161 w_k1_MAT_s102
                                 + 0.808692 w_k1_MSFT_s102
                                 + 0.745425 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.971009 w_k1_ADM_s102
                                 + 0.967721 w_k1_AXP_s102
                                 + 1.21232 w_k1_DIS_s102
                                 + 1.14055 w_k1_FDX_s102
                                 + 0.752621 w_k1_IBM_s102
                                 + 1.19726 w_k1_INTC_s102
                                 + 0.948844 w_k1_KO_s102
                                 + 0.810758 w_k1_MAT_s102
                                 + 1.11002 w_k1_MSFT_s102
                                 + 0.981086 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.24335 w_k1_ADM_s102
                                 + 0.98308 w_k1_AXP_s102
                                 + 0.98585 w_k1_DIS_s102
                                 + 1.04454 w_k1_FDX_s102
                                 + 0.886239 w_k1_IBM_s102
                                 + 0.933584 w_k1_INTC_s102
                                 + 0.803378 w_k1_KO_s102
                                 + 0.739577 w_k1_MAT_s102
                                 + 0.89883 w_k1_MSFT_s102
                                 + 0.920721 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.05645 w_k1_ADM_s102
                                 + 0.963075 w_k1_AXP_s102
                                 + 0.938348 w_k1_DIS_s102
                                 + 0.900384 w_k1_FDX_s102
                                 + 1.01744 w_k1_IBM_s102
                                 + 0.917729 w_k1_INTC_s102
                                 + 0.893628 w_k1_KO_s102
                                 + 0.935606 w_k1_MAT_s102
                                 + 0.873025 w_k1_MSFT_s102
                                 + 0.865402 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.24362 w_k1_ADM_s102
                                 + 1.24724 w_k1_AXP_s102 + 1.4994 w_k1_DIS_s102
                                 + 1.37769 w_k1_FDX_s102
                                 + 0.843095 w_k1_IBM_s102
                                 + 1.3092 w_k1_INTC_s102 + 0.98068 w_k1_KO_s102
                                 + 0.816101 w_k1_MAT_s102
                                 + 1.19095 w_k1_MSFT_s102
                                 + 1.00033 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.87252 w_k1_ADM_s102
                                 + 1.22398 w_k1_AXP_s102
                                 + 1.43075 w_k1_DIS_s102
                                 + 1.15327 w_k1_FDX_s102
                                 + 1.27039 w_k1_IBM_s102
                                 + 1.19338 w_k1_INTC_s102
                                 + 1.26382 w_k1_KO_s102 + 1.40403 w_k1_MAT_s102
                                 + 1.18364 w_k1_MSFT_s102
                                 + 1.05042 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.14224 w_k1_ADM_s103
                                 + 1.00342 w_k1_AXP_s103
                                 + 1.17515 w_k1_DIS_s103
                                 + 1.21236 w_k1_FDX_s103
                                 + 0.935803 w_k1_IBM_s103
                                 + 1.05524 w_k1_INTC_s103
                                 + 1.16891 w_k1_KO_s103 + 1.07338 w_k1_MAT_s103
                                 + 1.087 w_k1_MSFT_s103 + 1.15735 w_k1_PG_s103
                                 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.891207 w_k1_ADM_s103
                                 + 0.842013 w_k1_AXP_s103
                                 + 0.769859 w_k1_DIS_s103
                                 + 0.801045 w_k1_FDX_s103
                                 + 1.01847 w_k1_IBM_s103
                                 + 0.786037 w_k1_INTC_s103
                                 + 0.851544 w_k1_KO_s103
                                 + 1.38442 w_k1_MAT_s103
                                 + 0.825658 w_k1_MSFT_s103
                                 + 0.988043 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.999096 w_k1_ADM_s103
                                 + 1.06102 w_k1_AXP_s103
                                 + 0.847226 w_k1_DIS_s103
                                 + 0.973076 w_k1_FDX_s103
                                 + 0.922997 w_k1_IBM_s103
                                 + 0.814983 w_k1_INTC_s103
                                 + 0.831918 w_k1_KO_s103
                                 + 1.31385 w_k1_MAT_s103
                                 + 0.822788 w_k1_MSFT_s103
                                 + 1.01812 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.06861 w_k1_ADM_s103
                                 + 0.764003 w_k1_AXP_s103
                                 + 0.808376 w_k1_DIS_s103
                                 + 0.910167 w_k1_FDX_s103
                                 + 0.744562 w_k1_IBM_s103
                                 + 0.777818 w_k1_INTC_s103
                                 + 0.858298 w_k1_KO_s103
                                 + 0.709219 w_k1_MAT_s103
                                 + 0.91508 w_k1_MSFT_s103
                                 + 0.890757 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.14839 w_k1_ADM_s103
                                 + 1.28765 w_k1_AXP_s103 + 1.2652 w_k1_DIS_s103
                                 + 1.10429 w_k1_FDX_s103
                                 + 1.44647 w_k1_IBM_s103
                                 + 0.938265 w_k1_INTC_s103
                                 + 1.21541 w_k1_KO_s103 + 1.54613 w_k1_MAT_s103
                                 + 1.04406 w_k1_MSFT_s103
                                 + 1.32909 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.918145 w_k1_ADM_s103
                                 + 0.887355 w_k1_AXP_s103
                                 + 0.686995 w_k1_DIS_s103
                                 + 0.867077 w_k1_FDX_s103
                                 + 0.715852 w_k1_IBM_s103
                                 + 0.750236 w_k1_INTC_s103
                                 + 0.733395 w_k1_KO_s103
                                 + 0.72499 w_k1_MAT_s103
                                 + 0.799631 w_k1_MSFT_s103
                                 + 0.809095 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.966297 w_k1_ADM_s103
                                 + 1.15013 w_k1_AXP_s103 + 1.119 w_k1_DIS_s103
                                 + 1.02718 w_k1_FDX_s103
                                 + 1.01331 w_k1_IBM_s103
                                 + 0.919817 w_k1_INTC_s103
                                 + 1.03644 w_k1_KO_s103 + 1.45694 w_k1_MAT_s103
                                 + 1.04099 w_k1_MSFT_s103
                                 + 0.995654 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.659198 w_k1_ADM_s103
                                 + 0.575595 w_k1_AXP_s103
                                 + 0.693221 w_k1_DIS_s103
                                 + 0.793355 w_k1_FDX_s103
                                 + 0.618181 w_k1_IBM_s103
                                 + 0.809743 w_k1_INTC_s103
                                 + 0.705667 w_k1_KO_s103
                                 + 0.708363 w_k1_MAT_s103
                                 + 0.828837 w_k1_MSFT_s103
                                 + 0.782712 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.893223 w_k1_ADM_s104
                                 + 1.2745 w_k1_AXP_s104
                                 + 0.882254 w_k1_DIS_s104
                                 + 0.915651 w_k1_FDX_s104
                                 + 1.44935 w_k1_IBM_s104
                                 + 1.12569 w_k1_INTC_s104
                                 + 0.992694 w_k1_KO_s104
                                 + 1.44129 w_k1_MAT_s104
                                 + 0.889412 w_k1_MSFT_s104
                                 + 0.952151 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.941102 w_k1_ADM_s104
                                 + 0.773398 w_k1_AXP_s104
                                 + 0.659515 w_k1_DIS_s104
                                 + 0.822457 w_k1_FDX_s104
                                 + 0.727178 w_k1_IBM_s104
                                 + 0.973745 w_k1_INTC_s104
                                 + 0.764382 w_k1_KO_s104
                                 + 0.708914 w_k1_MAT_s104
                                 + 0.859737 w_k1_MSFT_s104
                                 + 0.860129 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.770847 w_k1_ADM_s104
                                 + 0.607595 w_k1_AXP_s104
                                 + 0.48503 w_k1_DIS_s104
                                 + 0.687342 w_k1_FDX_s104
                                 + 0.753159 w_k1_IBM_s104
                                 + 0.912213 w_k1_INTC_s104
                                 + 0.729934 w_k1_KO_s104
                                 + 0.678545 w_k1_MAT_s104
                                 + 0.814208 w_k1_MSFT_s104
                                 + 0.821089 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.11323 w_k1_ADM_s104
                                 + 1.09879 w_k1_AXP_s104
                                 + 1.14076 w_k1_DIS_s104
                                 + 1.23927 w_k1_FDX_s104
                                 + 0.969565 w_k1_IBM_s104
                                 + 1.18323 w_k1_INTC_s104
                                 + 1.04234 w_k1_KO_s104
                                 + 0.869796 w_k1_MAT_s104
                                 + 1.27038 w_k1_MSFT_s104
                                 + 1.20669 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.960428 w_k1_ADM_s104
                                 + 0.863336 w_k1_AXP_s104
                                 + 0.864147 w_k1_DIS_s104
                                 + 0.988018 w_k1_FDX_s104
                                 + 0.677138 w_k1_IBM_s104
                                 + 1.05081 w_k1_INTC_s104
                                 + 0.718249 w_k1_KO_s104
                                 + 0.477589 w_k1_MAT_s104
                                 + 0.960946 w_k1_MSFT_s104
                                 + 0.906984 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.07952 w_k1_ADM_s104
                                 + 1.34398 w_k1_AXP_s104
                                 + 1.30406 w_k1_DIS_s104 + 1.2107 w_k1_FDX_s104
                                 + 1.41618 w_k1_IBM_s104
                                 + 1.14053 w_k1_INTC_s104
                                 + 1.17889 w_k1_KO_s104 + 1.37708 w_k1_MAT_s104
                                 + 1.15611 w_k1_MSFT_s104
                                 + 1.23114 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.785395 w_k1_ADM_s104
                                 + 0.757857 w_k1_AXP_s104
                                 + 0.871881 w_k1_DIS_s104
                                 + 0.843509 w_k1_FDX_s104
                                 + 0.960221 w_k1_IBM_s104
                                 + 1.08098 w_k1_INTC_s104
                                 + 0.830709 w_k1_KO_s104
                                 + 0.68932 w_k1_MAT_s104
                                 + 0.929458 w_k1_MSFT_s104
                                 + 0.946359 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.944955 w_k1_ADM_s104
                                 + 0.925023 w_k1_AXP_s104
                                 + 1.01463 w_k1_DIS_s104
                                 + 1.02195 w_k1_FDX_s104
                                 + 0.941067 w_k1_IBM_s104
                                 + 1.24162 w_k1_INTC_s104
                                 + 1.01184 w_k1_KO_s104
                                 + 0.765999 w_k1_MAT_s104
                                 + 1.16444 w_k1_MSFT_s104
                                 + 0.970704 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.01259 w_k1_ADM_s105
                                 + 1.09449 w_k1_AXP_s105
                                 + 1.03073 w_k1_DIS_s105
                                 + 1.01493 w_k1_FDX_s105
                                 + 1.49163 w_k1_IBM_s105
                                 + 0.862236 w_k1_INTC_s105
                                 + 1.16373 w_k1_KO_s105 + 1.68432 w_k1_MAT_s105
                                 + 0.870718 w_k1_MSFT_s105
                                 + 1.04062 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.744331 w_k1_ADM_s105
                                 + 0.750113 w_k1_AXP_s105
                                 + 0.878021 w_k1_DIS_s105
                                 + 0.813478 w_k1_FDX_s105
                                 + 1.25704 w_k1_IBM_s105
                                 + 1.03718 w_k1_INTC_s105
                                 + 1.09642 w_k1_KO_s105
                                 + 0.945105 w_k1_MAT_s105
                                 + 0.842383 w_k1_MSFT_s105
                                 + 0.879009 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.11301 w_k1_ADM_s105
                                 + 1.29553 w_k1_AXP_s105
                                 + 1.61259 w_k1_DIS_s105
                                 + 1.25099 w_k1_FDX_s105
                                 + 1.14827 w_k1_IBM_s105
                                 + 1.16016 w_k1_INTC_s105
                                 + 1.25219 w_k1_KO_s105 + 1.11278 w_k1_MAT_s105
                                 + 1.18598 w_k1_MSFT_s105
                                 + 1.15061 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.807626 w_k1_ADM_s105
                                 + 0.819921 w_k1_AXP_s105
                                 + 0.737335 w_k1_DIS_s105
                                 + 0.867682 w_k1_FDX_s105
                                 + 0.869946 w_k1_IBM_s105
                                 + 1.00088 w_k1_INTC_s105
                                 + 0.825641 w_k1_KO_s105
                                 + 0.902834 w_k1_MAT_s105
                                 + 0.828181 w_k1_MSFT_s105
                                 + 0.889575 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.07391 w_k1_ADM_s105
                                 + 1.18608 w_k1_AXP_s105
                                 + 0.911639 w_k1_DIS_s105
                                 + 1.04925 w_k1_FDX_s105
                                 + 0.99913 w_k1_IBM_s105
                                 + 1.17471 w_k1_INTC_s105
                                 + 0.930606 w_k1_KO_s105
                                 + 1.10636 w_k1_MAT_s105
                                 + 0.981153 w_k1_MSFT_s105
                                 + 0.93038 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.731427 w_k1_ADM_s105
                                 + 0.733525 w_k1_AXP_s105
                                 + 0.610745 w_k1_DIS_s105
                                 + 0.732968 w_k1_FDX_s105
                                 + 0.654672 w_k1_IBM_s105
                                 + 0.807047 w_k1_INTC_s105
                                 + 0.683752 w_k1_KO_s105
                                 + 0.665774 w_k1_MAT_s105
                                 + 0.750187 w_k1_MSFT_s105
                                 + 0.717116 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.31894 w_k1_ADM_s105
                                 + 1.13985 w_k1_AXP_s105
                                 + 1.32295 w_k1_DIS_s105
                                 + 1.22893 w_k1_FDX_s105
                                 + 0.92832 w_k1_IBM_s105
                                 + 1.00327 w_k1_INTC_s105
                                 + 1.09699 w_k1_KO_s105 + 1.03024 w_k1_MAT_s105
                                 + 1.00524 w_k1_MSFT_s105
                                 + 1.20647 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.0943 w_k1_ADM_s105
                                 + 0.968259 w_k1_AXP_s105
                                 + 1.01939 w_k1_DIS_s105
                                 + 1.15834 w_k1_FDX_s105
                                 + 0.710445 w_k1_IBM_s105
                                 + 1.05368 w_k1_INTC_s105
                                 + 0.837359 w_k1_KO_s105
                                 + 0.734571 w_k1_MAT_s105
                                 + 1.09141 w_k1_MSFT_s105
                                 + 1.03633 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.4443 w_k1_ADM_s106
                                 + 1.44332 w_k1_AXP_s106
                                 + 1.25622 w_k1_DIS_s106 + 1.0019 w_k1_FDX_s106
                                 + 1.80269 w_k1_IBM_s106
                                 + 1.24917 w_k1_INTC_s106
                                 + 1.65971 w_k1_KO_s106 + 1.69083 w_k1_MAT_s106
                                 + 1.02327 w_k1_MSFT_s106
                                 + 1.29752 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.917803 w_k1_ADM_s106
                                 + 0.90264 w_k1_AXP_s106
                                 + 1.10199 w_k1_DIS_s106
                                 + 0.884361 w_k1_FDX_s106
                                 + 1.24447 w_k1_IBM_s106
                                 + 0.99952 w_k1_INTC_s106
                                 + 1.09629 w_k1_KO_s106
                                 + 0.990023 w_k1_MAT_s106
                                 + 0.939916 w_k1_MSFT_s106
                                 + 0.894468 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.45094 w_k1_ADM_s106
                                 + 1.45441 w_k1_AXP_s106
                                 + 1.26931 w_k1_DIS_s106
                                 + 1.19442 w_k1_FDX_s106
                                 + 1.28631 w_k1_IBM_s106
                                 + 1.18544 w_k1_INTC_s106
                                 + 1.20602 w_k1_KO_s106 + 1.33208 w_k1_MAT_s106
                                 + 1.16101 w_k1_MSFT_s106
                                 + 1.24843 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.705235 w_k1_ADM_s106
                                 + 0.673404 w_k1_AXP_s106
                                 + 0.795417 w_k1_DIS_s106
                                 + 0.802652 w_k1_FDX_s106
                                 + 0.795904 w_k1_IBM_s106
                                 + 1.00682 w_k1_INTC_s106
                                 + 0.774917 w_k1_KO_s106
                                 + 0.676636 w_k1_MAT_s106
                                 + 0.871225 w_k1_MSFT_s106
                                 + 0.906811 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.70135 w_k1_ADM_s106
                                 + 1.98594 w_k1_AXP_s106
                                 + 1.96216 w_k1_DIS_s106
                                 + 1.72426 w_k1_FDX_s106
                                 + 1.16533 w_k1_IBM_s106
                                 + 1.49004 w_k1_INTC_s106
                                 + 1.44956 w_k1_KO_s106 + 1.14886 w_k1_MAT_s106
                                 + 1.32121 w_k1_MSFT_s106
                                 + 1.40752 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.981176 w_k1_ADM_s106
                                 + 0.913309 w_k1_AXP_s106
                                 + 1.13081 w_k1_DIS_s106
                                 + 0.988657 w_k1_FDX_s106
                                 + 0.924982 w_k1_IBM_s106
                                 + 0.999252 w_k1_INTC_s106
                                 + 0.927143 w_k1_KO_s106
                                 + 0.929265 w_k1_MAT_s106
                                 + 1.02706 w_k1_MSFT_s106
                                 + 1.14391 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.14706 w_k1_ADM_s106
                                 + 1.0566 w_k1_AXP_s106
                                 + 0.840858 w_k1_DIS_s106
                                 + 0.973693 w_k1_FDX_s106
                                 + 0.85787 w_k1_IBM_s106
                                 + 0.915582 w_k1_INTC_s106
                                 + 0.782658 w_k1_KO_s106
                                 + 0.996052 w_k1_MAT_s106
                                 + 0.866464 w_k1_MSFT_s106
                                 + 1.03832 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.723825 w_k1_ADM_s106
                                 + 0.686028 w_k1_AXP_s106
                                 + 0.601642 w_k1_DIS_s106
                                 + 0.780648 w_k1_FDX_s106
                                 + 0.621503 w_k1_IBM_s106
                                 + 0.696485 w_k1_INTC_s106
                                 + 0.608897 w_k1_KO_s106
                                 + 0.545922 w_k1_MAT_s106
                                 + 0.636973 w_k1_MSFT_s106
                                 + 0.801085 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.880356 w_k1_ADM_s107
                                 + 0.853447 w_k1_AXP_s107
                                 + 0.863842 w_k1_DIS_s107
                                 + 0.964512 w_k1_FDX_s107
                                 + 0.629256 w_k1_IBM_s107
                                 + 1.23404 w_k1_INTC_s107
                                 + 0.839789 w_k1_KO_s107
                                 + 0.614927 w_k1_MAT_s107
                                 + 0.983165 w_k1_MSFT_s107
                                 + 0.890791 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.78251 w_k1_ADM_s107
                                 + 0.818281 w_k1_AXP_s107
                                 + 0.710457 w_k1_DIS_s107
                                 + 0.796887 w_k1_FDX_s107
                                 + 0.86328 w_k1_IBM_s107
                                 + 1.08747 w_k1_INTC_s107
                                 + 0.844857 w_k1_KO_s107
                                 + 0.735796 w_k1_MAT_s107
                                 + 0.791404 w_k1_MSFT_s107
                                 + 0.723916 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.919871 w_k1_ADM_s107
                                 + 0.830461 w_k1_AXP_s107
                                 + 1.21037 w_k1_DIS_s107
                                 + 1.05656 w_k1_FDX_s107
                                 + 1.03745 w_k1_IBM_s107
                                 + 1.28992 w_k1_INTC_s107
                                 + 1.27587 w_k1_KO_s107
                                 + 0.762575 w_k1_MAT_s107
                                 + 1.04042 w_k1_MSFT_s107
                                 + 1.02088 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.81985 w_k1_ADM_s107
                                 + 0.935999 w_k1_AXP_s107
                                 + 0.772324 w_k1_DIS_s107
                                 + 0.909106 w_k1_FDX_s107
                                 + 1.03407 w_k1_IBM_s107
                                 + 1.06406 w_k1_INTC_s107
                                 + 0.948079 w_k1_KO_s107
                                 + 1.00857 w_k1_MAT_s107
                                 + 0.938192 w_k1_MSFT_s107
                                 + 0.781352 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.23161 w_k1_ADM_s107
                                 + 1.42559 w_k1_AXP_s107
                                 + 1.51473 w_k1_DIS_s107
                                 + 1.38578 w_k1_FDX_s107
                                 + 1.22866 w_k1_IBM_s107
                                 + 1.74536 w_k1_INTC_s107
                                 + 1.71989 w_k1_KO_s107 + 1.86728 w_k1_MAT_s107
                                 + 1.53792 w_k1_MSFT_s107
                                 + 1.28748 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.680525 w_k1_ADM_s107
                                 + 0.615245 w_k1_AXP_s107
                                 + 0.560901 w_k1_DIS_s107
                                 + 0.809344 w_k1_FDX_s107
                                 + 0.558344 w_k1_IBM_s107
                                 + 0.753753 w_k1_INTC_s107
                                 + 0.599799 w_k1_KO_s107
                                 + 0.520722 w_k1_MAT_s107
                                 + 0.808883 w_k1_MSFT_s107
                                 + 0.731598 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.778829 w_k1_ADM_s107
                                 + 1.15294 w_k1_AXP_s107
                                 + 1.02007 w_k1_DIS_s107
                                 + 0.955606 w_k1_FDX_s107
                                 + 1.47448 w_k1_IBM_s107
                                 + 1.43037 w_k1_INTC_s107
                                 + 1.19474 w_k1_KO_s107 + 1.42338 w_k1_MAT_s107
                                 + 0.915566 w_k1_MSFT_s107
                                 + 0.880995 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.11034 w_k1_ADM_s107
                                 + 0.917566 w_k1_AXP_s107
                                 + 1.39062 w_k1_DIS_s107
                                 + 1.21492 w_k1_FDX_s107
                                 + 1.00481 w_k1_IBM_s107
                                 + 1.23418 w_k1_INTC_s107
                                 + 1.23179 w_k1_KO_s107
                                 + 0.791337 w_k1_MAT_s107
                                 + 1.2669 w_k1_MSFT_s107
                                 + 0.974726 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.596011 w_k1_ADM_s110
                                 + 0.737664 w_k1_AXP_s110
                                 + 0.636238 w_k1_DIS_s110
                                 + 0.748202 w_k1_FDX_s110
                                 + 0.794279 w_k1_IBM_s110
                                 + 0.720692 w_k1_INTC_s110
                                 + 0.860794 w_k1_KO_s110
                                 + 0.735488 w_k1_MAT_s110
                                 + 0.665705 w_k1_MSFT_s110
                                 + 0.758214 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.822413 w_k1_ADM_s110
                                 + 0.986816 w_k1_AXP_s110
                                 + 0.971678 w_k1_DIS_s110
                                 + 0.955763 w_k1_FDX_s110
                                 + 1.58402 w_k1_IBM_s110
                                 + 0.792583 w_k1_INTC_s110
                                 + 1.22447 w_k1_KO_s110 + 1.73372 w_k1_MAT_s110
                                 + 0.840515 w_k1_MSFT_s110
                                 + 0.947914 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.760403 w_k1_ADM_s110
                                 + 0.595136 w_k1_AXP_s110
                                 + 0.779821 w_k1_DIS_s110
                                 + 0.844334 w_k1_FDX_s110
                                 + 1.16132 w_k1_IBM_s110
                                 + 0.76869 w_k1_INTC_s110
                                 + 1.03964 w_k1_KO_s110 + 1.23256 w_k1_MAT_s110
                                 + 0.784092 w_k1_MSFT_s110
                                 + 0.864967 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.00273 w_k1_ADM_s110
                                 + 1.23332 w_k1_AXP_s110
                                 + 0.929145 w_k1_DIS_s110
                                 + 1.04127 w_k1_FDX_s110
                                 + 1.00927 w_k1_IBM_s110
                                 + 0.936088 w_k1_INTC_s110
                                 + 1.09976 w_k1_KO_s110 + 1.61929 w_k1_MAT_s110
                                 + 0.837656 w_k1_MSFT_s110
                                 + 0.960999 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.931325 w_k1_ADM_s110
                                 + 0.666198 w_k1_AXP_s110
                                 + 0.93008 w_k1_DIS_s110
                                 + 0.949782 w_k1_FDX_s110
                                 + 0.725738 w_k1_IBM_s110
                                 + 0.933641 w_k1_INTC_s110
                                 + 1.01113 w_k1_KO_s110 + 0.53892 w_k1_MAT_s110
                                 + 0.944547 w_k1_MSFT_s110
                                 + 0.966731 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.989464 w_k1_ADM_s110
                                 + 0.8827 w_k1_AXP_s110
                                 + 0.927114 w_k1_DIS_s110
                                 + 0.978187 w_k1_FDX_s110
                                 + 0.8203 w_k1_IBM_s110
                                 + 0.901573 w_k1_INTC_s110
                                 + 0.907222 w_k1_KO_s110
                                 + 0.679519 w_k1_MAT_s110
                                 + 0.846275 w_k1_MSFT_s110
                                 + 0.89589 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.947408 w_k1_ADM_s110
                                 + 1.03597 w_k1_AXP_s110
                                 + 1.37197 w_k1_DIS_s110
                                 + 1.15794 w_k1_FDX_s110
                                 + 0.950266 w_k1_IBM_s110
                                 + 1.01086 w_k1_INTC_s110
                                 + 1.22447 w_k1_KO_s110
                                 + 0.824004 w_k1_MAT_s110
                                 + 1.08275 w_k1_MSFT_s110
                                 + 1.03527 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.09032 w_k1_ADM_s110
                                 + 1.37187 w_k1_AXP_s110
                                 + 1.45023 w_k1_DIS_s110
                                 + 1.20587 w_k1_FDX_s110
                                 + 1.30717 w_k1_IBM_s110
                                 + 1.07382 w_k1_INTC_s110 + 1.5273 w_k1_KO_s110
                                 + 1.71037 w_k1_MAT_s110
                                 + 1.12288 w_k1_MSFT_s110
                                 + 1.09552 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.72811 w_k1_ADM_s111
                                 + 1.41059 w_k1_AXP_s111
                                 + 1.79019 w_k1_DIS_s111
                                 + 1.53688 w_k1_FDX_s111 + 1.0092 w_k1_IBM_s111
                                 + 1.44101 w_k1_INTC_s111
                                 + 1.20244 w_k1_KO_s111
                                 + 0.976026 w_k1_MAT_s111
                                 + 1.58573 w_k1_MSFT_s111
                                 + 1.27017 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.732168 w_k1_ADM_s111
                                 + 0.661281 w_k1_AXP_s111
                                 + 0.622734 w_k1_DIS_s111
                                 + 0.69651 w_k1_FDX_s111
                                 + 0.782529 w_k1_IBM_s111
                                 + 0.904756 w_k1_INTC_s111
                                 + 0.63028 w_k1_KO_s111
                                 + 0.388849 w_k1_MAT_s111
                                 + 0.779812 w_k1_MSFT_s111
                                 + 0.717399 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.863091 w_k1_ADM_s111
                                 + 1.20593 w_k1_AXP_s111
                                 + 0.96068 w_k1_DIS_s111
                                 + 0.898063 w_k1_FDX_s111
                                 + 1.33549 w_k1_IBM_s111
                                 + 1.04533 w_k1_INTC_s111
                                 + 1.02023 w_k1_KO_s111 + 1.22407 w_k1_MAT_s111
                                 + 1.05162 w_k1_MSFT_s111
                                 + 1.00158 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.41246 w_k1_ADM_s111
                                 + 0.870875 w_k1_AXP_s111
                                 + 1.21862 w_k1_DIS_s111
                                 + 1.05169 w_k1_FDX_s111
                                 + 0.913094 w_k1_IBM_s111
                                 + 1.23473 w_k1_INTC_s111
                                 + 1.00999 w_k1_KO_s111
                                 + 0.565763 w_k1_MAT_s111
                                 + 1.25977 w_k1_MSFT_s111 + 1.0596 w_k1_PG_s111
                                 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.799705 w_k1_ADM_s111
                                 + 0.763026 w_k1_AXP_s111
                                 + 0.818016 w_k1_DIS_s111
                                 + 0.89116 w_k1_FDX_s111
                                 + 0.719579 w_k1_IBM_s111
                                 + 1.0013 w_k1_INTC_s111
                                 + 0.729686 w_k1_KO_s111
                                 + 0.588134 w_k1_MAT_s111
                                 + 0.963432 w_k1_MSFT_s111
                                 + 0.804128 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.11318 w_k1_ADM_s111
                                 + 0.768744 w_k1_AXP_s111
                                 + 0.757511 w_k1_DIS_s111
                                 + 0.739954 w_k1_FDX_s111
                                 + 1.0301 w_k1_IBM_s111
                                 + 0.79976 w_k1_INTC_s111
                                 + 0.773626 w_k1_KO_s111
                                 + 0.825971 w_k1_MAT_s111
                                 + 0.90672 w_k1_MSFT_s111
                                 + 0.904914 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.998658 w_k1_ADM_s111
                                 + 1.34786 w_k1_AXP_s111
                                 + 1.16286 w_k1_DIS_s111
                                 + 1.25891 w_k1_FDX_s111
                                 + 0.633543 w_k1_IBM_s111
                                 + 1.46442 w_k1_INTC_s111
                                 + 0.800659 w_k1_KO_s111
                                 + 0.554407 w_k1_MAT_s111
                                 + 1.29241 w_k1_MSFT_s111
                                 + 0.911666 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.17523 w_k1_ADM_s111
                                 + 1.06616 w_k1_AXP_s111
                                 + 1.44624 w_k1_DIS_s111 + 1.0161 w_k1_FDX_s111
                                 + 1.55223 w_k1_IBM_s111
                                 + 0.994829 w_k1_INTC_s111
                                 + 1.23694 w_k1_KO_s111 + 1.24676 w_k1_MAT_s111
                                 + 1.28865 w_k1_MSFT_s111
                                 + 1.21498 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.0941 w_k1_ADM_s112
                                 + 0.776879 w_k1_AXP_s112
                                 + 1.04732 w_k1_DIS_s112
                                 + 0.937583 w_k1_FDX_s112
                                 + 0.773744 w_k1_IBM_s112
                                 + 0.80806 w_k1_INTC_s112
                                 + 0.982031 w_k1_KO_s112
                                 + 0.879149 w_k1_MAT_s112
                                 + 1.15761 w_k1_MSFT_s112
                                 + 1.09698 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.997063 w_k1_ADM_s112
                                 + 0.942438 w_k1_AXP_s112
                                 + 0.824569 w_k1_DIS_s112
                                 + 0.899516 w_k1_FDX_s112
                                 + 1.01208 w_k1_IBM_s112
                                 + 1.10391 w_k1_INTC_s112
                                 + 0.945893 w_k1_KO_s112
                                 + 1.32877 w_k1_MAT_s112
                                 + 0.971002 w_k1_MSFT_s112
                                 + 1.04847 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.1796 w_k1_ADM_s112
                                 + 1.01985 w_k1_AXP_s112
                                 + 1.14305 w_k1_DIS_s112
                                 + 0.941768 w_k1_FDX_s112
                                 + 1.0637 w_k1_IBM_s112
                                 + 0.972093 w_k1_INTC_s112
                                 + 1.05908 w_k1_KO_s112 + 1.17479 w_k1_MAT_s112
                                 + 1.18034 w_k1_MSFT_s112
                                 + 1.13358 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.03751 w_k1_ADM_s112
                                 + 0.846849 w_k1_AXP_s112
                                 + 0.608998 w_k1_DIS_s112
                                 + 0.731999 w_k1_FDX_s112
                                 + 0.827177 w_k1_IBM_s112
                                 + 0.81375 w_k1_INTC_s112
                                 + 0.823303 w_k1_KO_s112
                                 + 0.864612 w_k1_MAT_s112
                                 + 0.886208 w_k1_MSFT_s112
                                 + 0.957724 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.01841 w_k1_ADM_s112
                                 + 1.29435 w_k1_AXP_s112
                                 + 1.11748 w_k1_DIS_s112
                                 + 0.922361 w_k1_FDX_s112
                                 + 1.56116 w_k1_IBM_s112
                                 + 0.781186 w_k1_INTC_s112
                                 + 1.33451 w_k1_KO_s112 + 1.80697 w_k1_MAT_s112
                                 + 0.968692 w_k1_MSFT_s112
                                 + 1.27971 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.04975 w_k1_ADM_s112
                                 + 0.845257 w_k1_AXP_s112
                                 + 0.86808 w_k1_DIS_s112
                                 + 0.929448 w_k1_FDX_s112
                                 + 0.729054 w_k1_IBM_s112
                                 + 1.00187 w_k1_INTC_s112
                                 + 0.693606 w_k1_KO_s112
                                 + 0.597915 w_k1_MAT_s112
                                 + 1.12204 w_k1_MSFT_s112
                                 + 0.97659 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.0922 w_k1_ADM_s112
                                 + 1.34814 w_k1_AXP_s112
                                 + 1.34434 w_k1_DIS_s112
                                 + 1.27034 w_k1_FDX_s112
                                 + 0.910442 w_k1_IBM_s112
                                 + 1.0465 w_k1_INTC_s112 + 1.08834 w_k1_KO_s112
                                 + 1.21542 w_k1_MAT_s112
                                 + 1.28699 w_k1_MSFT_s112 + 1.2026 w_k1_PG_s112
                                 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.19309 w_k1_ADM_s112
                                 + 1.47589 w_k1_AXP_s112
                                 + 1.46549 w_k1_DIS_s112
                                 + 1.23775 w_k1_FDX_s112 + 1.2627 w_k1_IBM_s112
                                 + 1.04837 w_k1_INTC_s112
                                 + 1.12546 w_k1_KO_s112 + 1.39257 w_k1_MAT_s112
                                 + 1.24645 w_k1_MSFT_s112
                                 + 1.27106 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.873546 w_k1_ADM_s113
                                 + 0.978088 w_k1_AXP_s113
                                 + 0.925438 w_k1_DIS_s113
                                 + 0.843417 w_k1_FDX_s113
                                 + 1.14398 w_k1_IBM_s113
                                 + 1.14028 w_k1_INTC_s113
                                 + 1.14877 w_k1_KO_s113 + 1.0847 w_k1_MAT_s113
                                 + 0.936077 w_k1_MSFT_s113
                                 + 0.97272 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.08537 w_k1_ADM_s113
                                 + 0.921402 w_k1_AXP_s113
                                 + 1.20294 w_k1_DIS_s113
                                 + 1.08988 w_k1_FDX_s113
                                 + 0.896933 w_k1_IBM_s113
                                 + 1.04806 w_k1_INTC_s113
                                 + 1.03184 w_k1_KO_s113
                                 + 0.908697 w_k1_MAT_s113
                                 + 1.11172 w_k1_MSFT_s113
                                 + 1.04122 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.11067 w_k1_ADM_s113
                                 + 1.34806 w_k1_AXP_s113
                                 + 1.09857 w_k1_DIS_s113
                                 + 1.33666 w_k1_FDX_s113
                                 + 0.851755 w_k1_IBM_s113
                                 + 1.72846 w_k1_INTC_s113
                                 + 1.07437 w_k1_KO_s113 + 1.12171 w_k1_MAT_s113
                                 + 1.19075 w_k1_MSFT_s113
                                 + 0.949559 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.07389 w_k1_ADM_s113
                                 + 0.869236 w_k1_AXP_s113
                                 + 0.855404 w_k1_DIS_s113
                                 + 0.902092 w_k1_FDX_s113
                                 + 0.635844 w_k1_IBM_s113
                                 + 0.920234 w_k1_INTC_s113
                                 + 0.816884 w_k1_KO_s113
                                 + 0.725938 w_k1_MAT_s113
                                 + 0.824132 w_k1_MSFT_s113
                                 + 0.855576 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.00378 w_k1_ADM_s113
                                 + 1.30604 w_k1_AXP_s113
                                 + 1.21908 w_k1_DIS_s113
                                 + 1.06084 w_k1_FDX_s113
                                 + 1.64382 w_k1_IBM_s113
                                 + 1.24255 w_k1_INTC_s113
                                 + 1.33678 w_k1_KO_s113 + 1.95305 w_k1_MAT_s113
                                 + 1.07279 w_k1_MSFT_s113
                                 + 1.29073 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.4309 w_k1_ADM_s113
                                 + 1.60596 w_k1_AXP_s113 + 1.5554 w_k1_DIS_s113
                                 + 1.38221 w_k1_FDX_s113
                                 + 1.26545 w_k1_IBM_s113
                                 + 1.88745 w_k1_INTC_s113
                                 + 1.28885 w_k1_KO_s113
                                 + 0.940123 w_k1_MAT_s113
                                 + 1.3329 w_k1_MSFT_s113 + 1.22814 w_k1_PG_s113
                                 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.997773 w_k1_ADM_s113
                                 + 0.883689 w_k1_AXP_s113
                                 + 0.84026 w_k1_DIS_s113
                                 + 0.990564 w_k1_FDX_s113
                                 + 0.962903 w_k1_IBM_s113
                                 + 1.06561 w_k1_INTC_s113
                                 + 0.893021 w_k1_KO_s113
                                 + 0.945204 w_k1_MAT_s113
                                 + 0.86477 w_k1_MSFT_s113
                                 + 1.02839 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.04745 w_k1_ADM_s113
                                 + 1.33111 w_k1_AXP_s113 + 1.388 w_k1_DIS_s113
                                 + 1.15492 w_k1_FDX_s113
                                 + 1.23686 w_k1_IBM_s113
                                 + 1.50763 w_k1_INTC_s113
                                 + 1.14955 w_k1_KO_s113 + 1.05839 w_k1_MAT_s113
                                 + 1.07306 w_k1_MSFT_s113 + 1.0553 w_k1_PG_s113
                                 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.872859 w_k1_ADM_s114
                                 + 1.33592 w_k1_AXP_s114
                                 + 0.87022 w_k1_DIS_s114
                                 + 0.955612 w_k1_FDX_s114
                                 + 1.33669 w_k1_IBM_s114
                                 + 1.10063 w_k1_INTC_s114 + 1.0003 w_k1_KO_s114
                                 + 1.68221 w_k1_MAT_s114
                                 + 0.935769 w_k1_MSFT_s114
                                 + 0.922225 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.897256 w_k1_ADM_s114
                                 + 1.67034 w_k1_AXP_s114
                                 + 1.02408 w_k1_DIS_s114
                                 + 1.18393 w_k1_FDX_s114
                                 + 0.992902 w_k1_IBM_s114
                                 + 1.35403 w_k1_INTC_s114
                                 + 0.916802 w_k1_KO_s114
                                 + 1.72559 w_k1_MAT_s114 + 0.968 w_k1_MSFT_s114
                                 + 1.14574 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.24315 w_k1_ADM_s114
                                 + 1.40485 w_k1_AXP_s114
                                 + 1.77762 w_k1_DIS_s114
                                 + 1.31292 w_k1_FDX_s114
                                 + 1.39672 w_k1_IBM_s114
                                 + 1.04025 w_k1_INTC_s114
                                 + 1.37969 w_k1_KO_s114 + 1.40321 w_k1_MAT_s114
                                 + 1.37845 w_k1_MSFT_s114
                                 + 1.31006 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.840289 w_k1_ADM_s114
                                 + 1.0264 w_k1_AXP_s114
                                 + 0.782474 w_k1_DIS_s114
                                 + 0.864443 w_k1_FDX_s114
                                 + 0.917317 w_k1_IBM_s114
                                 + 0.98619 w_k1_INTC_s114
                                 + 0.83965 w_k1_KO_s114 + 1.22706 w_k1_MAT_s114
                                 + 0.964125 w_k1_MSFT_s114
                                 + 1.04312 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.931298 w_k1_ADM_s114
                                 + 0.942432 w_k1_AXP_s114
                                 + 1.11987 w_k1_DIS_s114
                                 + 1.01849 w_k1_FDX_s114
                                 + 1.17515 w_k1_IBM_s114
                                 + 1.0946 w_k1_INTC_s114 + 1.03103 w_k1_KO_s114
                                 + 1.37075 w_k1_MAT_s114
                                 + 1.17425 w_k1_MSFT_s114
                                 + 1.16825 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.84413 w_k1_ADM_s114
                                 + 0.724917 w_k1_AXP_s114
                                 + 0.659282 w_k1_DIS_s114
                                 + 0.786115 w_k1_FDX_s114
                                 + 0.772488 w_k1_IBM_s114
                                 + 0.791878 w_k1_INTC_s114
                                 + 0.687967 w_k1_KO_s114
                                 + 0.922131 w_k1_MAT_s114
                                 + 0.820917 w_k1_MSFT_s114
                                 + 0.929052 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.870217 w_k1_ADM_s114
                                 + 0.884077 w_k1_AXP_s114
                                 + 1.02229 w_k1_DIS_s114
                                 + 0.908673 w_k1_FDX_s114
                                 + 1.56091 w_k1_IBM_s114
                                 + 0.960152 w_k1_INTC_s114
                                 + 1.02307 w_k1_KO_s114 + 1.06723 w_k1_MAT_s114
                                 + 0.868865 w_k1_MSFT_s114
                                 + 0.98832 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.15122 w_k1_ADM_s114
                                 + 0.982155 w_k1_AXP_s114
                                 + 0.921446 w_k1_DIS_s114
                                 + 1.08071 w_k1_FDX_s114
                                 + 0.75542 w_k1_IBM_s114
                                 + 1.07232 w_k1_INTC_s114 + 1.0009 w_k1_KO_s114
                                 + 0.766531 w_k1_MAT_s114
                                 + 1.01078 w_k1_MSFT_s114
                                 + 1.01474 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.84356 w_k1_ADM_s115
                                 + 0.656366 w_k1_AXP_s115
                                 + 0.66102 w_k1_DIS_s115
                                 + 0.736831 w_k1_FDX_s115
                                 + 0.942936 w_k1_IBM_s115
                                 + 1.05812 w_k1_INTC_s115
                                 + 0.86504 w_k1_KO_s115
                                 + 0.845725 w_k1_MAT_s115
                                 + 0.962262 w_k1_MSFT_s115
                                 + 0.838432 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.33494 w_k1_ADM_s115
                                 + 1.88867 w_k1_AXP_s115
                                 + 1.50858 w_k1_DIS_s115
                                 + 1.36962 w_k1_FDX_s115
                                 + 1.33799 w_k1_IBM_s115
                                 + 1.3213 w_k1_INTC_s115 + 1.31443 w_k1_KO_s115
                                 + 2.34828 w_k1_MAT_s115
                                 + 1.29041 w_k1_MSFT_s115
                                 + 1.21187 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.06125 w_k1_ADM_s115
                                 + 1.27489 w_k1_AXP_s115
                                 + 1.37032 w_k1_DIS_s115
                                 + 1.25182 w_k1_FDX_s115
                                 + 0.948959 w_k1_IBM_s115
                                 + 1.4235 w_k1_INTC_s115
                                 + 0.998101 w_k1_KO_s115
                                 + 0.90849 w_k1_MAT_s115
                                 + 1.39002 w_k1_MSFT_s115 + 1.1598 w_k1_PG_s115
                                 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.06251 w_k1_ADM_s115
                                 + 1.05179 w_k1_AXP_s115
                                 + 1.20005 w_k1_DIS_s115
                                 + 1.26932 w_k1_FDX_s115
                                 + 0.755082 w_k1_IBM_s115
                                 + 1.36897 w_k1_INTC_s115
                                 + 0.943388 w_k1_KO_s115
                                 + 0.707734 w_k1_MAT_s115
                                 + 1.19236 w_k1_MSFT_s115
                                 + 0.902325 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.926824 w_k1_ADM_s115
                                 + 1.14027 w_k1_AXP_s115
                                 + 1.21151 w_k1_DIS_s115
                                 + 1.16237 w_k1_FDX_s115
                                 + 1.41029 w_k1_IBM_s115
                                 + 1.35582 w_k1_INTC_s115
                                 + 1.10109 w_k1_KO_s115 + 1.69172 w_k1_MAT_s115
                                 + 1.28162 w_k1_MSFT_s115
                                 + 1.00617 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.832481 w_k1_ADM_s115
                                 + 1.45671 w_k1_AXP_s115
                                 + 0.868891 w_k1_DIS_s115
                                 + 0.963447 w_k1_FDX_s115
                                 + 1.08888 w_k1_IBM_s115
                                 + 1.02727 w_k1_INTC_s115
                                 + 0.878358 w_k1_KO_s115
                                 + 1.53492 w_k1_MAT_s115
                                 + 0.888784 w_k1_MSFT_s115
                                 + 0.967469 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.04236 w_k1_ADM_s115
                                 + 0.690193 w_k1_AXP_s115
                                 + 0.923734 w_k1_DIS_s115
                                 + 0.811465 w_k1_FDX_s115
                                 + 1.13518 w_k1_IBM_s115
                                 + 0.921479 w_k1_INTC_s115
                                 + 0.929959 w_k1_KO_s115
                                 + 0.947429 w_k1_MAT_s115
                                 + 0.958673 w_k1_MSFT_s115
                                 + 0.865046 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.851977 w_k1_ADM_s115
                                 + 0.878621 w_k1_AXP_s115
                                 + 0.807263 w_k1_DIS_s115
                                 + 0.881015 w_k1_FDX_s115
                                 + 0.760111 w_k1_IBM_s115
                                 + 0.739315 w_k1_INTC_s115
                                 + 0.714644 w_k1_KO_s115
                                 + 1.04275 w_k1_MAT_s115
                                 + 0.837699 w_k1_MSFT_s115
                                 + 0.853012 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.804525 w_k1_ADM_s116
                                 + 0.807946 w_k1_AXP_s116
                                 + 0.61516 w_k1_DIS_s116
                                 + 0.828528 w_k1_FDX_s116
                                 + 0.757278 w_k1_IBM_s116
                                 + 0.725755 w_k1_INTC_s116
                                 + 0.813578 w_k1_KO_s116
                                 + 0.779649 w_k1_MAT_s116
                                 + 0.880813 w_k1_MSFT_s116
                                 + 0.863334 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.843186 w_k1_ADM_s116
                                 + 0.965573 w_k1_AXP_s116
                                 + 0.931769 w_k1_DIS_s116
                                 + 0.962255 w_k1_FDX_s116
                                 + 1.04844 w_k1_IBM_s116
                                 + 1.07287 w_k1_INTC_s116
                                 + 1.08267 w_k1_KO_s116
                                 + 0.975789 w_k1_MAT_s116
                                 + 1.0164 w_k1_MSFT_s116
                                 + 0.928583 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.09232 w_k1_ADM_s116
                                 + 1.17906 w_k1_AXP_s116 + 1.5367 w_k1_DIS_s116
                                 + 1.25676 w_k1_FDX_s116
                                 + 0.982393 w_k1_IBM_s116
                                 + 1.09693 w_k1_INTC_s116
                                 + 1.20323 w_k1_KO_s116
                                 + 0.820771 w_k1_MAT_s116
                                 + 1.41454 w_k1_MSFT_s116
                                 + 0.951188 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.09988 w_k1_ADM_s116
                                 + 0.93267 w_k1_AXP_s116
                                 + 1.05208 w_k1_DIS_s116 + 0.9237 w_k1_FDX_s116
                                 + 1.52496 w_k1_IBM_s116
                                 + 1.01496 w_k1_INTC_s116
                                 + 1.36844 w_k1_KO_s116 + 1.32057 w_k1_MAT_s116
                                 + 1.01608 w_k1_MSFT_s116
                                 + 0.964762 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.890622 w_k1_ADM_s116
                                 + 0.977084 w_k1_AXP_s116
                                 + 0.936473 w_k1_DIS_s116
                                 + 1.01326 w_k1_FDX_s116
                                 + 0.71964 w_k1_IBM_s116
                                 + 1.06504 w_k1_INTC_s116
                                 + 0.783297 w_k1_KO_s116
                                 + 0.651444 w_k1_MAT_s116
                                 + 1.07325 w_k1_MSFT_s116
                                 + 0.879766 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.18725 w_k1_ADM_s116
                                 + 1.48572 w_k1_AXP_s116
                                 + 1.40411 w_k1_DIS_s116 + 1.2717 w_k1_FDX_s116
                                 + 1.27772 w_k1_IBM_s116
                                 + 1.02761 w_k1_INTC_s116
                                 + 1.22518 w_k1_KO_s116 + 1.5496 w_k1_MAT_s116
                                 + 1.05078 w_k1_MSFT_s116
                                 + 1.04987 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.44607 w_k1_ADM_s116
                                 + 1.25345 w_k1_AXP_s116
                                 + 1.02792 w_k1_DIS_s116
                                 + 1.16387 w_k1_FDX_s116
                                 + 1.15948 w_k1_IBM_s116
                                 + 0.933324 w_k1_INTC_s116
                                 + 1.21819 w_k1_KO_s116 + 1.21038 w_k1_MAT_s116
                                 + 1.14717 w_k1_MSFT_s116 + 1.1775 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.01383 w_k1_ADM_s116
                                 + 0.788348 w_k1_AXP_s116
                                 + 0.876826 w_k1_DIS_s116
                                 + 0.840748 w_k1_FDX_s116
                                 + 0.88594 w_k1_IBM_s116
                                 + 0.694197 w_k1_INTC_s116
                                 + 0.932982 w_k1_KO_s116
                                 + 0.725785 w_k1_MAT_s116
                                 + 0.925882 w_k1_MSFT_s116
                                 + 0.771388 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.27156 w_k1_ADM_s117
                                 + 1.18604 w_k1_AXP_s117
                                 + 1.25131 w_k1_DIS_s117
                                 + 1.23917 w_k1_FDX_s117
                                 + 0.866138 w_k1_IBM_s117
                                 + 1.16736 w_k1_INTC_s117
                                 + 1.01396 w_k1_KO_s117 + 1.0129 w_k1_MAT_s117
                                 + 1.22385 w_k1_MSFT_s117
                                 + 1.07634 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.949044 w_k1_ADM_s117
                                 + 1.44231 w_k1_AXP_s117
                                 + 1.23145 w_k1_DIS_s117 + 1.1135 w_k1_FDX_s117
                                 + 1.65768 w_k1_IBM_s117
                                 + 1.61403 w_k1_INTC_s117
                                 + 1.44622 w_k1_KO_s117 + 1.6654 w_k1_MAT_s117
                                 + 1.25529 w_k1_MSFT_s117
                                 + 1.34988 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.708512 w_k1_ADM_s117
                                 + 0.846081 w_k1_AXP_s117
                                 + 0.571233 w_k1_DIS_s117
                                 + 0.778507 w_k1_FDX_s117
                                 + 0.815589 w_k1_IBM_s117
                                 + 0.827604 w_k1_INTC_s117
                                 + 0.730566 w_k1_KO_s117
                                 + 1.09484 w_k1_MAT_s117
                                 + 0.724347 w_k1_MSFT_s117
                                 + 0.918321 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.56633 w_k1_ADM_s117
                                 + 1.92159 w_k1_AXP_s117 + 1.74 w_k1_DIS_s117
                                 + 1.47706 w_k1_FDX_s117
                                 + 1.29635 w_k1_IBM_s117
                                 + 1.59432 w_k1_INTC_s117
                                 + 1.52452 w_k1_KO_s117 + 1.81499 w_k1_MAT_s117
                                 + 1.51665 w_k1_MSFT_s117
                                 + 1.45772 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.01937 w_k1_ADM_s117
                                 + 1.01791 w_k1_AXP_s117
                                 + 0.926145 w_k1_DIS_s117
                                 + 0.949631 w_k1_FDX_s117
                                 + 1.20391 w_k1_IBM_s117
                                 + 1.03697 w_k1_INTC_s117 + 0.9878 w_k1_KO_s117
                                 + 1.12146 w_k1_MAT_s117
                                 + 0.969434 w_k1_MSFT_s117
                                 + 1.12606 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.17305 w_k1_ADM_s117
                                 + 1.57891 w_k1_AXP_s117
                                 + 1.32245 w_k1_DIS_s117
                                 + 1.20746 w_k1_FDX_s117
                                 + 1.13245 w_k1_IBM_s117
                                 + 1.2774 w_k1_INTC_s117 + 1.03099 w_k1_KO_s117
                                 + 1.46831 w_k1_MAT_s117
                                 + 1.25531 w_k1_MSFT_s117
                                 + 1.25611 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.855444 w_k1_ADM_s117
                                 + 0.840433 w_k1_AXP_s117
                                 + 0.752162 w_k1_DIS_s117
                                 + 0.754405 w_k1_FDX_s117
                                 + 0.991295 w_k1_IBM_s117
                                 + 0.890199 w_k1_INTC_s117
                                 + 0.800176 w_k1_KO_s117
                                 + 0.693088 w_k1_MAT_s117
                                 + 0.727432 w_k1_MSFT_s117
                                 + 0.865413 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.962657 w_k1_ADM_s117
                                 + 1.05662 w_k1_AXP_s117
                                 + 0.91477 w_k1_DIS_s117
                                 + 0.96796 w_k1_FDX_s117
                                 + 0.749531 w_k1_IBM_s117
                                 + 1.10214 w_k1_INTC_s117
                                 + 0.795855 w_k1_KO_s117
                                 + 0.843894 w_k1_MAT_s117
                                 + 0.968705 w_k1_MSFT_s117
                                 + 1.0382 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0490242776043445 w_k1_ADM_s000
                                 + 0.0170646621301462 w_k1_ADM_s001
                                 + 0.0261180443900376 w_k1_ADM_s002
                                 + 0.0331567622733721 w_k1_ADM_s003
                                 + 0.0587948160907833 w_k1_ADM_s004
                                 + 0.0352467999804776 w_k1_ADM_s005
                                 + 0.040368640187806 w_k1_ADM_s006
                                 + 0.0235824826157508 w_k1_ADM_s007
                                 + 0.0330558456051856 w_k1_ADM_s010
                                 + 0.0270603296619678 w_k1_ADM_s011
                                 + 0.0362820459178375 w_k1_ADM_s012
                                 + 0.016383470039485 w_k1_ADM_s013
                                 + 0.0131289574745975 w_k1_ADM_s014
                                 + 0.045560702484063 w_k1_ADM_s015
                                 + 0.0559640423743124 w_k1_ADM_s016
                                 + 0.0757479246773 w_k1_ADM_s017
                                 + 0.0134138393766208 w_k1_ADM_s100
                                 + 0.0457010853578242 w_k1_ADM_s101
                                 + 0.0626349648377057 w_k1_ADM_s102
                                 + 0.0213889727754079 w_k1_ADM_s103
                                 + 0.0147542513738024 w_k1_ADM_s104
                                 + 0.0675128228625596 w_k1_ADM_s105
                                 + 0.0521890118454585 w_k1_ADM_s106
                                 + 0.014169109801304 w_k1_ADM_s107
                                 + 0.0114026714353024 w_k1_ADM_s110
                                 + 0.0081713327849505 w_k1_ADM_s111
                                 + 0.0219226940591818 w_k1_ADM_s112
                                 + 0.00641167290308599 w_k1_ADM_s113
                                 + 0.0176113387004558 w_k1_ADM_s114
                                 + 0.00811245287866667 w_k1_ADM_s115
                                 + 0.00692106878304914 w_k1_ADM_s116
                                 + 0.0241678842580418 w_k1_ADM_s117
                                 + 0.0567973322137379 w_k1_AXP_s000
                                 + 0.0163858236217482 w_k1_AXP_s001
                                 + 0.0267039013061749 w_k1_AXP_s002
                                 + 0.0343687700807646 w_k1_AXP_s003
                                 + 0.0650716993773032 w_k1_AXP_s004
                                 + 0.0342121781142454 w_k1_AXP_s005
                                 + 0.0359401641990748 w_k1_AXP_s006
                                 + 0.0243001690732521 w_k1_AXP_s007
                                 + 0.0403822137042187 w_k1_AXP_s010
                                 + 0.0243094787962244 w_k1_AXP_s011
                                 + 0.0371639486698675 w_k1_AXP_s012
                                 + 0.0192687998948531 w_k1_AXP_s013
                                 + 0.0178537023853813 w_k1_AXP_s014
                                 + 0.0505406084225457 w_k1_AXP_s015
                                 + 0.0586744972799711 w_k1_AXP_s016
                                 + 0.0704873663546611 w_k1_AXP_s017
                                 + 0.0159461082401784 w_k1_AXP_s100
                                 + 0.0450510504081438 w_k1_AXP_s101
                                 + 0.0648511471940104 w_k1_AXP_s102
                                 + 0.0212501167489418 w_k1_AXP_s103
                                 + 0.0153018696162767 w_k1_AXP_s104
                                 + 0.0690354733584139 w_k1_AXP_s105
                                 + 0.0508223708455409 w_k1_AXP_s106
                                 + 0.0152038663739099 w_k1_AXP_s107
                                 + 0.0123706497362524 w_k1_AXP_s110
                                 + 0.00692892412887561 w_k1_AXP_s111
                                 + 0.0221200609082631 w_k1_AXP_s112
                                 + 0.00667557231429484 w_k1_AXP_s113
                                 + 0.0185529061857974 w_k1_AXP_s114
                                 + 0.00937222425168102 w_k1_AXP_s115
                                 + 0.00705609444081292 w_k1_AXP_s116
                                 + 0.027764591359307 w_k1_AXP_s117
                                 + 0.0540693705991228 w_k1_DIS_s000
                                 + 0.0185765136708912 w_k1_DIS_s001
                                 + 0.0238946078367958 w_k1_DIS_s002
                                 + 0.0388597705519629 w_k1_DIS_s003
                                 + 0.0682828400456465 w_k1_DIS_s004
                                 + 0.0363335214472884 w_k1_DIS_s005
                                 + 0.0377665867214219 w_k1_DIS_s006
                                 + 0.0261088185718473 w_k1_DIS_s007
                                 + 0.0406130696679263 w_k1_DIS_s010
                                 + 0.0240023609957327 w_k1_DIS_s011
                                 + 0.0368243911866379 w_k1_DIS_s012
                                 + 0.0189884645214089 w_k1_DIS_s013
                                 + 0.0168223257768576 w_k1_DIS_s014
                                 + 0.0506271961270057 w_k1_DIS_s015
                                 + 0.0663670796858154 w_k1_DIS_s016
                                 + 0.0737043401665535 w_k1_DIS_s017
                                 + 0.0143640469527069 w_k1_DIS_s100
                                 + 0.0451118553522323 w_k1_DIS_s101
                                 + 0.0719055423600975 w_k1_DIS_s102
                                 + 0.0211058903766235 w_k1_DIS_s103
                                 + 0.0147550293497915 w_k1_DIS_s104
                                 + 0.0682566847874762 w_k1_DIS_s105
                                 + 0.0519574146835388 w_k1_DIS_s106
                                 + 0.016164353745151 w_k1_DIS_s107
                                 + 0.0135965534637845 w_k1_DIS_s110
                                 + 0.00770654828198955 w_k1_DIS_s111
                                 + 0.0213622350792402 w_k1_DIS_s112
                                 + 0.00651662634582034 w_k1_DIS_s113
                                 + 0.0180392821578894 w_k1_DIS_s114
                                 + 0.00871096159672903 w_k1_DIS_s115
                                 + 0.00716860794108066 w_k1_DIS_s116
                                 + 0.0242282026093258 w_k1_DIS_s117
                                 + 0.0545089274342502 w_k1_FDX_s000
                                 + 0.0175626181523402 w_k1_FDX_s001
                                 + 0.0241932496850935 w_k1_FDX_s002
                                 + 0.0349905256932646 w_k1_FDX_s003
                                 + 0.0671069987117845 w_k1_FDX_s004
                                 + 0.0339176539837151 w_k1_FDX_s005
                                 + 0.0375326039509754 w_k1_FDX_s006
                                 + 0.0254797198097156 w_k1_FDX_s007
                                 + 0.038974297915263 w_k1_FDX_s010
                                 + 0.02311589366214 w_k1_FDX_s011
                                 + 0.0374019846241987 w_k1_FDX_s012
                                 + 0.0185852293266307 w_k1_FDX_s013
                                 + 0.0155742357372981 w_k1_FDX_s014
                                 + 0.0490727453660122 w_k1_FDX_s015
                                 + 0.0592289744799248 w_k1_FDX_s016
                                 + 0.0739240346321042 w_k1_FDX_s017
                                 + 0.0152498249305342 w_k1_FDX_s100
                                 + 0.0450334992259217 w_k1_FDX_s101
                                 + 0.0659779732793586 w_k1_FDX_s102
                                 + 0.0212464516706978 w_k1_FDX_s103
                                 + 0.0154143990272213 w_k1_FDX_s104
                                 + 0.0699125580012492 w_k1_FDX_s105
                                 + 0.0493757220113025 w_k1_FDX_s106
                                 + 0.0159594631128776 w_k1_FDX_s107
                                 + 0.0128234769214667 w_k1_FDX_s110
                                 + 0.00702695869478679 w_k1_FDX_s111
                                 + 0.0200297920802326 w_k1_FDX_s112
                                 + 0.00647680375987736 w_k1_FDX_s113
                                 + 0.0179958822640035 w_k1_FDX_s114
                                 + 0.00872006821500748 w_k1_FDX_s115
                                 + 0.00698262666273138 w_k1_FDX_s116
                                 + 0.0241364866775597 w_k1_FDX_s117
                                 + 0.0557243381394838 w_k1_IBM_s000
                                 + 0.0181829453809023 w_k1_IBM_s001
                                 + 0.0269119092403562 w_k1_IBM_s002
                                 + 0.0355547906379666 w_k1_IBM_s003
                                 + 0.0659329779555967 w_k1_IBM_s004
                                 + 0.0372935634464287 w_k1_IBM_s005
                                 + 0.0372547920694771 w_k1_IBM_s006
                                 + 0.0259048690815221 w_k1_IBM_s007
                                 + 0.0400567964250602 w_k1_IBM_s010
                                 + 0.0225965178981335 w_k1_IBM_s011
                                 + 0.0378953412027214 w_k1_IBM_s012
                                 + 0.0196251417985558 w_k1_IBM_s013
                                 + 0.0160688189390469 w_k1_IBM_s014
                                 + 0.0532811375913415 w_k1_IBM_s015
                                 + 0.0604790061141969 w_k1_IBM_s016
                                 + 0.076114498921294 w_k1_IBM_s017
                                 + 0.0152291645705354 w_k1_IBM_s100
                                 + 0.0436826872255755 w_k1_IBM_s101
                                 + 0.0669375898213765 w_k1_IBM_s102
                                 + 0.0210460869839521 w_k1_IBM_s103
                                 + 0.0156785338279478 w_k1_IBM_s104
                                 + 0.071530140674842 w_k1_IBM_s105
                                 + 0.0514032939277178 w_k1_IBM_s106
                                 + 0.0159802231537276 w_k1_IBM_s107
                                 + 0.0131946681614845 w_k1_IBM_s110
                                 + 0.00746240416806652 w_k1_IBM_s111
                                 + 0.0213236376019877 w_k1_IBM_s112
                                 + 0.00616209458693801 w_k1_IBM_s113
                                 + 0.0188964767592733 w_k1_IBM_s114
                                 + 0.0085341344710982 w_k1_IBM_s115
                                 + 0.00704326647766782 w_k1_IBM_s116
                                 + 0.0248050215667142 w_k1_IBM_s117
                                 + 0.0585505261235627 w_k1_INTC_s000
                                 + 0.0171992768451644 w_k1_INTC_s001
                                 + 0.0257927671000775 w_k1_INTC_s002
                                 + 0.0395269949407122 w_k1_INTC_s003
                                 + 0.0702054714813641 w_k1_INTC_s004
                                 + 0.0333939978321538 w_k1_INTC_s005
                                 + 0.0351346432074557 w_k1_INTC_s006
                                 + 0.0262316062201567 w_k1_INTC_s007
                                 + 0.0431028899704216 w_k1_INTC_s010
                                 + 0.0247502170949059 w_k1_INTC_s011
                                 + 0.0350383987146051 w_k1_INTC_s012
                                 + 0.0182184093947887 w_k1_INTC_s013
                                 + 0.0151400726365746 w_k1_INTC_s014
                                 + 0.0459804744301448 w_k1_INTC_s015
                                 + 0.0599279846312834 w_k1_INTC_s016
                                 + 0.0728071922037822 w_k1_INTC_s017
                                 + 0.018427091424319 w_k1_INTC_s100
                                 + 0.0509502793447969 w_k1_INTC_s101
                                 + 0.0681874286377046 w_k1_INTC_s102
                                 + 0.0188843806981666 w_k1_INTC_s103
                                 + 0.0167277319081547 w_k1_INTC_s104
                                 + 0.0707992246157552 w_k1_INTC_s105
                                 + 0.0520268288617377 w_k1_INTC_s106
                                 + 0.0196883706281509 w_k1_INTC_s107
                                 + 0.0115501629990397 w_k1_INTC_s110
                                 + 0.00772581525744914 w_k1_INTC_s111
                                 + 0.019729515346847 w_k1_INTC_s112
                                 + 0.00766284439520597 w_k1_INTC_s113
                                 + 0.018128616168366 w_k1_INTC_s114
                                 + 0.00925626864186444 w_k1_INTC_s115
                                 + 0.0065506688625395 w_k1_INTC_s116
                                 + 0.0267880090065273 w_k1_INTC_s117
                                 + 0.0576138900725175 w_k1_KO_s000
                                 + 0.0184771695159764 w_k1_KO_s001
                                 + 0.0237893783529187 w_k1_KO_s002
                                 + 0.0386024659002691 w_k1_KO_s003
                                 + 0.060316585392064 w_k1_KO_s004
                                 + 0.0343946643880621 w_k1_KO_s005
                                 + 0.0378643002310165 w_k1_KO_s006
                                 + 0.0280959113692633 w_k1_KO_s007
                                 + 0.0418040315502914 w_k1_KO_s010
                                 + 0.0233176397180751 w_k1_KO_s011
                                 + 0.0355251100420197 w_k1_KO_s012
                                 + 0.0191324279064923 w_k1_KO_s013
                                 + 0.0139715809182507 w_k1_KO_s014
                                 + 0.049645645876575 w_k1_KO_s015
                                 + 0.058404521938281 w_k1_KO_s016
                                 + 0.0738013895826512 w_k1_KO_s017
                                 + 0.0151737264639098 w_k1_KO_s100
                                 + 0.0408213397210616 w_k1_KO_s101
                                 + 0.0641899586250176 w_k1_KO_s102
                                 + 0.0209722219637984 w_k1_KO_s103
                                 + 0.0145124890132576 w_k1_KO_s104
                                 + 0.0681833806730246 w_k1_KO_s105
                                 + 0.0491163543147527 w_k1_KO_s106
                                 + 0.0175675267356207 w_k1_KO_s107
                                 + 0.0145691925496915 w_k1_KO_s110
                                 + 0.00672752912282923 w_k1_KO_s111
                                 + 0.0205945187183405 w_k1_KO_s112
                                 + 0.00629359580291807 w_k1_KO_s113
                                 + 0.0173466983763163 w_k1_KO_s114
                                 + 0.00782875615185453 w_k1_KO_s115
                                 + 0.00729209333377286 w_k1_KO_s116
                                 + 0.0231672371998042 w_k1_KO_s117
                                 + 0.0675585571265479 w_k1_MAT_s000
                                 + 0.0141145385466973 w_k1_MAT_s001
                                 + 0.0286886929228057 w_k1_MAT_s002
                                 + 0.0347701215441664 w_k1_MAT_s003
                                 + 0.0700082070763561 w_k1_MAT_s004
                                 + 0.0295455083386867 w_k1_MAT_s005
                                 + 0.0420019886231465 w_k1_MAT_s006
                                 + 0.0232104359450843 w_k1_MAT_s007
                                 + 0.0394241628989712 w_k1_MAT_s010
                                 + 0.0241746828152372 w_k1_MAT_s011
                                 + 0.0330601985587645 w_k1_MAT_s012
                                 + 0.0156364052231732 w_k1_MAT_s013
                                 + 0.0153801420681483 w_k1_MAT_s014
                                 + 0.0512718601115343 w_k1_MAT_s015
                                 + 0.0513736073966421 w_k1_MAT_s016
                                 + 0.0904016660100325 w_k1_MAT_s017
                                 + 0.0184150303443154 w_k1_MAT_s100
                                 + 0.0412570811192835 w_k1_MAT_s101
                                 + 0.0651914483693878 w_k1_MAT_s102
                                 + 0.0256600485708185 w_k1_MAT_s103
                                 + 0.0142055702259445 w_k1_MAT_s104
                                 + 0.0738990367445574 w_k1_MAT_s105
                                 + 0.0487703794554277 w_k1_MAT_s106
                                 + 0.0160933845609127 w_k1_MAT_s107
                                 + 0.0142009738985037 w_k1_MAT_s110
                                 + 0.00587669050932125 w_k1_MAT_s111
                                 + 0.0246531084689442 w_k1_MAT_s112
                                 + 0.00625102096176043 w_k1_MAT_s113
                                 + 0.0210576272012657 w_k1_MAT_s114
                                 + 0.0104233840109918 w_k1_MAT_s115
                                 + 0.00670780585983895 w_k1_MAT_s116
                                 + 0.0274481052516735 w_k1_MAT_s117
                                 + 0.0508741909453795 w_k1_MSFT_s000
                                 + 0.0170318730291884 w_k1_MSFT_s001
                                 + 0.0246282050445722 w_k1_MSFT_s002
                                 + 0.0346336880737251 w_k1_MSFT_s003
                                 + 0.0697080311373772 w_k1_MSFT_s004
                                 + 0.032490443544052 w_k1_MSFT_s005
                                 + 0.0379159146644851 w_k1_MSFT_s006
                                 + 0.0225741403645605 w_k1_MSFT_s007
                                 + 0.0378618346442306 w_k1_MSFT_s010
                                 + 0.0241894915699972 w_k1_MSFT_s011
                                 + 0.0340791462876695 w_k1_MSFT_s012
                                 + 0.0175246161180725 w_k1_MSFT_s013
                                 + 0.0147183354066728 w_k1_MSFT_s014
                                 + 0.0481210228850659 w_k1_MSFT_s015
                                 + 0.0619210723177355 w_k1_MSFT_s016
                                 + 0.0698202196094792 w_k1_MSFT_s017
                                 + 0.0152070244237367 w_k1_MSFT_s100
                                 + 0.0514166869651438 w_k1_MSFT_s101
                                 + 0.06595432437631 w_k1_MSFT_s102
                                 + 0.0203994653738989 w_k1_MSFT_s103
                                 + 0.0158637651036869 w_k1_MSFT_s104
                                 + 0.0650753034417493 w_k1_MSFT_s105
                                 + 0.0483391166770287 w_k1_MSFT_s106
                                 + 0.0163793270512409 w_k1_MSFT_s107
                                 + 0.01169611618387 w_k1_MSFT_s110
                                 + 0.0081141714692612 w_k1_MSFT_s111
                                 + 0.0221386148245901 w_k1_MSFT_s112
                                 + 0.00607307676412487 w_k1_MSFT_s113
                                 + 0.0183250490155796 w_k1_MSFT_s114
                                 + 0.00897039756710388 w_k1_MSFT_s115
                                 + 0.0072133738794962 w_k1_MSFT_s116
                                 + 0.0244982668309953 w_k1_MSFT_s117
                                 + 0.0535408149685593 w_k1_PG_s000
                                 + 0.0179298749031673 w_k1_PG_s001
                                 + 0.0251851692342211 w_k1_PG_s002
                                 + 0.0347645672703412 w_k1_PG_s003
                                 + 0.0662118984478895 w_k1_PG_s004
                                 + 0.0336425284464516 w_k1_PG_s005
                                 + 0.0386442842554367 w_k1_PG_s006
                                 + 0.026169949824058 w_k1_PG_s007
                                 + 0.0369538173537388 w_k1_PG_s010
                                 + 0.021304249957814 w_k1_PG_s011
                                 + 0.0372878121093032 w_k1_PG_s012
                                 + 0.019091075510443 w_k1_PG_s013
                                 + 0.0148469380054813 w_k1_PG_s014
                                 + 0.0527149770045403 w_k1_PG_s015
                                 + 0.0539514377081647 w_k1_PG_s016
                                 + 0.0725300397666896 w_k1_PG_s017
                                 + 0.0150970628365667 w_k1_PG_s100
                                 + 0.0466467007661839 w_k1_PG_s101
                                 + 0.0608378838510192 w_k1_PG_s102
                                 + 0.0221880434014889 w_k1_PG_s103
                                 + 0.0157013480892904 w_k1_PG_s104
                                 + 0.0682310651863969 w_k1_PG_s105
                                 + 0.0529968328314906 w_k1_PG_s106
                                 + 0.0144100019906413 w_k1_PG_s107
                                 + 0.0120657231924172 w_k1_PG_s110
                                 + 0.00719485772094496 w_k1_PG_s111
                                 + 0.0228255706418647 w_k1_PG_s112
                                 + 0.00618281647394206 w_k1_PG_s113
                                 + 0.0191797369113211 w_k1_PG_s114
                                 + 0.00810493440688936 w_k1_PG_s115
                                 + 0.00633452060149043 w_k1_PG_s116
                                 + 0.0261784365020553 w_k1_PG_s117 >= 1.09
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
