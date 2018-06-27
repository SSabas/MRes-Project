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
 w_k1_ADM_s0_return_constraint:  1.17319 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.950753 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.983531 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.12141 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.983923 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.924313 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.69563 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.17214 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.00823 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.24474 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.888049 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.756238 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.988576 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.831155 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.05489 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.44655 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.1853 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.942613 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.10385 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.08288 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.01077 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.00882 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.0667 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.15393 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.49763 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.823261 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.11112 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.904093 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.743131 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.933086 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.860056 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.03005 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.931081 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.775295 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.944998 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.854252 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.19805 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.06959 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.32174 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.960656 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.882278 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.20734 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.939134 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.921185 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.668764 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.29742 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.916055 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.25681 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.988702 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.599987 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.909884 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.921036 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.729687 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.66332 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.40621 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.03717 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.912454 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.16183 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.889777 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.879726 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.06152 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.22834 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.72523 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.747861 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.03355 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.865194 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.674806 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.914269 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.68884 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.52727 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.0207 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.804841 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.1082 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.696043 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 2.04168 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.990993 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.35998 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.879998 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.858246 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.32554 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.11412 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.966008 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.650579 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.37848 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.769355 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.25878 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.954999 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.553009 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.05141 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.79445 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.692487 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 2.0232 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.20279 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.86771 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.00786 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.47538 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.653152 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.979954 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.02794 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.17454 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.82206 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.768268 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.08222 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.73313 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.510102 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.754568 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.63378 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.54319 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.989781 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.769145 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.05357 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.612869 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 2.27238 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.18191 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.21885 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.924367 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.899621 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.26617 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.08451 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.907327 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.779174 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.20804 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.07536 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.37769 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.00659 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.789533 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.07449 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.811283 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.795165 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.36671 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.14908 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.859243 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.9992 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.12895 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.876052 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.762233 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.939274 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.18417 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.4469 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.809518 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.01224 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.876691 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.627846 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.764145 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.779389 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.33526 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.895769 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.758506 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.07949 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.841098 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.53015 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.981533 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.18921 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.959617 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.926528 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.15619 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.875439 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.27228 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.84182 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.55249 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.53707 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.873555 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.899559 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.542729 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.680123 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.3273 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.71937 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.69002 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.02217 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.09839 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.662581 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.20879 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.503319 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.429 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.08019 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.883366 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.5087 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.857402 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.33443 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.656095 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.749634 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.37011 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.711109 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.22248 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.50188 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.03463 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.851291 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.546999 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 2.03653 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.29319 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.13504 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.944561 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.97711 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.24319 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.13594 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.975817 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.756644 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.15775 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.01675 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.21714 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.989576 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.713108 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.08418 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.03006 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.757654 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.35093 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.02567 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.855776 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.964923 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.13151 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.860632 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.825457 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.822296 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.01938 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.24085 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.790807 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.02646 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.821041 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.688672 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.836348 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.742182 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.22839 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.894389 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.705182 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.06313 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.851163 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.46567 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.02729 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.27179 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.938458 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.929349 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.09531 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.947336 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.10033 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.680979 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.32093 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.648034 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.05774 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.822669 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.542291 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.761499 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.04156 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.801483 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.64767 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.05718 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.05525 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.829084 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.26521 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.639927 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.27014 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.01767 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.15929 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.46129 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.897357 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.2562 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.734229 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.665675 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.07644 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.692496 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.31547 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.07741 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.890161 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.972489 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.645899 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.68215 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.21508 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.24621 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.871005 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.945344 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.12369 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.920996 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.28501 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.731933 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.44232 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.446643 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.995528 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.00315 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.440899 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.737283 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.20151 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.774871 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 2.11657 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.40613 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.45646 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.667929 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.00171 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.709297 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.40663 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.18782 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.01348 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.66238 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.825731 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.19713 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.648575 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.872084 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.35722 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.735945 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.58355 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.69309 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.03923 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.857717 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.469239 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 2.19931 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.23947 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.15121 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.934647 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.921405 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.22316 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.04626 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.01253 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.752011 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.20523 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.993057 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.30484 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.03165 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.751989 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.10808 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.910517 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.829312 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.52366 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.12328 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.934154 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.946781 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.17325 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.827221 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.915094 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.940825 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.19236 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.36189 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.925293 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.0989 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.875587 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.689573 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.852306 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.745789 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.20722 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.936422 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.710687 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.959963 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.787819 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.44578 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.02652 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.14615 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.961699 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.94482 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.16884 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.995974 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.01166 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.802662 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.26376 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.848402 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.06954 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.972403 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.772085 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.955588 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.949311 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.833346 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.25955 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.05832 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.897445 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.866224 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.13825 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.805907 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.955334 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.05131 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.11299 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.32833 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.843676 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.10384 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.845747 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.710781 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.984054 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.82822 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.24557 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.05265 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.881731 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.960458 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.743967 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.39039 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.12687 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.081999979645528 z_k1_s0000
                                 - 0.119999835213049 z_k1_s0001
                                 - 0.27599928889021 z_k1_s0002
                                 - 0.133000435435759 z_k1_s0003
                                 - 0.166999706096531 z_k1_s0004
                                 - 0.09099999496149 z_k1_s0005
                                 - 0.113999898802363 z_k1_s0006
                                 - 0.123999423820393 z_k1_s0007
                                 - 0.280999804240547 z_k1_s0010
                                 - 0.510001172109118 z_k1_s0011
                                 - 1.15999879475814 z_k1_s0012
                                 - 0.300000058299506 z_k1_s0013
                                 - 0.312000060631486 z_k1_s0014
                                 - 0.577999325457079 z_k1_s0015
                                 - 0.757001470475701 z_k1_s0016
                                 - 0.394000005033354 z_k1_s0017
                                 - 0.104000110373023 z_k1_s0020
                                 - 0.126999862214816 z_k1_s0021
                                 - 0.0929997435350776 z_k1_s0022
                                 - 0.257000214931346 z_k1_s0023
                                 - 0.0760001136957095 z_k1_s0024
                                 - 0.0530001041536159 z_k1_s0025
                                 - 0.0620000294569523 z_k1_s0026
                                 - 0.0870004805336553 z_k1_s0027
                                 - 0.146000093154119 z_k1_s0030
                                 - 0.282999822326056 z_k1_s0031
                                 - 0.092000032261494 z_k1_s0032
                                 - 0.25200015549889 z_k1_s0033
                                 - 0.327999712236542 z_k1_s0034
                                 - 0.259000267040549 z_k1_s0035
                                 - 0.134000122511323 z_k1_s0036
                                 - 0.436000606021963 z_k1_s0037
                                 - 0.407001679287898 z_k1_s0040
                                 - 0.309000687329216 z_k1_s0041
                                 - 0.993003456104104 z_k1_s0042
                                 - 0.204000499825204 z_k1_s0043
                                 - 0.259000673336977 z_k1_s0044
                                 - 0.662002304069403 z_k1_s0045
                                 - 0.682000570342883 z_k1_s0046
                                 - 0.437001055201999 z_k1_s0047
                                 - 0.225000132072541 z_k1_s0050
                                 - 0.113000134065365 z_k1_s0051
                                 - 0.294000196481469 z_k1_s0052
                                 - 0.205000996911114 z_k1_s0053
                                 - 0.14200013515026 z_k1_s0054
                                 - 0.209999884200881 z_k1_s0055
                                 - 0.446000078512561 z_k1_s0056
                                 - 0.158000160479873 z_k1_s0057
                                 - 0.284000385074661 z_k1_s0060
                                 - 0.137000047715557 z_k1_s0061
                                 - 0.207999549984065 z_k1_s0062
                                 - 0.0840000867051741 z_k1_s0063
                                 - 0.27100069163823 z_k1_s0064
                                 - 0.194999856547635 z_k1_s0065
                                 - 0.0860000852338678 z_k1_s0066
                                 - 0.219999838156306 z_k1_s0067
                                 - 0.549000918315696 z_k1_s0070
                                 - 0.84899771862093 z_k1_s0071
                                 - 0.317999672642244 z_k1_s0072
                                 - 0.21699972843325 z_k1_s0073
                                 - 0.741000644590291 z_k1_s0074
                                 - 0.268000071524762 z_k1_s0075
                                 - 0.646998636602599 z_k1_s0076
                                 - 0.44299968343552 z_k1_s0077
                                 - 0.300999540492669 z_k1_s0100
                                 - 0.1269998413982 z_k1_s0101
                                 - 0.0939998532923984 z_k1_s0102
                                 - 0.147999645783901 z_k1_s0103
                                 - 0.0969998647474585 z_k1_s0104
                                 - 0.0969998647474585 z_k1_s0105
                                 - 0.287999720687175 z_k1_s0106
                                 - 0.226999717600719 z_k1_s0107
                                 - 0.134999710111607 z_k1_s0110
                                 - 0.119999656432604 z_k1_s0111
                                 - 0.0489999692182307 z_k1_s0112
                                 - 0.0399999370108291 z_k1_s0113
                                 - 0.173999849677013 z_k1_s0114
                                 - 0.0739999298499989 z_k1_s0115
                                 - 0.0439999770918771 z_k1_s0116
                                 - 0.137000232601751 z_k1_s0117
                                 - 0.454998755918882 z_k1_s0120
                                 - 0.165999584219762 z_k1_s0121
                                 - 0.121999672438674 z_k1_s0122
                                 - 0.196999140475558 z_k1_s0123
                                 - 0.110999694493402 z_k1_s0124
                                 - 0.214999071204266 z_k1_s0125
                                 - 0.0979997039672128 z_k1_s0126
                                 - 0.460999341160288 z_k1_s0127
                                 - 0.132999816880809 z_k1_s0130
                                 - 0.188999461683612 z_k1_s0131
                                 - 0.291000356908168 z_k1_s0132
                                 - 0.149999783885135 z_k1_s0133
                                 - 0.199998861581271 z_k1_s0134
                                 - 0.46100002695143 z_k1_s0135
                                 - 0.228999598399399 z_k1_s0136
                                 - 0.168999757725251 z_k1_s0137
                                 - 0.256999768413506 z_k1_s0140
                                 - 0.429999738862926 z_k1_s0141
                                 - 0.269999214170426 z_k1_s0142
                                 - 0.340999566377592 z_k1_s0143
                                 - 0.166999903023504 z_k1_s0144
                                 - 0.099999850433336 z_k1_s0145
                                 - 0.119999820520003 z_k1_s0146
                                 - 0.224999663475006 z_k1_s0147
                                 - 0.184999825279937 z_k1_s0150
                                 - 0.100999720396686 z_k1_s0151
                                 - 0.2879991830322 z_k1_s0152
                                 - 0.235998955819298 z_k1_s0153
                                 - 0.0669997793721759 z_k1_s0154
                                 - 0.0989996570425251 z_k1_s0155
                                 - 0.274999126228974 z_k1_s0156
                                 - 0.16899931844665 z_k1_s0157
                                 - 0.0820000311395359 z_k1_s0160
                                 - 0.10899991122208 z_k1_s0161
                                 - 0.0909999911670506 z_k1_s0162
                                 - 0.0369999938019879 z_k1_s0163
                                 - 0.0969999645187271 z_k1_s0164
                                 - 0.0790000444636976 z_k1_s0165
                                 - 0.0539999973650627 z_k1_s0166
                                 - 0.0440000022456062 z_k1_s0167
                                 - 0.273999498455896 z_k1_s0170
                                 - 0.363000511020837 z_k1_s0171
                                 - 0.157999379518138 z_k1_s0172
                                 - 0.0970000212082444 z_k1_s0173
                                 - 0.185999862572389 z_k1_s0174
                                 - 0.194000628016887 z_k1_s0175
                                 - 0.114000147176783 z_k1_s0176
                                 - 0.0780000698788304 z_k1_s0177
                                 - 0.350000585310215 z_k1_s1000
                                 - 0.371001233228593 z_k1_s1001
                                 - 0.285999777939474 z_k1_s1002
                                 - 0.36499885953847 z_k1_s1003
                                 - 0.223999761798774 z_k1_s1004
                                 - 0.518998504275201 z_k1_s1005
                                 - 0.276999894202875 z_k1_s1006
                                 - 0.671999285396321 z_k1_s1007
                                 - 0.756998334779326 z_k1_s1010
                                 - 0.936993295009189 z_k1_s1011
                                 - 0.957994302857272 z_k1_s1012
                                 - 2.10599164502095 z_k1_s1013
                                 - 0.884997790602337 z_k1_s1014
                                 - 1.27199721394753 z_k1_s1015
                                 - 1.0169940947709 z_k1_s1016
                                 - 1.18699921388154 z_k1_s1017
                                 - 1.303998920975 z_k1_s1020
                                 - 1.27299853542313 z_k1_s1021
                                 - 0.48899928361597 z_k1_s1022
                                 - 1.27099710264716 z_k1_s1023
                                 - 0.686996476598239 z_k1_s1024
                                 - 0.472999667895005 z_k1_s1025
                                 - 0.295999437580566 z_k1_s1026
                                 - 0.441999282343143 z_k1_s1027
                                 - 0.311999907794585 z_k1_s1030
                                 - 0.538999867321399 z_k1_s1031
                                 - 0.591999197559308 z_k1_s1032
                                 - 0.835998010540315 z_k1_s1033
                                 - 0.439001213476242 z_k1_s1034
                                 - 0.932998846068915 z_k1_s1035
                                 - 0.378999836170374 z_k1_s1036
                                 - 0.339999793071082 z_k1_s1037
                                 - 0.437998600974063 z_k1_s1040
                                 - 1.05500052870619 z_k1_s1041
                                 - 1.91499295117891 z_k1_s1042
                                 - 0.612998247869571 z_k1_s1043
                                 - 1.30899528558365 z_k1_s1044
                                 - 0.767997955405541 z_k1_s1045
                                 - 0.518998745879203 z_k1_s1046
                                 - 0.482998997031675 z_k1_s1047
                                 - 1.21199579922405 z_k1_s1050
                                 - 0.78999573423462 z_k1_s1051
                                 - 1.22399348451008 z_k1_s1052
                                 - 0.594998540237458 z_k1_s1053
                                 - 0.326999134264131 z_k1_s1054
                                 - 0.708998027792535 z_k1_s1055
                                 - 1.52399486448923 z_k1_s1056
                                 - 1.02499484029067 z_k1_s1057
                                 - 0.259000071872858 z_k1_s1060
                                 - 0.555998867934113 z_k1_s1061
                                 - 0.411000042226046 z_k1_s1062
                                 - 0.466999474987791 z_k1_s1063
                                 - 0.538998081281516 z_k1_s1064
                                 - 0.854998776236769 z_k1_s1065
                                 - 0.330000024645731 z_k1_s1066
                                 - 0.810997983724067 z_k1_s1067
                                 - 0.170000038083026 z_k1_s1070
                                 - 0.139999843091989 z_k1_s1071
                                 - 0.615999506564663 z_k1_s1072
                                 - 0.287000133808371 z_k1_s1073
                                 - 0.373999788342791 z_k1_s1074
                                 - 0.525998675391664 z_k1_s1075
                                 - 0.204999814206107 z_k1_s1076
                                 - 0.143999852677468 z_k1_s1077
                                 - 0.372001198608628 z_k1_s1100
                                 - 0.441001098686801 z_k1_s1101
                                 - 0.308000638879332 z_k1_s1102
                                 - 1.07400337076016 z_k1_s1103
                                 - 0.422001177514685 z_k1_s1104
                                 - 1.19100533448963 z_k1_s1105
                                 - 0.737999547029027 z_k1_s1106
                                 - 0.512002454629046 z_k1_s1107
                                 - 0.484998969388194 z_k1_s1110
                                 - 0.360000788742674 z_k1_s1111
                                 - 0.923001896525559 z_k1_s1112
                                 - 0.313000547083781 z_k1_s1113
                                 - 0.279000739590819 z_k1_s1114
                                 - 0.612003067285861 z_k1_s1115
                                 - 1.09300326699485 z_k1_s1116
                                 - 0.601000618156845 z_k1_s1117
                                 - 0.175999984371202 z_k1_s1120
                                 - 0.150999399545399 z_k1_s1121
                                 - 0.203999720975556 z_k1_s1122
                                 - 0.173999593184933 z_k1_s1123
                                 - 0.103000056080912 z_k1_s1124
                                 - 0.430000874998364 z_k1_s1125
                                 - 0.098000082615837 z_k1_s1126
                                 - 0.098999991208801 z_k1_s1127
                                 - 0.684003332049094 z_k1_s1130
                                 - 0.40500105128937 z_k1_s1131
                                 - 0.154000562720414 z_k1_s1132
                                 - 0.273000030155917 z_k1_s1133
                                 - 0.231000709380203 z_k1_s1134
                                 - 0.144000403725202 z_k1_s1135
                                 - 0.649002775565854 z_k1_s1136
                                 - 0.154000562720414 z_k1_s1137
                                 - 1.08800151004554 z_k1_s1140
                                 - 0.562000179784227 z_k1_s1141
                                 - 0.425000022986231 z_k1_s1142
                                 - 0.392000170589734 z_k1_s1143
                                 - 0.548000627191982 z_k1_s1144
                                 - 0.737999090232046 z_k1_s1145
                                 - 0.278000185765204 z_k1_s1146
                                 - 0.504002033330642 z_k1_s1147
                                 - 0.301000631332118 z_k1_s1150
                                 - 0.0910000446351169 z_k1_s1151
                                 - 0.256000505611332 z_k1_s1152
                                 - 0.150000104668911 z_k1_s1153
                                 - 0.0890001438476515 z_k1_s1154
                                 - 0.280000258262043 z_k1_s1155
                                 - 0.301000631332118 z_k1_s1156
                                 - 0.10400018575458 z_k1_s1157
                                 - 0.0870000664884463 z_k1_s1160
                                 - 0.135000324120206 z_k1_s1161
                                 - 0.138999772797326 z_k1_s1162
                                 - 0.0720000951973718 z_k1_s1163
                                 - 0.183000698252073 z_k1_s1164
                                 - 0.0840000722302314 z_k1_s1165
                                 - 0.127999832683907 z_k1_s1166
                                 - 0.336999859839938 z_k1_s1167
                                 - 0.209000163134282 z_k1_s1170
                                 - 0.24000038069514 z_k1_s1171
                                 - 0.94500189780017 z_k1_s1172
                                 - 0.59200023004702 z_k1_s1173
                                 - 0.264000206064356 z_k1_s1174
                                 - 0.32400040758829 z_k1_s1175
                                 - 0.991000464139631 z_k1_s1176
                                 - 0.689000073723371 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.54625 w_k1_ADM_s000
                                 + 1.43267 w_k1_AXP_s000
                                 + 1.39033 w_k1_DIS_s000
                                 + 1.21972 w_k1_FDX_s000
                                 + 1.64148 w_k1_IBM_s000
                                 + 1.38834 w_k1_INTC_s000
                                 + 1.64638 w_k1_KO_s000 + 1.14217 w_k1_MAT_s000
                                 + 1.39252 w_k1_MSFT_s000
                                 + 1.24302 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.925126 w_k1_ADM_s000
                                 + 0.9637 w_k1_AXP_s000 + 1.18235 w_k1_DIS_s000
                                 + 0.92879 w_k1_FDX_s000
                                 + 1.04104 w_k1_IBM_s000
                                 + 1.12667 w_k1_INTC_s000
                                 + 1.18292 w_k1_KO_s000
                                 + 0.940797 w_k1_MAT_s000
                                 + 1.13913 w_k1_MSFT_s000
                                 + 1.10394 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.923342 w_k1_ADM_s000
                                 + 0.74242 w_k1_AXP_s000
                                 + 0.843052 w_k1_DIS_s000
                                 + 0.831474 w_k1_FDX_s000
                                 + 0.741991 w_k1_IBM_s000
                                 + 1.15422 w_k1_INTC_s000
                                 + 0.879432 w_k1_KO_s000
                                 + 0.581891 w_k1_MAT_s000
                                 + 1.01891 w_k1_MSFT_s000
                                 + 0.835839 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.03561 w_k1_ADM_s000
                                 + 0.904541 w_k1_AXP_s000
                                 + 0.655739 w_k1_DIS_s000
                                 + 0.763927 w_k1_FDX_s000
                                 + 0.995659 w_k1_IBM_s000
                                 + 0.909005 w_k1_INTC_s000
                                 + 0.904024 w_k1_KO_s000
                                 + 0.856473 w_k1_MAT_s000
                                 + 0.792639 w_k1_MSFT_s000
                                 + 0.879371 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.54943 w_k1_ADM_s000
                                 + 1.37846 w_k1_AXP_s000
                                 + 1.52246 w_k1_DIS_s000
                                 + 1.33768 w_k1_FDX_s000
                                 + 1.09851 w_k1_IBM_s000
                                 + 1.14696 w_k1_INTC_s000
                                 + 1.31586 w_k1_KO_s000
                                 + 0.943032 w_k1_MAT_s000
                                 + 1.25788 w_k1_MSFT_s000
                                 + 1.15118 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.79554 w_k1_ADM_s000
                                 + 0.846823 w_k1_AXP_s000
                                 + 0.918988 w_k1_DIS_s000
                                 + 0.840003 w_k1_FDX_s000
                                 + 1.38111 w_k1_IBM_s000
                                 + 1.01826 w_k1_INTC_s000
                                 + 1.37826 w_k1_KO_s000 + 1.22518 w_k1_MAT_s000
                                 + 0.939125 w_k1_MSFT_s000
                                 + 1.01167 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.20967 w_k1_ADM_s000
                                 + 1.46112 w_k1_AXP_s000
                                 + 1.01454 w_k1_DIS_s000
                                 + 1.07417 w_k1_FDX_s000
                                 + 1.17588 w_k1_IBM_s000
                                 + 1.16802 w_k1_INTC_s000
                                 + 1.19666 w_k1_KO_s000 + 1.3908 w_k1_MAT_s000
                                 + 0.983061 w_k1_MSFT_s000
                                 + 1.18681 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.13029 w_k1_ADM_s000
                                 + 1.05562 w_k1_AXP_s000
                                 + 1.03902 w_k1_DIS_s000
                                 + 1.12354 w_k1_FDX_s000
                                 + 0.771296 w_k1_IBM_s000
                                 + 0.936642 w_k1_INTC_s000
                                 + 0.956561 w_k1_KO_s000
                                 + 1.18423 w_k1_MAT_s000
                                 + 1.2378 w_k1_MSFT_s000 + 1.1599 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.00274 w_k1_ADM_s001
                                 + 0.976046 w_k1_AXP_s001
                                 + 1.67566 w_k1_DIS_s001
                                 + 1.19881 w_k1_FDX_s001
                                 + 0.745963 w_k1_IBM_s001
                                 + 1.27813 w_k1_INTC_s001
                                 + 1.05386 w_k1_KO_s001
                                 + 0.754783 w_k1_MAT_s001
                                 + 1.23219 w_k1_MSFT_s001 + 1.0198 w_k1_PG_s001
                                 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.678568 w_k1_ADM_s001
                                 + 0.987824 w_k1_AXP_s001
                                 + 0.82576 w_k1_DIS_s001
                                 + 0.824317 w_k1_FDX_s001
                                 + 0.84205 w_k1_IBM_s001
                                 + 0.811793 w_k1_INTC_s001
                                 + 0.736868 w_k1_KO_s001
                                 + 1.08524 w_k1_MAT_s001
                                 + 0.7659 w_k1_MSFT_s001
                                 + 0.882997 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.951523 w_k1_ADM_s001
                                 + 0.996749 w_k1_AXP_s001
                                 + 1.01775 w_k1_DIS_s001
                                 + 0.933497 w_k1_FDX_s001
                                 + 0.97526 w_k1_IBM_s001
                                 + 0.850748 w_k1_INTC_s001
                                 + 0.858917 w_k1_KO_s001
                                 + 0.856597 w_k1_MAT_s001
                                 + 0.926814 w_k1_MSFT_s001
                                 + 0.941584 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.692799 w_k1_ADM_s001
                                 + 0.665342 w_k1_AXP_s001
                                 + 0.709676 w_k1_DIS_s001
                                 + 0.717691 w_k1_FDX_s001
                                 + 0.597981 w_k1_IBM_s001
                                 + 0.758966 w_k1_INTC_s001
                                 + 0.729766 w_k1_KO_s001
                                 + 0.758482 w_k1_MAT_s001
                                 + 0.832292 w_k1_MSFT_s001
                                 + 0.752943 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.998298 w_k1_ADM_s001
                                 + 1.33672 w_k1_AXP_s001
                                 + 1.37044 w_k1_DIS_s001
                                 + 1.00859 w_k1_FDX_s001
                                 + 2.06589 w_k1_IBM_s001
                                 + 1.13264 w_k1_INTC_s001 + 1.3459 w_k1_KO_s001
                                 + 2.10069 w_k1_MAT_s001
                                 + 1.05743 w_k1_MSFT_s001
                                 + 1.02725 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.773344 w_k1_ADM_s001
                                 + 1.16383 w_k1_AXP_s001
                                 + 1.32829 w_k1_DIS_s001
                                 + 1.02144 w_k1_FDX_s001
                                 + 1.32372 w_k1_IBM_s001
                                 + 0.982644 w_k1_INTC_s001
                                 + 1.33099 w_k1_KO_s001 + 1.62729 w_k1_MAT_s001
                                 + 1.18427 w_k1_MSFT_s001
                                 + 1.02286 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.10773 w_k1_ADM_s001
                                 + 0.818197 w_k1_AXP_s001
                                 + 1.00065 w_k1_DIS_s001
                                 + 1.07468 w_k1_FDX_s001
                                 + 0.623761 w_k1_IBM_s001
                                 + 1.01612 w_k1_INTC_s001
                                 + 0.79885 w_k1_KO_s001 + 0.61836 w_k1_MAT_s001
                                 + 0.97198 w_k1_MSFT_s001
                                 + 0.891289 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.06605 w_k1_ADM_s001
                                 + 1.51934 w_k1_AXP_s001
                                 + 1.65499 w_k1_DIS_s001
                                 + 1.40743 w_k1_FDX_s001
                                 + 1.12813 w_k1_IBM_s001
                                 + 1.3501 w_k1_INTC_s001 + 1.30565 w_k1_KO_s001
                                 + 1.65263 w_k1_MAT_s001
                                 + 1.41969 w_k1_MSFT_s001
                                 + 1.10313 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.03333 w_k1_ADM_s002
                                 + 0.846621 w_k1_AXP_s002
                                 + 0.772124 w_k1_DIS_s002
                                 + 1.00868 w_k1_FDX_s002
                                 + 0.708462 w_k1_IBM_s002
                                 + 0.92453 w_k1_INTC_s002
                                 + 0.873631 w_k1_KO_s002
                                 + 1.19118 w_k1_MAT_s002
                                 + 0.970133 w_k1_MSFT_s002
                                 + 0.95304 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.29031 w_k1_ADM_s002
                                 + 1.66614 w_k1_AXP_s002
                                 + 1.12862 w_k1_DIS_s002
                                 + 1.27081 w_k1_FDX_s002
                                 + 1.32997 w_k1_IBM_s002
                                 + 0.924764 w_k1_INTC_s002
                                 + 1.11291 w_k1_KO_s002 + 2.15793 w_k1_MAT_s002
                                 + 0.96615 w_k1_MSFT_s002
                                 + 1.16377 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.15821 w_k1_ADM_s002
                                 + 1.18155 w_k1_AXP_s002
                                 + 1.30109 w_k1_DIS_s002 + 1.088 w_k1_FDX_s002
                                 + 1.64005 w_k1_IBM_s002
                                 + 1.06179 w_k1_INTC_s002
                                 + 1.47289 w_k1_KO_s002 + 2.69664 w_k1_MAT_s002
                                 + 1.12731 w_k1_MSFT_s002
                                 + 1.30636 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.11519 w_k1_ADM_s002
                                 + 1.11763 w_k1_AXP_s002
                                 + 1.11803 w_k1_DIS_s002
                                 + 1.19668 w_k1_FDX_s002
                                 + 1.15627 w_k1_IBM_s002
                                 + 1.07437 w_k1_INTC_s002 + 1.1269 w_k1_KO_s002
                                 + 1.20684 w_k1_MAT_s002
                                 + 1.12305 w_k1_MSFT_s002
                                 + 1.14583 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.793656 w_k1_ADM_s002
                                 + 0.958604 w_k1_AXP_s002
                                 + 0.797598 w_k1_DIS_s002
                                 + 0.809431 w_k1_FDX_s002
                                 + 1.10482 w_k1_IBM_s002
                                 + 0.950742 w_k1_INTC_s002
                                 + 1.00619 w_k1_KO_s002 + 1.25233 w_k1_MAT_s002
                                 + 0.870991 w_k1_MSFT_s002
                                 + 1.01302 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.708488 w_k1_ADM_s002
                                 + 0.688973 w_k1_AXP_s002
                                 + 0.65149 w_k1_DIS_s002
                                 + 0.725322 w_k1_FDX_s002
                                 + 0.965475 w_k1_IBM_s002
                                 + 0.872011 w_k1_INTC_s002
                                 + 0.847939 w_k1_KO_s002
                                 + 0.935829 w_k1_MAT_s002
                                 + 0.623706 w_k1_MSFT_s002
                                 + 0.909425 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.895648 w_k1_ADM_s002
                                 + 0.700002 w_k1_AXP_s002
                                 + 0.778665 w_k1_DIS_s002
                                 + 0.861577 w_k1_FDX_s002
                                 + 0.667582 w_k1_IBM_s002
                                 + 0.870448 w_k1_INTC_s002
                                 + 0.848422 w_k1_KO_s002
                                 + 0.674263 w_k1_MAT_s002
                                 + 0.946981 w_k1_MSFT_s002
                                 + 0.903701 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.880121 w_k1_ADM_s002
                                 + 0.8397 w_k1_AXP_s002
                                 + 0.960904 w_k1_DIS_s002
                                 + 0.912328 w_k1_FDX_s002
                                 + 0.991107 w_k1_IBM_s002
                                 + 0.852378 w_k1_INTC_s002
                                 + 0.926791 w_k1_KO_s002
                                 + 1.17643 w_k1_MAT_s002
                                 + 0.981696 w_k1_MSFT_s002
                                 + 0.89887 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.28315 w_k1_ADM_s003
                                 + 1.55008 w_k1_AXP_s003
                                 + 1.39629 w_k1_DIS_s003
                                 + 1.14658 w_k1_FDX_s003
                                 + 1.60607 w_k1_IBM_s003
                                 + 1.10862 w_k1_INTC_s003
                                 + 1.42374 w_k1_KO_s003 + 1.73987 w_k1_MAT_s003
                                 + 1.05803 w_k1_MSFT_s003 + 1.3033 w_k1_PG_s003
                                 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.06674 w_k1_ADM_s003
                                 + 0.99275 w_k1_AXP_s003
                                 + 1.05958 w_k1_DIS_s003
                                 + 0.928562 w_k1_FDX_s003
                                 + 1.37922 w_k1_IBM_s003
                                 + 0.7059 w_k1_INTC_s003 + 1.01435 w_k1_KO_s003
                                 + 1.18651 w_k1_MAT_s003
                                 + 0.879304 w_k1_MSFT_s003
                                 + 1.00693 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.687772 w_k1_ADM_s003
                                 + 0.764356 w_k1_AXP_s003
                                 + 0.78598 w_k1_DIS_s003
                                 + 0.758405 w_k1_FDX_s003
                                 + 0.977899 w_k1_IBM_s003
                                 + 0.782947 w_k1_INTC_s003
                                 + 0.741104 w_k1_KO_s003
                                 + 0.84448 w_k1_MAT_s003
                                 + 0.783885 w_k1_MSFT_s003
                                 + 0.840304 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.718765 w_k1_ADM_s003
                                 + 0.752041 w_k1_AXP_s003
                                 + 0.940329 w_k1_DIS_s003
                                 + 0.836354 w_k1_FDX_s003
                                 + 1.07391 w_k1_IBM_s003
                                 + 0.927696 w_k1_INTC_s003
                                 + 0.858314 w_k1_KO_s003
                                 + 1.04361 w_k1_MAT_s003
                                 + 0.962075 w_k1_MSFT_s003
                                 + 0.892157 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.14151 w_k1_ADM_s003
                                 + 1.38458 w_k1_AXP_s003 + 1.3138 w_k1_DIS_s003
                                 + 1.15267 w_k1_FDX_s003
                                 + 1.05145 w_k1_IBM_s003
                                 + 1.34291 w_k1_INTC_s003
                                 + 1.04422 w_k1_KO_s003
                                 + 0.853901 w_k1_MAT_s003
                                 + 1.1491 w_k1_MSFT_s003 + 1.17265 w_k1_PG_s003
                                 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.06842 w_k1_ADM_s003
                                 + 0.898387 w_k1_AXP_s003
                                 + 0.781682 w_k1_DIS_s003
                                 + 0.924441 w_k1_FDX_s003
                                 + 0.659279 w_k1_IBM_s003
                                 + 0.868579 w_k1_INTC_s003
                                 + 0.715493 w_k1_KO_s003
                                 + 0.665507 w_k1_MAT_s003
                                 + 0.885269 w_k1_MSFT_s003
                                 + 0.929964 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.766252 w_k1_ADM_s003
                                 + 0.641389 w_k1_AXP_s003
                                 + 0.634052 w_k1_DIS_s003
                                 + 0.658487 w_k1_FDX_s003
                                 + 0.805986 w_k1_IBM_s003
                                 + 0.630903 w_k1_INTC_s003
                                 + 0.698433 w_k1_KO_s003
                                 + 0.62581 w_k1_MAT_s003
                                 + 0.672292 w_k1_MSFT_s003
                                 + 0.807676 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.953288 w_k1_ADM_s003
                                 + 0.979117 w_k1_AXP_s003
                                 + 1.16576 w_k1_DIS_s003
                                 + 1.05806 w_k1_FDX_s003
                                 + 0.930459 w_k1_IBM_s003
                                 + 1.03545 w_k1_INTC_s003
                                 + 0.947382 w_k1_KO_s003
                                 + 0.988877 w_k1_MAT_s003
                                 + 1.04938 w_k1_MSFT_s003
                                 + 1.01015 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.816407 w_k1_ADM_s004
                                 + 0.578066 w_k1_AXP_s004
                                 + 0.687147 w_k1_DIS_s004
                                 + 0.753198 w_k1_FDX_s004
                                 + 0.816997 w_k1_IBM_s004
                                 + 0.878567 w_k1_INTC_s004
                                 + 0.812661 w_k1_KO_s004
                                 + 0.594385 w_k1_MAT_s004
                                 + 0.79669 w_k1_MSFT_s004
                                 + 0.818495 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.836969 w_k1_ADM_s004
                                 + 0.935777 w_k1_AXP_s004
                                 + 0.869232 w_k1_DIS_s004
                                 + 0.947994 w_k1_FDX_s004
                                 + 0.757984 w_k1_IBM_s004
                                 + 1.04977 w_k1_INTC_s004
                                 + 0.903613 w_k1_KO_s004
                                 + 0.750499 w_k1_MAT_s004
                                 + 1.00605 w_k1_MSFT_s004
                                 + 0.894083 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.12873 w_k1_ADM_s004
                                 + 1.06316 w_k1_AXP_s004
                                 + 0.891797 w_k1_DIS_s004
                                 + 0.966608 w_k1_FDX_s004
                                 + 1.11402 w_k1_IBM_s004
                                 + 1.07201 w_k1_INTC_s004
                                 + 1.07289 w_k1_KO_s004
                                 + 0.974564 w_k1_MAT_s004
                                 + 1.02002 w_k1_MSFT_s004
                                 + 0.965209 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.885103 w_k1_ADM_s004
                                 + 0.657101 w_k1_AXP_s004
                                 + 0.724425 w_k1_DIS_s004
                                 + 0.844204 w_k1_FDX_s004
                                 + 0.604619 w_k1_IBM_s004
                                 + 0.853507 w_k1_INTC_s004
                                 + 0.70103 w_k1_KO_s004
                                 + 0.530406 w_k1_MAT_s004
                                 + 0.800191 w_k1_MSFT_s004
                                 + 0.792619 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.01813 w_k1_ADM_s004
                                 + 0.962513 w_k1_AXP_s004
                                 + 0.753112 w_k1_DIS_s004
                                 + 0.939255 w_k1_FDX_s004
                                 + 0.843531 w_k1_IBM_s004
                                 + 1.01364 w_k1_INTC_s004
                                 + 0.909815 w_k1_KO_s004
                                 + 0.930525 w_k1_MAT_s004
                                 + 0.985555 w_k1_MSFT_s004
                                 + 0.938172 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.41089 w_k1_ADM_s004
                                 + 1.3252 w_k1_AXP_s004 + 1.53331 w_k1_DIS_s004
                                 + 1.18638 w_k1_FDX_s004
                                 + 1.54514 w_k1_IBM_s004
                                 + 0.871437 w_k1_INTC_s004
                                 + 1.45275 w_k1_KO_s004 + 1.574 w_k1_MAT_s004
                                 + 1.13962 w_k1_MSFT_s004
                                 + 1.19159 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.26416 w_k1_ADM_s004
                                 + 1.01192 w_k1_AXP_s004
                                 + 1.34113 w_k1_DIS_s004
                                 + 1.21253 w_k1_FDX_s004
                                 + 0.939022 w_k1_IBM_s004
                                 + 1.16869 w_k1_INTC_s004
                                 + 1.34499 w_k1_KO_s004
                                 + 0.690136 w_k1_MAT_s004
                                 + 1.1246 w_k1_MSFT_s004 + 1.20329 w_k1_PG_s004
                                 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.745724 w_k1_ADM_s004
                                 + 0.960636 w_k1_AXP_s004
                                 + 1.0355 w_k1_DIS_s004
                                 + 0.913392 w_k1_FDX_s004
                                 + 1.37738 w_k1_IBM_s004
                                 + 1.14501 w_k1_INTC_s004
                                 + 1.22488 w_k1_KO_s004 + 1.14472 w_k1_MAT_s004
                                 + 0.967581 w_k1_MSFT_s004
                                 + 1.03774 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.741201 w_k1_ADM_s005
                                 + 0.771077 w_k1_AXP_s005
                                 + 0.715327 w_k1_DIS_s005
                                 + 0.739458 w_k1_FDX_s005
                                 + 0.726012 w_k1_IBM_s005
                                 + 1.08729 w_k1_INTC_s005
                                 + 0.719919 w_k1_KO_s005
                                 + 0.685482 w_k1_MAT_s005
                                 + 0.883939 w_k1_MSFT_s005
                                 + 0.831296 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.89785 w_k1_ADM_s005
                                 + 0.659628 w_k1_AXP_s005
                                 + 0.79608 w_k1_DIS_s005
                                 + 0.716904 w_k1_FDX_s005
                                 + 1.03214 w_k1_IBM_s005
                                 + 0.882659 w_k1_INTC_s005
                                 + 1.05351 w_k1_KO_s005 + 1.08986 w_k1_MAT_s005
                                 + 0.943518 w_k1_MSFT_s005
                                 + 0.989917 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.1803 w_k1_ADM_s005
                                 + 1.2372 w_k1_AXP_s005 + 1.59857 w_k1_DIS_s005
                                 + 1.2949 w_k1_FDX_s005 + 1.02842 w_k1_IBM_s005
                                 + 1.16242 w_k1_INTC_s005
                                 + 1.27624 w_k1_KO_s005 + 1.42196 w_k1_MAT_s005
                                 + 1.33819 w_k1_MSFT_s005 + 1.1468 w_k1_PG_s005
                                 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.16165 w_k1_ADM_s005
                                 + 1.14619 w_k1_AXP_s005
                                 + 1.04323 w_k1_DIS_s005
                                 + 1.06975 w_k1_FDX_s005
                                 + 1.09021 w_k1_IBM_s005
                                 + 1.03685 w_k1_INTC_s005
                                 + 1.12502 w_k1_KO_s005 + 1.67175 w_k1_MAT_s005
                                 + 0.900092 w_k1_MSFT_s005
                                 + 1.03425 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.28211 w_k1_ADM_s005
                                 + 1.2945 w_k1_AXP_s005 + 1.06882 w_k1_DIS_s005
                                 + 0.902102 w_k1_FDX_s005
                                 + 1.56608 w_k1_IBM_s005
                                 + 0.990627 w_k1_INTC_s005
                                 + 1.31465 w_k1_KO_s005 + 2.11077 w_k1_MAT_s005
                                 + 1.01902 w_k1_MSFT_s005
                                 + 1.06173 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.19821 w_k1_ADM_s005
                                 + 1.60317 w_k1_AXP_s005
                                 + 1.57538 w_k1_DIS_s005
                                 + 1.27575 w_k1_FDX_s005
                                 + 1.48788 w_k1_IBM_s005
                                 + 1.37256 w_k1_INTC_s005 + 1.534 w_k1_KO_s005
                                 + 2.06353 w_k1_MAT_s005
                                 + 1.31154 w_k1_MSFT_s005
                                 + 1.15821 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.888835 w_k1_ADM_s005
                                 + 0.905202 w_k1_AXP_s005
                                 + 1.08254 w_k1_DIS_s005
                                 + 1.01079 w_k1_FDX_s005
                                 + 0.977855 w_k1_IBM_s005
                                 + 1.18464 w_k1_INTC_s005
                                 + 1.25599 w_k1_KO_s005 + 1.13995 w_k1_MAT_s005
                                 + 1.21866 w_k1_MSFT_s005
                                 + 1.00466 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.9053 w_k1_ADM_s005
                                 + 1.08438 w_k1_AXP_s005
                                 + 0.790364 w_k1_DIS_s005
                                 + 0.820222 w_k1_FDX_s005
                                 + 1.14535 w_k1_IBM_s005
                                 + 0.982244 w_k1_INTC_s005
                                 + 0.984956 w_k1_KO_s005
                                 + 1.45594 w_k1_MAT_s005
                                 + 0.802032 w_k1_MSFT_s005
                                 + 0.773192 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.941299 w_k1_ADM_s006
                                 + 0.852927 w_k1_AXP_s006
                                 + 1.07341 w_k1_DIS_s006
                                 + 0.978892 w_k1_FDX_s006
                                 + 1.01427 w_k1_IBM_s006
                                 + 0.965621 w_k1_INTC_s006
                                 + 1.08963 w_k1_KO_s006 + 1.03563 w_k1_MAT_s006
                                 + 1.03048 w_k1_MSFT_s006 + 1.0309 w_k1_PG_s006
                                 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.789411 w_k1_ADM_s006
                                 + 0.656746 w_k1_AXP_s006
                                 + 0.844278 w_k1_DIS_s006
                                 + 0.96279 w_k1_FDX_s006
                                 + 0.720902 w_k1_IBM_s006
                                 + 0.931606 w_k1_INTC_s006
                                 + 0.906254 w_k1_KO_s006
                                 + 0.711313 w_k1_MAT_s006
                                 + 1.04143 w_k1_MSFT_s006
                                 + 0.990358 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.27935 w_k1_ADM_s006
                                 + 1.23122 w_k1_AXP_s006
                                 + 1.37492 w_k1_DIS_s006
                                 + 1.24452 w_k1_FDX_s006
                                 + 1.28101 w_k1_IBM_s006
                                 + 1.08007 w_k1_INTC_s006
                                 + 1.38528 w_k1_KO_s006 + 1.44086 w_k1_MAT_s006
                                 + 1.10714 w_k1_MSFT_s006
                                 + 1.17977 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.953908 w_k1_ADM_s006
                                 + 0.864158 w_k1_AXP_s006
                                 + 1.43246 w_k1_DIS_s006
                                 + 1.15706 w_k1_FDX_s006
                                 + 0.900413 w_k1_IBM_s006
                                 + 0.920981 w_k1_INTC_s006
                                 + 1.15234 w_k1_KO_s006
                                 + 0.757781 w_k1_MAT_s006
                                 + 1.12345 w_k1_MSFT_s006
                                 + 1.03173 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.819014 w_k1_ADM_s006
                                 + 0.769801 w_k1_AXP_s006
                                 + 0.628043 w_k1_DIS_s006
                                 + 0.836754 w_k1_FDX_s006
                                 + 0.651221 w_k1_IBM_s006
                                 + 0.773491 w_k1_INTC_s006
                                 + 0.750793 w_k1_KO_s006
                                 + 0.586991 w_k1_MAT_s006
                                 + 0.856423 w_k1_MSFT_s006
                                 + 0.893901 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.05725 w_k1_ADM_s006
                                 + 1.19357 w_k1_AXP_s006
                                 + 0.978322 w_k1_DIS_s006
                                 + 1.14435 w_k1_FDX_s006
                                 + 0.796862 w_k1_IBM_s006
                                 + 1.09309 w_k1_INTC_s006
                                 + 1.08324 w_k1_KO_s006
                                 + 0.681475 w_k1_MAT_s006
                                 + 0.996883 w_k1_MSFT_s006
                                 + 1.0317 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.36053 w_k1_ADM_s006
                                 + 1.67398 w_k1_AXP_s006
                                 + 1.44434 w_k1_DIS_s006
                                 + 1.51336 w_k1_FDX_s006
                                 + 1.17897 w_k1_IBM_s006
                                 + 1.35935 w_k1_INTC_s006
                                 + 1.37049 w_k1_KO_s006 + 1.03283 w_k1_MAT_s006
                                 + 1.35776 w_k1_MSFT_s006
                                 + 1.20447 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.907392 w_k1_ADM_s006
                                 + 1.09328 w_k1_AXP_s006
                                 + 0.990061 w_k1_DIS_s006
                                 + 0.968443 w_k1_FDX_s006
                                 + 1.31613 w_k1_IBM_s006
                                 + 0.819735 w_k1_INTC_s006
                                 + 1.23511 w_k1_KO_s006 + 1.36627 w_k1_MAT_s006
                                 + 0.947758 w_k1_MSFT_s006
                                 + 1.11539 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.11375 w_k1_ADM_s007
                                 + 1.23787 w_k1_AXP_s007
                                 + 1.43129 w_k1_DIS_s007 + 1.2987 w_k1_FDX_s007
                                 + 1.08113 w_k1_IBM_s007
                                 + 0.792325 w_k1_INTC_s007
                                 + 0.963234 w_k1_KO_s007
                                 + 1.17855 w_k1_MAT_s007
                                 + 1.12233 w_k1_MSFT_s007 + 1.1211 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.808696 w_k1_ADM_s007
                                 + 0.859321 w_k1_AXP_s007
                                 + 0.969589 w_k1_DIS_s007
                                 + 0.889165 w_k1_FDX_s007
                                 + 1.0731 w_k1_IBM_s007
                                 + 0.759248 w_k1_INTC_s007
                                 + 1.03517 w_k1_KO_s007 + 1.22653 w_k1_MAT_s007
                                 + 0.794877 w_k1_MSFT_s007
                                 + 1.01708 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.680207 w_k1_ADM_s007
                                 + 0.572122 w_k1_AXP_s007
                                 + 0.636266 w_k1_DIS_s007
                                 + 0.666752 w_k1_FDX_s007
                                 + 0.657528 w_k1_IBM_s007
                                 + 0.537878 w_k1_INTC_s007
                                 + 0.616558 w_k1_KO_s007
                                 + 0.696243 w_k1_MAT_s007
                                 + 0.680155 w_k1_MSFT_s007
                                 + 0.711519 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.928747 w_k1_ADM_s007
                                 + 0.923455 w_k1_AXP_s007
                                 + 0.950913 w_k1_DIS_s007
                                 + 1.09634 w_k1_FDX_s007
                                 + 0.58566 w_k1_IBM_s007
                                 + 0.809782 w_k1_INTC_s007
                                 + 0.590322 w_k1_KO_s007
                                 + 0.52214 w_k1_MAT_s007
                                 + 0.896553 w_k1_MSFT_s007
                                 + 0.856352 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.02428 w_k1_ADM_s007
                                 + 0.751979 w_k1_AXP_s007
                                 + 0.843089 w_k1_DIS_s007
                                 + 0.878398 w_k1_FDX_s007
                                 + 0.781155 w_k1_IBM_s007
                                 + 0.766907 w_k1_INTC_s007
                                 + 0.727886 w_k1_KO_s007
                                 + 0.610635 w_k1_MAT_s007
                                 + 0.871904 w_k1_MSFT_s007
                                 + 0.851423 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.77199 w_k1_ADM_s007
                                 + 0.836068 w_k1_AXP_s007
                                 + 1.22829 w_k1_DIS_s007
                                 + 0.967455 w_k1_FDX_s007
                                 + 1.35176 w_k1_IBM_s007
                                 + 0.882808 w_k1_INTC_s007
                                 + 1.16094 w_k1_KO_s007 + 1.22981 w_k1_MAT_s007
                                 + 0.976838 w_k1_MSFT_s007
                                 + 1.05038 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.859412 w_k1_ADM_s007
                                 + 1.29109 w_k1_AXP_s007
                                 + 1.49143 w_k1_DIS_s007
                                 + 1.20484 w_k1_FDX_s007
                                 + 1.61952 w_k1_IBM_s007
                                 + 1.11785 w_k1_INTC_s007
                                 + 1.23789 w_k1_KO_s007 + 1.83881 w_k1_MAT_s007
                                 + 1.16816 w_k1_MSFT_s007
                                 + 1.24092 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.21106 w_k1_ADM_s007
                                 + 1.1548 w_k1_AXP_s007 + 1.31066 w_k1_DIS_s007
                                 + 1.17603 w_k1_FDX_s007
                                 + 0.848874 w_k1_IBM_s007
                                 + 0.891044 w_k1_INTC_s007
                                 + 0.998993 w_k1_KO_s007
                                 + 1.19655 w_k1_MAT_s007
                                 + 1.08625 w_k1_MSFT_s007
                                 + 1.10473 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.875886 w_k1_ADM_s010
                                 + 0.949437 w_k1_AXP_s010
                                 + 0.934602 w_k1_DIS_s010
                                 + 0.98799 w_k1_FDX_s010
                                 + 0.943045 w_k1_IBM_s010
                                 + 1.18873 w_k1_INTC_s010
                                 + 0.971451 w_k1_KO_s010
                                 + 0.828823 w_k1_MAT_s010
                                 + 1.00864 w_k1_MSFT_s010 + 1.0007 w_k1_PG_s010
                                 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.62429 w_k1_ADM_s010
                                 + 0.781549 w_k1_AXP_s010
                                 + 0.900969 w_k1_DIS_s010
                                 + 0.835185 w_k1_FDX_s010
                                 + 0.815276 w_k1_IBM_s010
                                 + 1.06522 w_k1_INTC_s010
                                 + 0.768753 w_k1_KO_s010
                                 + 0.790494 w_k1_MAT_s010
                                 + 0.808969 w_k1_MSFT_s010
                                 + 0.895984 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.779236 w_k1_ADM_s010
                                 + 0.918243 w_k1_AXP_s010
                                 + 1.21497 w_k1_DIS_s010
                                 + 0.959126 w_k1_FDX_s010
                                 + 1.39075 w_k1_IBM_s010
                                 + 1.03082 w_k1_INTC_s010
                                 + 1.18505 w_k1_KO_s010 + 1.44992 w_k1_MAT_s010
                                 + 1.22971 w_k1_MSFT_s010
                                 + 1.03534 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.59753 w_k1_ADM_s010
                                 + 0.940895 w_k1_AXP_s010
                                 + 0.769261 w_k1_DIS_s010
                                 + 0.739303 w_k1_FDX_s010
                                 + 1.01117 w_k1_IBM_s010
                                 + 0.954836 w_k1_INTC_s010
                                 + 0.833683 w_k1_KO_s010
                                 + 0.863579 w_k1_MAT_s010
                                 + 0.71858 w_k1_MSFT_s010
                                 + 0.772562 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.886696 w_k1_ADM_s010
                                 + 0.915924 w_k1_AXP_s010
                                 + 0.714821 w_k1_DIS_s010
                                 + 0.940231 w_k1_FDX_s010
                                 + 0.695219 w_k1_IBM_s010
                                 + 0.825649 w_k1_INTC_s010
                                 + 0.717384 w_k1_KO_s010
                                 + 0.768246 w_k1_MAT_s010
                                 + 0.81594 w_k1_MSFT_s010
                                 + 0.974342 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.729787 w_k1_ADM_s010
                                 + 0.648445 w_k1_AXP_s010
                                 + 0.679146 w_k1_DIS_s010
                                 + 0.731522 w_k1_FDX_s010
                                 + 0.630379 w_k1_IBM_s010
                                 + 0.882525 w_k1_INTC_s010
                                 + 0.739798 w_k1_KO_s010
                                 + 0.578062 w_k1_MAT_s010
                                 + 0.895902 w_k1_MSFT_s010
                                 + 0.811645 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.863171 w_k1_ADM_s010
                                 + 0.985802 w_k1_AXP_s010
                                 + 1.19925 w_k1_DIS_s010
                                 + 1.05135 w_k1_FDX_s010
                                 + 1.09454 w_k1_IBM_s010
                                 + 1.19527 w_k1_INTC_s010
                                 + 1.16405 w_k1_KO_s010
                                 + 0.923704 w_k1_MAT_s010
                                 + 1.10816 w_k1_MSFT_s010 + 1.0535 w_k1_PG_s010
                                 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.950995 w_k1_ADM_s010
                                 + 1.27091 w_k1_AXP_s010
                                 + 1.41538 w_k1_DIS_s010
                                 + 1.22689 w_k1_FDX_s010 + 1.3027 w_k1_IBM_s010
                                 + 1.42583 w_k1_INTC_s010
                                 + 1.26444 w_k1_KO_s010 + 1.42829 w_k1_MAT_s010
                                 + 1.21005 w_k1_MSFT_s010
                                 + 1.11912 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.1347 w_k1_ADM_s011
                                 + 1.22703 w_k1_AXP_s011
                                 + 1.10898 w_k1_DIS_s011
                                 + 1.33942 w_k1_FDX_s011
                                 + 0.697106 w_k1_IBM_s011
                                 + 1.11592 w_k1_INTC_s011
                                 + 0.861779 w_k1_KO_s011
                                 + 0.657732 w_k1_MAT_s011
                                 + 1.05072 w_k1_MSFT_s011 + 1.1089 w_k1_PG_s011
                                 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.704107 w_k1_ADM_s011
                                 + 0.854354 w_k1_AXP_s011
                                 + 0.900423 w_k1_DIS_s011
                                 + 0.789776 w_k1_FDX_s011
                                 + 1.37438 w_k1_IBM_s011
                                 + 0.830201 w_k1_INTC_s011
                                 + 1.09401 w_k1_KO_s011 + 1.05504 w_k1_MAT_s011
                                 + 0.843669 w_k1_MSFT_s011
                                 + 0.988971 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.957726 w_k1_ADM_s011
                                 + 1.13195 w_k1_AXP_s011
                                 + 1.71668 w_k1_DIS_s011
                                 + 1.33159 w_k1_FDX_s011
                                 + 1.28928 w_k1_IBM_s011
                                 + 1.39103 w_k1_INTC_s011
                                 + 1.63795 w_k1_KO_s011 + 1.64651 w_k1_MAT_s011
                                 + 1.4165 w_k1_MSFT_s011 + 1.42995 w_k1_PG_s011
                                 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.972536 w_k1_ADM_s011
                                 + 0.84601 w_k1_AXP_s011
                                 + 1.09685 w_k1_DIS_s011
                                 + 0.995167 w_k1_FDX_s011
                                 + 0.998114 w_k1_IBM_s011
                                 + 0.957014 w_k1_INTC_s011
                                 + 1.08714 w_k1_KO_s011
                                 + 0.747352 w_k1_MAT_s011
                                 + 1.08341 w_k1_MSFT_s011
                                 + 0.998109 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.01456 w_k1_ADM_s011
                                 + 1.1006 w_k1_AXP_s011
                                 + 0.845817 w_k1_DIS_s011
                                 + 0.941709 w_k1_FDX_s011
                                 + 0.863026 w_k1_IBM_s011
                                 + 0.804694 w_k1_INTC_s011
                                 + 0.949009 w_k1_KO_s011
                                 + 0.87555 w_k1_MAT_s011
                                 + 0.878692 w_k1_MSFT_s011
                                 + 0.947056 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.36317 w_k1_ADM_s011
                                 + 1.41533 w_k1_AXP_s011
                                 + 1.23076 w_k1_DIS_s011
                                 + 1.14024 w_k1_FDX_s011
                                 + 1.58142 w_k1_IBM_s011
                                 + 0.976392 w_k1_INTC_s011
                                 + 1.21772 w_k1_KO_s011 + 1.22147 w_k1_MAT_s011
                                 + 0.772767 w_k1_MSFT_s011
                                 + 1.10534 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.734463 w_k1_ADM_s011
                                 + 1.3862 w_k1_AXP_s011
                                 + 0.871577 w_k1_DIS_s011
                                 + 0.95549 w_k1_FDX_s011
                                 + 1.25697 w_k1_IBM_s011
                                 + 1.0877 w_k1_INTC_s011 + 1.1318 w_k1_KO_s011
                                 + 1.39491 w_k1_MAT_s011
                                 + 0.854963 w_k1_MSFT_s011 + 1.134 w_k1_PG_s011
                                 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.800367 w_k1_ADM_s011
                                 + 0.77075 w_k1_AXP_s011
                                 + 0.666842 w_k1_DIS_s011
                                 + 0.928154 w_k1_FDX_s011
                                 + 0.528826 w_k1_IBM_s011
                                 + 0.739682 w_k1_INTC_s011
                                 + 0.709723 w_k1_KO_s011
                                 + 0.652153 w_k1_MAT_s011
                                 + 0.867621 w_k1_MSFT_s011
                                 + 0.746874 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.13979 w_k1_ADM_s012
                                 + 0.776412 w_k1_AXP_s012
                                 + 0.838486 w_k1_DIS_s012
                                 + 0.931653 w_k1_FDX_s012
                                 + 0.697933 w_k1_IBM_s012
                                 + 1.05437 w_k1_INTC_s012
                                 + 0.858613 w_k1_KO_s012
                                 + 0.537739 w_k1_MAT_s012
                                 + 0.989163 w_k1_MSFT_s012
                                 + 0.957126 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.07373 w_k1_ADM_s012
                                 + 1.11097 w_k1_AXP_s012
                                 + 1.19135 w_k1_DIS_s012
                                 + 1.03788 w_k1_FDX_s012 + 1.8475 w_k1_IBM_s012
                                 + 1.32556 w_k1_INTC_s012 + 1.3674 w_k1_KO_s012
                                 + 1.33069 w_k1_MAT_s012
                                 + 1.06463 w_k1_MSFT_s012
                                 + 0.99042 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.896205 w_k1_ADM_s012
                                 + 1.06194 w_k1_AXP_s012
                                 + 0.983304 w_k1_DIS_s012
                                 + 0.888818 w_k1_FDX_s012
                                 + 1.65662 w_k1_IBM_s012
                                 + 1.02207 w_k1_INTC_s012
                                 + 1.47312 w_k1_KO_s012 + 1.72217 w_k1_MAT_s012
                                 + 1.00805 w_k1_MSFT_s012
                                 + 1.06408 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.34939 w_k1_ADM_s012
                                 + 1.53413 w_k1_AXP_s012
                                 + 1.64344 w_k1_DIS_s012
                                 + 1.42593 w_k1_FDX_s012
                                 + 1.31935 w_k1_IBM_s012
                                 + 1.28992 w_k1_INTC_s012
                                 + 1.22005 w_k1_KO_s012 + 1.51496 w_k1_MAT_s012
                                 + 1.20779 w_k1_MSFT_s012 + 1.157 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.07406 w_k1_ADM_s012
                                 + 0.95271 w_k1_AXP_s012
                                 + 0.954521 w_k1_DIS_s012
                                 + 0.893204 w_k1_FDX_s012
                                 + 1.29572 w_k1_IBM_s012
                                 + 1.10384 w_k1_INTC_s012
                                 + 1.14015 w_k1_KO_s012 + 1.10709 w_k1_MAT_s012
                                 + 0.934163 w_k1_MSFT_s012
                                 + 1.01053 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.972091 w_k1_ADM_s012
                                 + 0.899111 w_k1_AXP_s012
                                 + 0.768847 w_k1_DIS_s012
                                 + 0.848503 w_k1_FDX_s012
                                 + 1.01949 w_k1_IBM_s012
                                 + 0.878881 w_k1_INTC_s012
                                 + 0.898813 w_k1_KO_s012
                                 + 0.81348 w_k1_MAT_s012
                                 + 0.862948 w_k1_MSFT_s012
                                 + 0.989178 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.648395 w_k1_ADM_s012
                                 + 0.620351 w_k1_AXP_s012
                                 + 0.583886 w_k1_DIS_s012
                                 + 0.696333 w_k1_FDX_s012
                                 + 0.845396 w_k1_IBM_s012
                                 + 0.814509 w_k1_INTC_s012
                                 + 0.899635 w_k1_KO_s012
                                 + 0.875652 w_k1_MAT_s012
                                 + 0.782101 w_k1_MSFT_s012
                                 + 0.876027 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.18391 w_k1_ADM_s012
                                 + 1.00444 w_k1_AXP_s012
                                 + 1.23361 w_k1_DIS_s012
                                 + 1.16581 w_k1_FDX_s012
                                 + 0.912456 w_k1_IBM_s012
                                 + 0.965127 w_k1_INTC_s012
                                 + 1.04717 w_k1_KO_s012
                                 + 0.800233 w_k1_MAT_s012
                                 + 1.08693 w_k1_MSFT_s012
                                 + 1.09195 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.818798 w_k1_ADM_s013
                                 + 1.17994 w_k1_AXP_s013
                                 + 0.777536 w_k1_DIS_s013
                                 + 0.974296 w_k1_FDX_s013
                                 + 1.07061 w_k1_IBM_s013
                                 + 1.10277 w_k1_INTC_s013
                                 + 0.960484 w_k1_KO_s013
                                 + 1.05994 w_k1_MAT_s013
                                 + 0.774914 w_k1_MSFT_s013
                                 + 0.937817 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.937694 w_k1_ADM_s013
                                 + 1.21193 w_k1_AXP_s013 + 1.1884 w_k1_DIS_s013
                                 + 1.0645 w_k1_FDX_s013 + 1.20202 w_k1_IBM_s013
                                 + 1.22469 w_k1_INTC_s013
                                 + 1.10075 w_k1_KO_s013
                                 + 0.938235 w_k1_MAT_s013
                                 + 1.04995 w_k1_MSFT_s013
                                 + 1.04569 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.26496 w_k1_ADM_s013
                                 + 1.12093 w_k1_AXP_s013
                                 + 1.19671 w_k1_DIS_s013
                                 + 1.20428 w_k1_FDX_s013
                                 + 0.810321 w_k1_IBM_s013
                                 + 1.07421 w_k1_INTC_s013
                                 + 1.00386 w_k1_KO_s013 + 1.06177 w_k1_MAT_s013
                                 + 1.12392 w_k1_MSFT_s013
                                 + 1.17216 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.942097 w_k1_ADM_s013
                                 + 1.03708 w_k1_AXP_s013
                                 + 1.00627 w_k1_DIS_s013
                                 + 1.00248 w_k1_FDX_s013
                                 + 1.13667 w_k1_IBM_s013
                                 + 0.976685 w_k1_INTC_s013
                                 + 0.92329 w_k1_KO_s013 + 1.04584 w_k1_MAT_s013
                                 + 0.90457 w_k1_MSFT_s013
                                 + 0.997784 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.20301 w_k1_ADM_s013
                                 + 1.15172 w_k1_AXP_s013
                                 + 1.02082 w_k1_DIS_s013
                                 + 1.09259 w_k1_FDX_s013
                                 + 1.56563 w_k1_IBM_s013
                                 + 1.10203 w_k1_INTC_s013 + 1.137 w_k1_KO_s013
                                 + 1.34131 w_k1_MAT_s013
                                 + 0.87541 w_k1_MSFT_s013
                                 + 1.23402 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.0963 w_k1_ADM_s013
                                 + 0.856877 w_k1_AXP_s013
                                 + 0.794943 w_k1_DIS_s013
                                 + 0.917129 w_k1_FDX_s013
                                 + 0.794356 w_k1_IBM_s013
                                 + 0.924357 w_k1_INTC_s013
                                 + 0.824464 w_k1_KO_s013
                                 + 0.576412 w_k1_MAT_s013
                                 + 0.819809 w_k1_MSFT_s013
                                 + 1.05043 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.18667 w_k1_ADM_s013
                                 + 1.55709 w_k1_AXP_s013
                                 + 1.55095 w_k1_DIS_s013
                                 + 1.32705 w_k1_FDX_s013 + 1.3811 w_k1_IBM_s013
                                 + 1.67464 w_k1_INTC_s013
                                 + 1.42867 w_k1_KO_s013 + 1.14264 w_k1_MAT_s013
                                 + 1.26925 w_k1_MSFT_s013
                                 + 1.19157 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.896122 w_k1_ADM_s013
                                 + 0.785503 w_k1_AXP_s013
                                 + 0.653288 w_k1_DIS_s013
                                 + 0.886909 w_k1_FDX_s013
                                 + 0.635359 w_k1_IBM_s013
                                 + 0.860503 w_k1_INTC_s013
                                 + 0.625837 w_k1_KO_s013
                                 + 0.514904 w_k1_MAT_s013
                                 + 0.84535 w_k1_MSFT_s013
                                 + 0.802692 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.937792 w_k1_ADM_s014
                                 + 0.999478 w_k1_AXP_s014
                                 + 1.32114 w_k1_DIS_s014
                                 + 1.00256 w_k1_FDX_s014
                                 + 1.28492 w_k1_IBM_s014
                                 + 1.2737 w_k1_INTC_s014 + 1.3446 w_k1_KO_s014
                                 + 0.950655 w_k1_MAT_s014
                                 + 1.14048 w_k1_MSFT_s014
                                 + 1.24799 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.845657 w_k1_ADM_s014
                                 + 0.913946 w_k1_AXP_s014
                                 + 0.841849 w_k1_DIS_s014
                                 + 0.957751 w_k1_FDX_s014
                                 + 0.91434 w_k1_IBM_s014
                                 + 0.821809 w_k1_INTC_s014
                                 + 1.00976 w_k1_KO_s014 + 1.13939 w_k1_MAT_s014
                                 + 0.927754 w_k1_MSFT_s014
                                 + 1.00999 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.695358 w_k1_ADM_s014
                                 + 0.699228 w_k1_AXP_s014
                                 + 0.803137 w_k1_DIS_s014
                                 + 0.796712 w_k1_FDX_s014
                                 + 0.868769 w_k1_IBM_s014
                                 + 0.872487 w_k1_INTC_s014
                                 + 1.0081 w_k1_KO_s014 + 0.9471 w_k1_MAT_s014
                                 + 0.996483 w_k1_MSFT_s014
                                 + 0.966175 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.09753 w_k1_ADM_s014
                                 + 1.19022 w_k1_AXP_s014 + 1.1072 w_k1_DIS_s014
                                 + 1.10534 w_k1_FDX_s014
                                 + 0.997259 w_k1_IBM_s014
                                 + 0.903917 w_k1_INTC_s014
                                 + 1.04547 w_k1_KO_s014 + 1.03069 w_k1_MAT_s014
                                 + 1.02608 w_k1_MSFT_s014
                                 + 1.06617 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.978332 w_k1_ADM_s014
                                 + 1.14614 w_k1_AXP_s014
                                 + 0.971865 w_k1_DIS_s014
                                 + 0.9896 w_k1_FDX_s014 + 1.49519 w_k1_IBM_s014
                                 + 1.03986 w_k1_INTC_s014
                                 + 1.25642 w_k1_KO_s014 + 1.84667 w_k1_MAT_s014
                                 + 1.02255 w_k1_MSFT_s014
                                 + 1.05589 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.06311 w_k1_ADM_s014
                                 + 1.37356 w_k1_AXP_s014
                                 + 1.23887 w_k1_DIS_s014
                                 + 1.22401 w_k1_FDX_s014
                                 + 1.43375 w_k1_IBM_s014
                                 + 1.07292 w_k1_INTC_s014
                                 + 1.16309 w_k1_KO_s014 + 1.98261 w_k1_MAT_s014
                                 + 1.33499 w_k1_MSFT_s014
                                 + 1.03854 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.823901 w_k1_ADM_s014
                                 + 0.560876 w_k1_AXP_s014
                                 + 0.575281 w_k1_DIS_s014
                                 + 0.737717 w_k1_FDX_s014
                                 + 0.703374 w_k1_IBM_s014
                                 + 0.783508 w_k1_INTC_s014
                                 + 0.75716 w_k1_KO_s014 + 0.90038 w_k1_MAT_s014
                                 + 0.946671 w_k1_MSFT_s014
                                 + 0.869438 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.31551 w_k1_ADM_s014
                                 + 1.2828 w_k1_AXP_s014 + 1.66334 w_k1_DIS_s014
                                 + 1.41294 w_k1_FDX_s014
                                 + 1.29614 w_k1_IBM_s014
                                 + 1.15068 w_k1_INTC_s014
                                 + 1.30476 w_k1_KO_s014 + 1.32096 w_k1_MAT_s014
                                 + 1.31568 w_k1_MSFT_s014
                                 + 1.16214 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.42969 w_k1_ADM_s015
                                 + 0.999053 w_k1_AXP_s015
                                 + 1.27346 w_k1_DIS_s015
                                 + 1.22424 w_k1_FDX_s015
                                 + 0.698336 w_k1_IBM_s015
                                 + 1.1238 w_k1_INTC_s015 + 1.12295 w_k1_KO_s015
                                 + 0.743083 w_k1_MAT_s015
                                 + 1.14023 w_k1_MSFT_s015
                                 + 1.04945 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.23548 w_k1_ADM_s015
                                 + 1.28377 w_k1_AXP_s015
                                 + 1.77413 w_k1_DIS_s015
                                 + 1.39094 w_k1_FDX_s015
                                 + 1.06874 w_k1_IBM_s015
                                 + 1.37643 w_k1_INTC_s015
                                 + 1.47403 w_k1_KO_s015 + 1.73716 w_k1_MAT_s015
                                 + 1.49218 w_k1_MSFT_s015
                                 + 1.07648 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.964522 w_k1_ADM_s015
                                 + 0.718904 w_k1_AXP_s015
                                 + 0.766258 w_k1_DIS_s015
                                 + 0.800787 w_k1_FDX_s015
                                 + 1.18746 w_k1_IBM_s015
                                 + 0.737953 w_k1_INTC_s015
                                 + 1.12635 w_k1_KO_s015 + 1.52625 w_k1_MAT_s015
                                 + 0.995847 w_k1_MSFT_s015
                                 + 0.904839 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.22916 w_k1_ADM_s015
                                 + 1.06725 w_k1_AXP_s015
                                 + 1.15906 w_k1_DIS_s015
                                 + 1.07186 w_k1_FDX_s015
                                 + 1.07018 w_k1_IBM_s015
                                 + 1.04614 w_k1_INTC_s015
                                 + 1.27065 w_k1_KO_s015 + 1.59121 w_k1_MAT_s015
                                 + 1.06418 w_k1_MSFT_s015
                                 + 1.02777 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.22113 w_k1_ADM_s015
                                 + 0.845922 w_k1_AXP_s015
                                 + 0.989733 w_k1_DIS_s015
                                 + 1.09574 w_k1_FDX_s015
                                 + 0.484621 w_k1_IBM_s015
                                 + 0.750587 w_k1_INTC_s015
                                 + 0.704096 w_k1_KO_s015
                                 + 0.626067 w_k1_MAT_s015
                                 + 0.908443 w_k1_MSFT_s015
                                 + 0.84068 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.868966 w_k1_ADM_s015
                                 + 0.614247 w_k1_AXP_s015
                                 + 0.734241 w_k1_DIS_s015
                                 + 0.79041 w_k1_FDX_s015
                                 + 0.577443 w_k1_IBM_s015
                                 + 0.765011 w_k1_INTC_s015
                                 + 0.688837 w_k1_KO_s015
                                 + 0.607116 w_k1_MAT_s015
                                 + 0.811853 w_k1_MSFT_s015
                                 + 0.816636 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.16913 w_k1_ADM_s015
                                 + 0.843428 w_k1_AXP_s015
                                 + 0.915298 w_k1_DIS_s015
                                 + 0.972287 w_k1_FDX_s015
                                 + 0.793015 w_k1_IBM_s015
                                 + 0.911337 w_k1_INTC_s015
                                 + 0.814945 w_k1_KO_s015
                                 + 0.841806 w_k1_MAT_s015
                                 + 1.00568 w_k1_MSFT_s015
                                 + 0.874528 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.37609 w_k1_ADM_s015
                                 + 1.33613 w_k1_AXP_s015
                                 + 1.35549 w_k1_DIS_s015
                                 + 1.05546 w_k1_FDX_s015
                                 + 1.62638 w_k1_IBM_s015
                                 + 1.13499 w_k1_INTC_s015
                                 + 1.71835 w_k1_KO_s015 + 2.18048 w_k1_MAT_s015
                                 + 1.17227 w_k1_MSFT_s015
                                 + 1.22251 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.657857 w_k1_ADM_s016
                                 + 0.660888 w_k1_AXP_s016
                                 + 0.63156 w_k1_DIS_s016
                                 + 0.774013 w_k1_FDX_s016
                                 + 0.791231 w_k1_IBM_s016
                                 + 0.822912 w_k1_INTC_s016
                                 + 0.730521 w_k1_KO_s016
                                 + 0.575326 w_k1_MAT_s016
                                 + 0.743559 w_k1_MSFT_s016
                                 + 0.790148 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.956482 w_k1_ADM_s016
                                 + 0.724989 w_k1_AXP_s016
                                 + 0.943005 w_k1_DIS_s016
                                 + 0.954213 w_k1_FDX_s016
                                 + 0.734264 w_k1_IBM_s016
                                 + 0.856305 w_k1_INTC_s016
                                 + 1.01048 w_k1_KO_s016
                                 + 0.628976 w_k1_MAT_s016
                                 + 0.997139 w_k1_MSFT_s016
                                 + 0.993195 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.804772 w_k1_ADM_s016
                                 + 0.939021 w_k1_AXP_s016
                                 + 1.11534 w_k1_DIS_s016
                                 + 1.14234 w_k1_FDX_s016
                                 + 0.818089 w_k1_IBM_s016
                                 + 1.06084 w_k1_INTC_s016
                                 + 0.923779 w_k1_KO_s016
                                 + 0.87369 w_k1_MAT_s016
                                 + 1.25147 w_k1_MSFT_s016
                                 + 1.02331 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.604328 w_k1_ADM_s016
                                 + 0.457318 w_k1_AXP_s016
                                 + 0.588595 w_k1_DIS_s016
                                 + 0.641288 w_k1_FDX_s016
                                 + 0.676374 w_k1_IBM_s016
                                 + 0.752517 w_k1_INTC_s016
                                 + 0.784105 w_k1_KO_s016
                                 + 0.44515 w_k1_MAT_s016
                                 + 0.772253 w_k1_MSFT_s016
                                 + 0.871804 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.31323 w_k1_ADM_s016
                                 + 1.08561 w_k1_AXP_s016
                                 + 1.08653 w_k1_DIS_s016 + 0.9809 w_k1_FDX_s016
                                 + 1.09982 w_k1_IBM_s016
                                 + 0.837465 w_k1_INTC_s016
                                 + 1.06292 w_k1_KO_s016
                                 + 0.975042 w_k1_MAT_s016
                                 + 1.06363 w_k1_MSFT_s016
                                 + 1.03169 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.41171 w_k1_ADM_s016
                                 + 1.40331 w_k1_AXP_s016
                                 + 1.44874 w_k1_DIS_s016
                                 + 1.43369 w_k1_FDX_s016
                                 + 1.04032 w_k1_IBM_s016
                                 + 1.17768 w_k1_INTC_s016
                                 + 1.30148 w_k1_KO_s016
                                 + 0.949262 w_k1_MAT_s016
                                 + 1.3939 w_k1_MSFT_s016 + 1.15205 w_k1_PG_s016
                                 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.751719 w_k1_ADM_s016
                                 + 0.955932 w_k1_AXP_s016
                                 + 0.880981 w_k1_DIS_s016
                                 + 0.852265 w_k1_FDX_s016
                                 + 1.3417 w_k1_IBM_s016
                                 + 0.780179 w_k1_INTC_s016
                                 + 0.867886 w_k1_KO_s016
                                 + 0.75505 w_k1_MAT_s016
                                 + 0.805696 w_k1_MSFT_s016
                                 + 0.835317 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.17971 w_k1_ADM_s016
                                 + 1.66346 w_k1_AXP_s016
                                 + 1.54412 w_k1_DIS_s016
                                 + 1.17815 w_k1_FDX_s016
                                 + 1.57916 w_k1_IBM_s016
                                 + 1.01139 w_k1_INTC_s016
                                 + 1.32948 w_k1_KO_s016 + 1.26852 w_k1_MAT_s016
                                 + 1.01841 w_k1_MSFT_s016
                                 + 1.13158 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.828337 w_k1_ADM_s017
                                 + 0.611148 w_k1_AXP_s017
                                 + 0.631428 w_k1_DIS_s017
                                 + 0.86387 w_k1_FDX_s017
                                 + 0.583938 w_k1_IBM_s017
                                 + 1.00833 w_k1_INTC_s017
                                 + 0.648828 w_k1_KO_s017
                                 + 0.541956 w_k1_MAT_s017
                                 + 0.717301 w_k1_MSFT_s017
                                 + 0.867173 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.886936 w_k1_ADM_s017
                                 + 0.933489 w_k1_AXP_s017
                                 + 0.950678 w_k1_DIS_s017
                                 + 1.08108 w_k1_FDX_s017
                                 + 0.792253 w_k1_IBM_s017
                                 + 1.17231 w_k1_INTC_s017
                                 + 0.786289 w_k1_KO_s017
                                 + 0.685132 w_k1_MAT_s017
                                 + 1.03379 w_k1_MSFT_s017
                                 + 0.956058 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.956848 w_k1_ADM_s017
                                 + 0.951914 w_k1_AXP_s017
                                 + 1.48718 w_k1_DIS_s017
                                 + 1.31874 w_k1_FDX_s017
                                 + 0.764115 w_k1_IBM_s017
                                 + 1.18824 w_k1_INTC_s017
                                 + 0.973055 w_k1_KO_s017
                                 + 0.63667 w_k1_MAT_s017
                                 + 1.20421 w_k1_MSFT_s017
                                 + 1.06354 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.799028 w_k1_ADM_s017
                                 + 1.01753 w_k1_AXP_s017
                                 + 0.992315 w_k1_DIS_s017
                                 + 1.1073 w_k1_FDX_s017 + 1.08791 w_k1_IBM_s017
                                 + 1.11692 w_k1_INTC_s017
                                 + 0.84493 w_k1_KO_s017
                                 + 0.896061 w_k1_MAT_s017
                                 + 1.00407 w_k1_MSFT_s017
                                 + 0.920266 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.0992 w_k1_ADM_s017
                                 + 1.29849 w_k1_AXP_s017
                                 + 1.46963 w_k1_DIS_s017
                                 + 1.22185 w_k1_FDX_s017
                                 + 1.15715 w_k1_IBM_s017
                                 + 1.29127 w_k1_INTC_s017
                                 + 1.13624 w_k1_KO_s017 + 1.12959 w_k1_MAT_s017
                                 + 1.24087 w_k1_MSFT_s017
                                 + 1.19893 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.895709 w_k1_ADM_s017
                                 + 0.774444 w_k1_AXP_s017
                                 + 0.727044 w_k1_DIS_s017
                                 + 0.890438 w_k1_FDX_s017
                                 + 1.01916 w_k1_IBM_s017
                                 + 0.981416 w_k1_INTC_s017
                                 + 0.838408 w_k1_KO_s017 + 1.0637 w_k1_MAT_s017
                                 + 0.845058 w_k1_MSFT_s017
                                 + 0.962636 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.817045 w_k1_ADM_s017
                                 + 1.16362 w_k1_AXP_s017
                                 + 1.00461 w_k1_DIS_s017
                                 + 1.03482 w_k1_FDX_s017
                                 + 1.62439 w_k1_IBM_s017
                                 + 1.13703 w_k1_INTC_s017
                                 + 1.36057 w_k1_KO_s017 + 1.42495 w_k1_MAT_s017
                                 + 1.16504 w_k1_MSFT_s017
                                 + 1.19963 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.42713 w_k1_ADM_s017
                                 + 1.8708 w_k1_AXP_s017 + 1.53457 w_k1_DIS_s017
                                 + 1.37533 w_k1_FDX_s017
                                 + 1.87416 w_k1_IBM_s017
                                 + 1.24717 w_k1_INTC_s017
                                 + 1.26152 w_k1_KO_s017 + 1.53384 w_k1_MAT_s017
                                 + 1.2294 w_k1_MSFT_s017 + 1.2941 w_k1_PG_s017
                                 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.896297 w_k1_ADM_s100
                                 + 0.899775 w_k1_AXP_s100
                                 + 1.07179 w_k1_DIS_s100
                                 + 1.08232 w_k1_FDX_s100 + 0.806 w_k1_IBM_s100
                                 + 1.15768 w_k1_INTC_s100
                                 + 1.10121 w_k1_KO_s100
                                 + 0.965201 w_k1_MAT_s100
                                 + 1.14893 w_k1_MSFT_s100
                                 + 0.907442 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.852065 w_k1_ADM_s100
                                 + 1.41145 w_k1_AXP_s100
                                 + 1.29893 w_k1_DIS_s100
                                 + 1.16989 w_k1_FDX_s100
                                 + 1.51802 w_k1_IBM_s100
                                 + 1.26069 w_k1_INTC_s100
                                 + 1.28663 w_k1_KO_s100 + 1.85931 w_k1_MAT_s100
                                 + 1.25259 w_k1_MSFT_s100
                                 + 1.09768 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.922445 w_k1_ADM_s100
                                 + 1.05239 w_k1_AXP_s100
                                 + 1.55664 w_k1_DIS_s100
                                 + 1.20178 w_k1_FDX_s100
                                 + 1.04531 w_k1_IBM_s100
                                 + 0.961002 w_k1_INTC_s100
                                 + 1.14885 w_k1_KO_s100
                                 + 0.988915 w_k1_MAT_s100
                                 + 1.16307 w_k1_MSFT_s100 + 1.0494 w_k1_PG_s100
                                 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.908302 w_k1_ADM_s100
                                 + 0.889941 w_k1_AXP_s100
                                 + 0.92381 w_k1_DIS_s100
                                 + 0.922657 w_k1_FDX_s100
                                 + 1.15234 w_k1_IBM_s100
                                 + 0.863748 w_k1_INTC_s100
                                 + 1.05464 w_k1_KO_s100 + 1.36651 w_k1_MAT_s100
                                 + 1.00032 w_k1_MSFT_s100
                                 + 0.943172 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.631952 w_k1_ADM_s100
                                 + 0.691042 w_k1_AXP_s100
                                 + 0.655323 w_k1_DIS_s100
                                 + 0.686985 w_k1_FDX_s100
                                 + 0.780943 w_k1_IBM_s100
                                 + 0.838748 w_k1_INTC_s100
                                 + 0.785156 w_k1_KO_s100
                                 + 0.62125 w_k1_MAT_s100
                                 + 0.69266 w_k1_MSFT_s100
                                 + 0.762095 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.890983 w_k1_ADM_s100
                                 + 1.03439 w_k1_AXP_s100
                                 + 0.755782 w_k1_DIS_s100
                                 + 0.879372 w_k1_FDX_s100
                                 + 0.903926 w_k1_IBM_s100
                                 + 0.849425 w_k1_INTC_s100
                                 + 0.787954 w_k1_KO_s100
                                 + 0.875933 w_k1_MAT_s100
                                 + 0.830982 w_k1_MSFT_s100
                                 + 0.859319 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.960037 w_k1_ADM_s100
                                 + 0.923129 w_k1_AXP_s100
                                 + 0.856047 w_k1_DIS_s100
                                 + 0.921646 w_k1_FDX_s100
                                 + 0.684986 w_k1_IBM_s100
                                 + 1.00742 w_k1_INTC_s100
                                 + 0.745164 w_k1_KO_s100
                                 + 0.59622 w_k1_MAT_s100
                                 + 0.886791 w_k1_MSFT_s100
                                 + 0.781905 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.05101 w_k1_ADM_s100
                                 + 1.66867 w_k1_AXP_s100
                                 + 1.16311 w_k1_DIS_s100
                                 + 1.16979 w_k1_FDX_s100 + 1.1142 w_k1_IBM_s100
                                 + 1.03328 w_k1_INTC_s100
                                 + 1.10075 w_k1_KO_s100 + 1.72346 w_k1_MAT_s100
                                 + 1.01897 w_k1_MSFT_s100
                                 + 1.07034 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.00506 w_k1_ADM_s101
                                 + 1.15474 w_k1_AXP_s101
                                 + 1.06793 w_k1_DIS_s101
                                 + 1.03642 w_k1_FDX_s101
                                 + 0.995783 w_k1_IBM_s101
                                 + 1.15563 w_k1_INTC_s101
                                 + 1.00649 w_k1_KO_s101 + 0.85768 w_k1_MAT_s101
                                 + 0.995701 w_k1_MSFT_s101
                                 + 1.0328 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.922884 w_k1_ADM_s101
                                 + 1.05667 w_k1_AXP_s101
                                 + 0.931015 w_k1_DIS_s101
                                 + 0.998943 w_k1_FDX_s101
                                 + 1.24888 w_k1_IBM_s101
                                 + 1.1534 w_k1_INTC_s101
                                 + 0.891525 w_k1_KO_s101 + 1.2711 w_k1_MAT_s101
                                 + 1.02328 w_k1_MSFT_s101
                                 + 0.940444 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.23402 w_k1_ADM_s101
                                 + 0.891204 w_k1_AXP_s101
                                 + 1.18579 w_k1_DIS_s101
                                 + 1.05336 w_k1_FDX_s101
                                 + 1.15242 w_k1_IBM_s101
                                 + 0.898879 w_k1_INTC_s101
                                 + 1.00608 w_k1_KO_s101
                                 + 0.775913 w_k1_MAT_s101
                                 + 1.04774 w_k1_MSFT_s101
                                 + 1.03589 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.902731 w_k1_ADM_s101
                                 + 0.903093 w_k1_AXP_s101
                                 + 0.69609 w_k1_DIS_s101
                                 + 0.811793 w_k1_FDX_s101
                                 + 0.992225 w_k1_IBM_s101
                                 + 0.744378 w_k1_INTC_s101
                                 + 0.722358 w_k1_KO_s101
                                 + 1.01655 w_k1_MAT_s101
                                 + 0.796302 w_k1_MSFT_s101
                                 + 0.79617 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.23527 w_k1_ADM_s101
                                 + 1.03156 w_k1_AXP_s101
                                 + 1.51171 w_k1_DIS_s101
                                 + 1.28831 w_k1_FDX_s101
                                 + 0.976962 w_k1_IBM_s101
                                 + 1.09184 w_k1_INTC_s101
                                 + 1.05284 w_k1_KO_s101
                                 + 0.800876 w_k1_MAT_s101
                                 + 1.2017 w_k1_MSFT_s101 + 1.11425 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.23037 w_k1_ADM_s101
                                 + 1.3958 w_k1_AXP_s101 + 1.05457 w_k1_DIS_s101
                                 + 1.21174 w_k1_FDX_s101
                                 + 0.974071 w_k1_IBM_s101
                                 + 0.985721 w_k1_INTC_s101
                                 + 0.924132 w_k1_KO_s101
                                 + 1.17547 w_k1_MAT_s101
                                 + 0.990548 w_k1_MSFT_s101
                                 + 0.979944 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.04727 w_k1_ADM_s101
                                 + 0.711298 w_k1_AXP_s101
                                 + 0.78335 w_k1_DIS_s101
                                 + 0.94231 w_k1_FDX_s101
                                 + 0.671561 w_k1_IBM_s101
                                 + 0.832933 w_k1_INTC_s101
                                 + 0.751343 w_k1_KO_s101
                                 + 0.52751 w_k1_MAT_s101
                                 + 0.793869 w_k1_MSFT_s101
                                 + 0.92093 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.10272 w_k1_ADM_s101
                                 + 1.43631 w_k1_AXP_s101
                                 + 1.33004 w_k1_DIS_s101
                                 + 1.22333 w_k1_FDX_s101
                                 + 1.75926 w_k1_IBM_s101
                                 + 1.11959 w_k1_INTC_s101
                                 + 1.21708 w_k1_KO_s101 + 1.73559 w_k1_MAT_s101
                                 + 1.15827 w_k1_MSFT_s101
                                 + 1.11585 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.16162 w_k1_ADM_s102
                                 + 1.39457 w_k1_AXP_s102
                                 + 1.19116 w_k1_DIS_s102
                                 + 1.14862 w_k1_FDX_s102
                                 + 1.47773 w_k1_IBM_s102
                                 + 1.05225 w_k1_INTC_s102
                                 + 1.22209 w_k1_KO_s102 + 1.64411 w_k1_MAT_s102
                                 + 0.99523 w_k1_MSFT_s102
                                 + 1.12536 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.870774 w_k1_ADM_s102
                                 + 0.855815 w_k1_AXP_s102
                                 + 0.934188 w_k1_DIS_s102
                                 + 0.853775 w_k1_FDX_s102
                                 + 1.19421 w_k1_IBM_s102
                                 + 0.892324 w_k1_INTC_s102
                                 + 1.06282 w_k1_KO_s102
                                 + 0.946343 w_k1_MAT_s102
                                 + 0.896006 w_k1_MSFT_s102
                                 + 0.955769 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.73799 w_k1_ADM_s102
                                 + 0.747769 w_k1_AXP_s102
                                 + 0.741617 w_k1_DIS_s102
                                 + 0.866067 w_k1_FDX_s102
                                 + 0.644989 w_k1_IBM_s102
                                 + 0.856098 w_k1_INTC_s102
                                 + 0.724877 w_k1_KO_s102
                                 + 0.760833 w_k1_MAT_s102
                                 + 0.864499 w_k1_MSFT_s102
                                 + 0.781615 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.05779 w_k1_ADM_s102
                                 + 1.00293 w_k1_AXP_s102
                                 + 0.806218 w_k1_DIS_s102
                                 + 0.924269 w_k1_FDX_s102
                                 + 0.884779 w_k1_IBM_s102
                                 + 0.918724 w_k1_INTC_s102
                                 + 0.921755 w_k1_KO_s102
                                 + 0.775831 w_k1_MAT_s102
                                 + 0.889803 w_k1_MSFT_s102
                                 + 0.928923 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.00618 w_k1_ADM_s102
                                 + 1.06923 w_k1_AXP_s102 + 1.1447 w_k1_DIS_s102
                                 + 1.18639 w_k1_FDX_s102
                                 + 0.782258 w_k1_IBM_s102
                                 + 1.14052 w_k1_INTC_s102
                                 + 1.11221 w_k1_KO_s102
                                 + 0.841674 w_k1_MAT_s102
                                 + 1.16509 w_k1_MSFT_s102 + 1.0606 w_k1_PG_s102
                                 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.22601 w_k1_ADM_s102
                                 + 1.62963 w_k1_AXP_s102
                                 + 2.01784 w_k1_DIS_s102
                                 + 1.49858 w_k1_FDX_s102
                                 + 1.26215 w_k1_IBM_s102
                                 + 1.39673 w_k1_INTC_s102
                                 + 1.43911 w_k1_KO_s102 + 1.53478 w_k1_MAT_s102
                                 + 1.5558 w_k1_MSFT_s102 + 1.30152 w_k1_PG_s102
                                 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.756461 w_k1_ADM_s102
                                 + 0.588126 w_k1_AXP_s102
                                 + 0.507907 w_k1_DIS_s102
                                 + 0.678804 w_k1_FDX_s102
                                 + 0.725292 w_k1_IBM_s102
                                 + 0.577311 w_k1_INTC_s102
                                 + 0.635327 w_k1_KO_s102
                                 + 0.669452 w_k1_MAT_s102
                                 + 0.682449 w_k1_MSFT_s102
                                 + 0.701609 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.844428 w_k1_ADM_s102
                                 + 0.894138 w_k1_AXP_s102
                                 + 1.2523 w_k1_DIS_s102 + 1.11976 w_k1_FDX_s102
                                 + 0.822461 w_k1_IBM_s102
                                 + 1.14975 w_k1_INTC_s102
                                 + 1.22625 w_k1_KO_s102 + 1.25729 w_k1_MAT_s102
                                 + 1.25951 w_k1_MSFT_s102
                                 + 1.07974 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.971772 w_k1_ADM_s103
                                 + 0.536073 w_k1_AXP_s103
                                 + 0.606837 w_k1_DIS_s103
                                 + 0.755395 w_k1_FDX_s103
                                 + 0.631757 w_k1_IBM_s103
                                 + 0.767738 w_k1_INTC_s103
                                 + 0.65929 w_k1_KO_s103
                                 + 0.554244 w_k1_MAT_s103
                                 + 0.729267 w_k1_MSFT_s103
                                 + 0.867919 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.43962 w_k1_ADM_s103
                                 + 1.43395 w_k1_AXP_s103 + 1.5297 w_k1_DIS_s103
                                 + 1.3546 w_k1_FDX_s103 + 1.22994 w_k1_IBM_s103
                                 + 1.34949 w_k1_INTC_s103
                                 + 1.29525 w_k1_KO_s103 + 1.05768 w_k1_MAT_s103
                                 + 1.17734 w_k1_MSFT_s103
                                 + 1.25456 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.938387 w_k1_ADM_s103
                                 + 0.952268 w_k1_AXP_s103
                                 + 0.863674 w_k1_DIS_s103
                                 + 0.890559 w_k1_FDX_s103
                                 + 0.987045 w_k1_IBM_s103
                                 + 1.17971 w_k1_INTC_s103
                                 + 0.851314 w_k1_KO_s103
                                 + 0.84925 w_k1_MAT_s103
                                 + 0.921023 w_k1_MSFT_s103
                                 + 0.985057 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.03727 w_k1_ADM_s103
                                 + 1.00683 w_k1_AXP_s103
                                 + 1.18505 w_k1_DIS_s103
                                 + 1.07696 w_k1_FDX_s103
                                 + 0.966941 w_k1_IBM_s103
                                 + 1.18728 w_k1_INTC_s103
                                 + 1.09343 w_k1_KO_s103
                                 + 0.724599 w_k1_MAT_s103
                                 + 1.11211 w_k1_MSFT_s103
                                 + 1.09568 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.00977 w_k1_ADM_s103
                                 + 0.790489 w_k1_AXP_s103
                                 + 0.77804 w_k1_DIS_s103
                                 + 0.936032 w_k1_FDX_s103
                                 + 0.705385 w_k1_IBM_s103
                                 + 0.801446 w_k1_INTC_s103
                                 + 0.807948 w_k1_KO_s103
                                 + 0.774968 w_k1_MAT_s103
                                 + 0.858255 w_k1_MSFT_s103
                                 + 0.966243 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.09404 w_k1_ADM_s103
                                 + 1.05038 w_k1_AXP_s103
                                 + 1.11065 w_k1_DIS_s103
                                 + 0.987169 w_k1_FDX_s103
                                 + 1.30801 w_k1_IBM_s103
                                 + 1.02774 w_k1_INTC_s103
                                 + 1.03525 w_k1_KO_s103 + 1.39588 w_k1_MAT_s103
                                 + 0.973487 w_k1_MSFT_s103
                                 + 1.19045 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.914218 w_k1_ADM_s103
                                 + 0.739195 w_k1_AXP_s103
                                 + 0.573978 w_k1_DIS_s103
                                 + 0.747889 w_k1_FDX_s103
                                 + 0.912707 w_k1_IBM_s103
                                 + 0.7955 w_k1_INTC_s103
                                 + 0.825652 w_k1_KO_s103
                                 + 0.750533 w_k1_MAT_s103
                                 + 0.76262 w_k1_MSFT_s103 + 0.8573 w_k1_PG_s103
                                 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.22628 w_k1_ADM_s103
                                 + 1.04882 w_k1_AXP_s103
                                 + 0.858161 w_k1_DIS_s103
                                 + 1.09573 w_k1_FDX_s103
                                 + 0.775376 w_k1_IBM_s103
                                 + 0.975464 w_k1_INTC_s103
                                 + 0.790301 w_k1_KO_s103
                                 + 0.918193 w_k1_MAT_s103
                                 + 0.91896 w_k1_MSFT_s103
                                 + 0.989369 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.98564 w_k1_ADM_s104
                                 + 0.951038 w_k1_AXP_s104
                                 + 0.900344 w_k1_DIS_s104
                                 + 0.951852 w_k1_FDX_s104
                                 + 0.85327 w_k1_IBM_s104
                                 + 0.8389 w_k1_INTC_s104
                                 + 0.877681 w_k1_KO_s104
                                 + 0.851626 w_k1_MAT_s104
                                 + 0.833601 w_k1_MSFT_s104
                                 + 0.895944 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.797911 w_k1_ADM_s104
                                 + 0.760382 w_k1_AXP_s104
                                 + 0.764285 w_k1_DIS_s104
                                 + 0.77247 w_k1_FDX_s104
                                 + 0.808826 w_k1_IBM_s104
                                 + 0.74727 w_k1_INTC_s104
                                 + 0.765005 w_k1_KO_s104
                                 + 0.810473 w_k1_MAT_s104
                                 + 0.810335 w_k1_MSFT_s104
                                 + 0.849943 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.28102 w_k1_ADM_s104
                                 + 1.08866 w_k1_AXP_s104
                                 + 1.28063 w_k1_DIS_s104
                                 + 1.15888 w_k1_FDX_s104 + 1.0087 w_k1_IBM_s104
                                 + 1.01489 w_k1_INTC_s104
                                 + 1.10453 w_k1_KO_s104 + 0.87346 w_k1_MAT_s104
                                 + 1.14808 w_k1_MSFT_s104
                                 + 1.03836 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.986078 w_k1_ADM_s104
                                 + 1.42677 w_k1_AXP_s104 + 1.0235 w_k1_DIS_s104
                                 + 1.10523 w_k1_FDX_s104
                                 + 1.01691 w_k1_IBM_s104
                                 + 1.02004 w_k1_INTC_s104
                                 + 1.00884 w_k1_KO_s104 + 1.18694 w_k1_MAT_s104
                                 + 0.921157 w_k1_MSFT_s104
                                 + 1.00005 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.876325 w_k1_ADM_s104
                                 + 0.886308 w_k1_AXP_s104
                                 + 0.910616 w_k1_DIS_s104
                                 + 1.05169 w_k1_FDX_s104
                                 + 0.623845 w_k1_IBM_s104
                                 + 1.08518 w_k1_INTC_s104
                                 + 0.805936 w_k1_KO_s104
                                 + 0.624278 w_k1_MAT_s104
                                 + 1.01624 w_k1_MSFT_s104
                                 + 0.860888 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.927054 w_k1_ADM_s104
                                 + 1.51396 w_k1_AXP_s104
                                 + 1.39198 w_k1_DIS_s104
                                 + 1.15405 w_k1_FDX_s104
                                 + 1.58943 w_k1_IBM_s104
                                 + 1.02334 w_k1_INTC_s104
                                 + 1.31048 w_k1_KO_s104 + 1.72453 w_k1_MAT_s104
                                 + 1.1894 w_k1_MSFT_s104 + 1.19322 w_k1_PG_s104
                                 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.828596 w_k1_ADM_s104
                                 + 0.885743 w_k1_AXP_s104
                                 + 0.873787 w_k1_DIS_s104
                                 + 0.826478 w_k1_FDX_s104
                                 + 1.31934 w_k1_IBM_s104
                                 + 0.925961 w_k1_INTC_s104
                                 + 1.07364 w_k1_KO_s104 + 1.2925 w_k1_MAT_s104
                                 + 0.88238 w_k1_MSFT_s104 + 1.0065 w_k1_PG_s104
                                 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.964818 w_k1_ADM_s104
                                 + 1.28832 w_k1_AXP_s104
                                 + 1.53139 w_k1_DIS_s104
                                 + 1.12856 w_k1_FDX_s104
                                 + 1.22504 w_k1_IBM_s104 + 1.118 w_k1_INTC_s104
                                 + 1.12094 w_k1_KO_s104 + 1.16024 w_k1_MAT_s104
                                 + 1.14324 w_k1_MSFT_s104
                                 + 1.02035 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.09386 w_k1_ADM_s105
                                 + 1.36944 w_k1_AXP_s105
                                 + 1.14256 w_k1_DIS_s105
                                 + 1.14593 w_k1_FDX_s105
                                 + 1.38409 w_k1_IBM_s105
                                 + 1.02315 w_k1_INTC_s105 + 1.2403 w_k1_KO_s105
                                 + 1.45428 w_k1_MAT_s105
                                 + 1.08386 w_k1_MSFT_s105
                                 + 1.18075 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.46073 w_k1_ADM_s105
                                 + 1.34453 w_k1_AXP_s105
                                 + 1.59687 w_k1_DIS_s105
                                 + 1.40281 w_k1_FDX_s105 + 1.2974 w_k1_IBM_s105
                                 + 1.11189 w_k1_INTC_s105
                                 + 1.35529 w_k1_KO_s105 + 2.01552 w_k1_MAT_s105
                                 + 1.36524 w_k1_MSFT_s105
                                 + 1.27871 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.904454 w_k1_ADM_s105
                                 + 0.661147 w_k1_AXP_s105
                                 + 0.875297 w_k1_DIS_s105
                                 + 0.909342 w_k1_FDX_s105
                                 + 0.705875 w_k1_IBM_s105
                                 + 0.909505 w_k1_INTC_s105
                                 + 0.843934 w_k1_KO_s105
                                 + 0.733216 w_k1_MAT_s105
                                 + 1.00606 w_k1_MSFT_s105
                                 + 0.906989 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.833349 w_k1_ADM_s105
                                 + 0.529856 w_k1_AXP_s105
                                 + 0.497811 w_k1_DIS_s105
                                 + 0.679213 w_k1_FDX_s105
                                 + 0.736422 w_k1_IBM_s105
                                 + 0.621505 w_k1_INTC_s105
                                 + 0.755194 w_k1_KO_s105
                                 + 0.603337 w_k1_MAT_s105
                                 + 0.697935 w_k1_MSFT_s105
                                 + 0.758129 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.937576 w_k1_ADM_s105
                                 + 0.968454 w_k1_AXP_s105
                                 + 0.603277 w_k1_DIS_s105
                                 + 0.849127 w_k1_FDX_s105
                                 + 0.767932 w_k1_IBM_s105
                                 + 0.846066 w_k1_INTC_s105
                                 + 0.720862 w_k1_KO_s105
                                 + 0.781663 w_k1_MAT_s105
                                 + 0.75315 w_k1_MSFT_s105
                                 + 0.817035 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.03665 w_k1_ADM_s105
                                 + 0.920143 w_k1_AXP_s105
                                 + 1.00088 w_k1_DIS_s105
                                 + 0.940984 w_k1_FDX_s105
                                 + 1.2026 w_k1_IBM_s105
                                 + 0.906478 w_k1_INTC_s105
                                 + 1.12352 w_k1_KO_s105 + 1.27773 w_k1_MAT_s105
                                 + 1.01826 w_k1_MSFT_s105
                                 + 1.05943 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.07248 w_k1_ADM_s105
                                 + 0.951355 w_k1_AXP_s105
                                 + 1.05632 w_k1_DIS_s105
                                 + 1.15947 w_k1_FDX_s105
                                 + 0.831808 w_k1_IBM_s105
                                 + 1.05762 w_k1_INTC_s105
                                 + 1.14124 w_k1_KO_s105 + 1.08251 w_k1_MAT_s105
                                 + 1.1165 w_k1_MSFT_s105 + 1.06878 w_k1_PG_s105
                                 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.965792 w_k1_ADM_s105
                                 + 0.914509 w_k1_AXP_s105
                                 + 0.684217 w_k1_DIS_s105
                                 + 0.831676 w_k1_FDX_s105
                                 + 1.03536 w_k1_IBM_s105
                                 + 0.820606 w_k1_INTC_s105
                                 + 0.925594 w_k1_KO_s105
                                 + 1.12326 w_k1_MAT_s105
                                 + 0.843511 w_k1_MSFT_s105
                                 + 0.999395 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.925221 w_k1_ADM_s106
                                 + 1.0118 w_k1_AXP_s106 + 1.043 w_k1_DIS_s106
                                 + 0.91735 w_k1_FDX_s106
                                 + 1.60165 w_k1_IBM_s106
                                 + 0.988423 w_k1_INTC_s106
                                 + 1.2876 w_k1_KO_s106 + 1.09694 w_k1_MAT_s106
                                 + 0.897805 w_k1_MSFT_s106
                                 + 1.04328 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.12555 w_k1_ADM_s106
                                 + 1.42059 w_k1_AXP_s106
                                 + 1.36052 w_k1_DIS_s106
                                 + 1.28458 w_k1_FDX_s106
                                 + 1.09143 w_k1_IBM_s106
                                 + 1.25679 w_k1_INTC_s106 + 1.1689 w_k1_KO_s106
                                 + 0.998783 w_k1_MAT_s106
                                 + 1.20084 w_k1_MSFT_s106
                                 + 1.17242 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.796511 w_k1_ADM_s106
                                 + 0.719011 w_k1_AXP_s106
                                 + 0.660341 w_k1_DIS_s106
                                 + 0.741808 w_k1_FDX_s106
                                 + 0.994927 w_k1_IBM_s106
                                 + 0.84652 w_k1_INTC_s106
                                 + 0.965663 w_k1_KO_s106
                                 + 0.584344 w_k1_MAT_s106
                                 + 0.755222 w_k1_MSFT_s106
                                 + 0.965067 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.662586 w_k1_ADM_s106
                                 + 0.611245 w_k1_AXP_s106
                                 + 0.579424 w_k1_DIS_s106
                                 + 0.666596 w_k1_FDX_s106
                                 + 0.670284 w_k1_IBM_s106
                                 + 0.797125 w_k1_INTC_s106
                                 + 0.794891 w_k1_KO_s106
                                 + 0.548887 w_k1_MAT_s106
                                 + 0.760591 w_k1_MSFT_s106
                                 + 0.811552 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.12294 w_k1_ADM_s106
                                 + 1.21392 w_k1_AXP_s106
                                 + 1.44382 w_k1_DIS_s106
                                 + 1.14063 w_k1_FDX_s106
                                 + 1.62269 w_k1_IBM_s106
                                 + 1.16571 w_k1_INTC_s106
                                 + 1.40674 w_k1_KO_s106 + 1.34517 w_k1_MAT_s106
                                 + 1.18992 w_k1_MSFT_s106
                                 + 1.23399 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.934815 w_k1_ADM_s106
                                 + 0.956158 w_k1_AXP_s106
                                 + 0.970097 w_k1_DIS_s106
                                 + 0.936262 w_k1_FDX_s106
                                 + 1.12126 w_k1_IBM_s106
                                 + 0.858137 w_k1_INTC_s106
                                 + 1.03184 w_k1_KO_s106 + 1.04473 w_k1_MAT_s106
                                 + 0.930075 w_k1_MSFT_s106
                                 + 1.15227 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.913333 w_k1_ADM_s106
                                 + 0.669111 w_k1_AXP_s106
                                 + 0.594004 w_k1_DIS_s106
                                 + 0.884244 w_k1_FDX_s106
                                 + 0.508491 w_k1_IBM_s106
                                 + 0.789498 w_k1_INTC_s106
                                 + 0.669139 w_k1_KO_s106
                                 + 0.482039 w_k1_MAT_s106
                                 + 0.893298 w_k1_MSFT_s106
                                 + 0.886101 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.908196 w_k1_ADM_s106
                                 + 0.844057 w_k1_AXP_s106
                                 + 0.934844 w_k1_DIS_s106
                                 + 1.04545 w_k1_FDX_s106
                                 + 0.794669 w_k1_IBM_s106
                                 + 1.00003 w_k1_INTC_s106
                                 + 0.984258 w_k1_KO_s106
                                 + 0.611852 w_k1_MAT_s106
                                 + 0.934527 w_k1_MSFT_s106
                                 + 0.930937 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.769826 w_k1_ADM_s107
                                 + 0.713004 w_k1_AXP_s107
                                 + 0.701306 w_k1_DIS_s107
                                 + 0.793619 w_k1_FDX_s107
                                 + 0.652807 w_k1_IBM_s107
                                 + 0.69688 w_k1_INTC_s107
                                 + 0.724783 w_k1_KO_s107
                                 + 0.595032 w_k1_MAT_s107
                                 + 0.734234 w_k1_MSFT_s107
                                 + 0.793335 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.897913 w_k1_ADM_s107
                                 + 0.878436 w_k1_AXP_s107
                                 + 0.698755 w_k1_DIS_s107
                                 + 0.808941 w_k1_FDX_s107
                                 + 1.16718 w_k1_IBM_s107
                                 + 0.687243 w_k1_INTC_s107
                                 + 0.796111 w_k1_KO_s107 + 1.0424 w_k1_MAT_s107
                                 + 0.778638 w_k1_MSFT_s107
                                 + 0.956813 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.797693 w_k1_ADM_s107
                                 + 0.948503 w_k1_AXP_s107
                                 + 0.794028 w_k1_DIS_s107
                                 + 0.961096 w_k1_FDX_s107
                                 + 0.884885 w_k1_IBM_s107
                                 + 0.807689 w_k1_INTC_s107
                                 + 0.824968 w_k1_KO_s107
                                 + 0.914675 w_k1_MAT_s107
                                 + 0.836623 w_k1_MSFT_s107
                                 + 0.83763 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.952094 w_k1_ADM_s107
                                 + 0.856091 w_k1_AXP_s107
                                 + 1.21466 w_k1_DIS_s107
                                 + 0.980547 w_k1_FDX_s107
                                 + 1.14136 w_k1_IBM_s107
                                 + 0.979573 w_k1_INTC_s107
                                 + 1.24451 w_k1_KO_s107 + 1.15594 w_k1_MAT_s107
                                 + 1.06437 w_k1_MSFT_s107
                                 + 1.11261 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.0724 w_k1_ADM_s107
                                 + 1.29341 w_k1_AXP_s107
                                 + 1.31452 w_k1_DIS_s107
                                 + 1.15907 w_k1_FDX_s107
                                 + 1.31183 w_k1_IBM_s107
                                 + 0.943033 w_k1_INTC_s107
                                 + 1.14336 w_k1_KO_s107 + 1.53671 w_k1_MAT_s107
                                 + 1.16324 w_k1_MSFT_s107
                                 + 1.03178 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.97033 w_k1_ADM_s107
                                 + 1.26916 w_k1_AXP_s107
                                 + 1.14679 w_k1_DIS_s107
                                 + 1.19706 w_k1_FDX_s107
                                 + 0.875043 w_k1_IBM_s107
                                 + 1.05677 w_k1_INTC_s107
                                 + 0.957744 w_k1_KO_s107
                                 + 1.06772 w_k1_MAT_s107
                                 + 1.01588 w_k1_MSFT_s107
                                 + 0.998765 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.40609 w_k1_ADM_s107
                                 + 1.63413 w_k1_AXP_s107
                                 + 1.75814 w_k1_DIS_s107 + 1.3143 w_k1_FDX_s107
                                 + 1.67564 w_k1_IBM_s107
                                 + 1.04359 w_k1_INTC_s107
                                 + 1.36396 w_k1_KO_s107 + 1.90669 w_k1_MAT_s107
                                 + 1.41881 w_k1_MSFT_s107
                                 + 1.20734 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.971275 w_k1_ADM_s107
                                 + 0.799633 w_k1_AXP_s107
                                 + 0.917204 w_k1_DIS_s107
                                 + 1.03004 w_k1_FDX_s107
                                 + 0.643896 w_k1_IBM_s107
                                 + 0.907477 w_k1_INTC_s107
                                 + 0.814708 w_k1_KO_s107
                                 + 0.452386 w_k1_MAT_s107
                                 + 0.949123 w_k1_MSFT_s107
                                 + 0.839321 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.830749 w_k1_ADM_s110
                                 + 0.690694 w_k1_AXP_s110
                                 + 0.65475 w_k1_DIS_s110
                                 + 0.726277 w_k1_FDX_s110
                                 + 1.02895 w_k1_IBM_s110
                                 + 0.942769 w_k1_INTC_s110
                                 + 0.892398 w_k1_KO_s110
                                 + 0.818776 w_k1_MAT_s110
                                 + 0.800144 w_k1_MSFT_s110
                                 + 0.891686 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.94763 w_k1_ADM_s110
                                 + 1.63382 w_k1_AXP_s110
                                 + 1.39836 w_k1_DIS_s110
                                 + 1.16259 w_k1_FDX_s110
                                 + 1.53329 w_k1_IBM_s110
                                 + 1.36854 w_k1_INTC_s110
                                 + 1.36275 w_k1_KO_s110 + 1.76909 w_k1_MAT_s110
                                 + 1.16915 w_k1_MSFT_s110
                                 + 1.20278 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.681599 w_k1_ADM_s110
                                 + 0.557254 w_k1_AXP_s110
                                 + 0.545313 w_k1_DIS_s110
                                 + 0.594162 w_k1_FDX_s110
                                 + 0.615373 w_k1_IBM_s110
                                 + 0.684736 w_k1_INTC_s110
                                 + 0.611667 w_k1_KO_s110
                                 + 0.379036 w_k1_MAT_s110
                                 + 0.609009 w_k1_MSFT_s110
                                 + 0.650379 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.03452 w_k1_ADM_s110
                                 + 0.918883 w_k1_AXP_s110
                                 + 1.02799 w_k1_DIS_s110
                                 + 1.00526 w_k1_FDX_s110
                                 + 0.76538 w_k1_IBM_s110
                                 + 1.07977 w_k1_INTC_s110
                                 + 0.879952 w_k1_KO_s110
                                 + 0.575305 w_k1_MAT_s110
                                 + 1.07273 w_k1_MSFT_s110
                                 + 0.935198 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.756175 w_k1_ADM_s110
                                 + 0.892073 w_k1_AXP_s110
                                 + 0.982047 w_k1_DIS_s110
                                 + 0.852953 w_k1_FDX_s110
                                 + 1.47385 w_k1_IBM_s110
                                 + 0.841344 w_k1_INTC_s110
                                 + 1.12126 w_k1_KO_s110 + 1.12787 w_k1_MAT_s110
                                 + 0.796342 w_k1_MSFT_s110
                                 + 0.920183 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.929444 w_k1_ADM_s110
                                 + 1.09003 w_k1_AXP_s110
                                 + 1.05893 w_k1_DIS_s110
                                 + 0.99811 w_k1_FDX_s110
                                 + 1.01014 w_k1_IBM_s110
                                 + 1.02613 w_k1_INTC_s110
                                 + 1.01487 w_k1_KO_s110
                                 + 0.897048 w_k1_MAT_s110
                                 + 0.977222 w_k1_MSFT_s110
                                 + 0.995081 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.28398 w_k1_ADM_s110
                                 + 1.28504 w_k1_AXP_s110
                                 + 1.71852 w_k1_DIS_s110
                                 + 1.35568 w_k1_FDX_s110
                                 + 1.11003 w_k1_IBM_s110
                                 + 1.32497 w_k1_INTC_s110
                                 + 1.42225 w_k1_KO_s110 + 1.1851 w_k1_MAT_s110
                                 + 1.31892 w_k1_MSFT_s110
                                 + 1.17219 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.698051 w_k1_ADM_s110
                                 + 0.700378 w_k1_AXP_s110
                                 + 0.683707 w_k1_DIS_s110
                                 + 0.732833 w_k1_FDX_s110
                                 + 0.739339 w_k1_IBM_s110
                                 + 0.92748 w_k1_INTC_s110
                                 + 0.832388 w_k1_KO_s110
                                 + 0.529792 w_k1_MAT_s110
                                 + 0.793342 w_k1_MSFT_s110
                                 + 0.750571 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.49671 w_k1_ADM_s111
                                 + 1.3512 w_k1_AXP_s111 + 1.15241 w_k1_DIS_s111
                                 + 1.30366 w_k1_FDX_s111
                                 + 0.920745 w_k1_IBM_s111
                                 + 1.08403 w_k1_INTC_s111
                                 + 0.922942 w_k1_KO_s111
                                 + 1.02323 w_k1_MAT_s111
                                 + 1.12192 w_k1_MSFT_s111
                                 + 0.920768 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.07093 w_k1_ADM_s111
                                 + 0.904217 w_k1_AXP_s111
                                 + 0.728927 w_k1_DIS_s111
                                 + 0.836362 w_k1_FDX_s111
                                 + 1.28495 w_k1_IBM_s111
                                 + 0.843928 w_k1_INTC_s111
                                 + 0.946983 w_k1_KO_s111 + 1.4038 w_k1_MAT_s111
                                 + 0.894728 w_k1_MSFT_s111
                                 + 0.845736 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.24121 w_k1_ADM_s111
                                 + 0.960879 w_k1_AXP_s111
                                 + 0.960889 w_k1_DIS_s111
                                 + 1.03708 w_k1_FDX_s111
                                 + 1.02595 w_k1_IBM_s111
                                 + 0.930246 w_k1_INTC_s111
                                 + 0.873707 w_k1_KO_s111
                                 + 0.957452 w_k1_MAT_s111
                                 + 0.953101 w_k1_MSFT_s111
                                 + 0.92947 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.991488 w_k1_ADM_s111
                                 + 0.955942 w_k1_AXP_s111
                                 + 0.651385 w_k1_DIS_s111
                                 + 0.903151 w_k1_FDX_s111
                                 + 0.980071 w_k1_IBM_s111
                                 + 0.922128 w_k1_INTC_s111
                                 + 0.708409 w_k1_KO_s111
                                 + 0.839488 w_k1_MAT_s111
                                 + 0.85902 w_k1_MSFT_s111
                                 + 0.82541 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.16828 w_k1_ADM_s111
                                 + 1.23949 w_k1_AXP_s111
                                 + 1.85165 w_k1_DIS_s111
                                 + 1.40154 w_k1_FDX_s111
                                 + 1.43217 w_k1_IBM_s111
                                 + 1.58396 w_k1_INTC_s111
                                 + 1.53157 w_k1_KO_s111 + 1.29426 w_k1_MAT_s111
                                 + 1.36858 w_k1_MSFT_s111
                                 + 1.16085 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.05898 w_k1_ADM_s111
                                 + 1.00349 w_k1_AXP_s111
                                 + 1.30385 w_k1_DIS_s111
                                 + 1.03911 w_k1_FDX_s111
                                 + 1.53142 w_k1_IBM_s111
                                 + 1.01306 w_k1_INTC_s111
                                 + 1.23013 w_k1_KO_s111 + 1.55199 w_k1_MAT_s111
                                 + 0.993073 w_k1_MSFT_s111
                                 + 1.13068 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.94532 w_k1_ADM_s111
                                 + 0.800412 w_k1_AXP_s111
                                 + 0.717707 w_k1_DIS_s111
                                 + 0.829609 w_k1_FDX_s111
                                 + 0.751503 w_k1_IBM_s111
                                 + 1.0533 w_k1_INTC_s111 + 0.81524 w_k1_KO_s111
                                 + 0.696135 w_k1_MAT_s111
                                 + 0.88113 w_k1_MSFT_s111
                                 + 0.80477 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.31942 w_k1_ADM_s111
                                 + 0.783546 w_k1_AXP_s111
                                 + 1.25525 w_k1_DIS_s111
                                 + 1.13142 w_k1_FDX_s111
                                 + 0.935692 w_k1_IBM_s111
                                 + 1.00608 w_k1_INTC_s111
                                 + 1.10253 w_k1_KO_s111
                                 + 0.945288 w_k1_MAT_s111
                                 + 1.08974 w_k1_MSFT_s111
                                 + 1.11547 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.23338 w_k1_ADM_s112
                                 + 1.42569 w_k1_AXP_s112
                                 + 1.36592 w_k1_DIS_s112
                                 + 1.08539 w_k1_FDX_s112
                                 + 1.76017 w_k1_IBM_s112
                                 + 1.27051 w_k1_INTC_s112
                                 + 1.70425 w_k1_KO_s112 + 1.34036 w_k1_MAT_s112
                                 + 1.13555 w_k1_MSFT_s112
                                 + 1.20754 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.34181 w_k1_ADM_s112
                                 + 1.50482 w_k1_AXP_s112 + 1.1816 w_k1_DIS_s112
                                 + 1.2748 w_k1_FDX_s112
                                 + 0.959969 w_k1_IBM_s112
                                 + 1.55113 w_k1_INTC_s112
                                 + 1.18038 w_k1_KO_s112
                                 + 0.876929 w_k1_MAT_s112
                                 + 1.05938 w_k1_MSFT_s112
                                 + 1.07176 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.939802 w_k1_ADM_s112
                                 + 1.38237 w_k1_AXP_s112
                                 + 0.89478 w_k1_DIS_s112
                                 + 0.996464 w_k1_FDX_s112
                                 + 0.979663 w_k1_IBM_s112
                                 + 1.18676 w_k1_INTC_s112
                                 + 0.975269 w_k1_KO_s112
                                 + 0.918864 w_k1_MAT_s112
                                 + 0.84422 w_k1_MSFT_s112
                                 + 1.04753 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.11037 w_k1_ADM_s112
                                 + 0.820199 w_k1_AXP_s112
                                 + 0.838508 w_k1_DIS_s112
                                 + 0.891637 w_k1_FDX_s112
                                 + 0.953677 w_k1_IBM_s112
                                 + 0.975933 w_k1_INTC_s112
                                 + 0.914728 w_k1_KO_s112
                                 + 0.945404 w_k1_MAT_s112
                                 + 0.829195 w_k1_MSFT_s112
                                 + 0.865923 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.25811 w_k1_ADM_s112
                                 + 1.56404 w_k1_AXP_s112
                                 + 1.05445 w_k1_DIS_s112
                                 + 1.12966 w_k1_FDX_s112
                                 + 1.18901 w_k1_IBM_s112
                                 + 1.27828 w_k1_INTC_s112
                                 + 1.21785 w_k1_KO_s112 + 1.29421 w_k1_MAT_s112
                                 + 1.07942 w_k1_MSFT_s112
                                 + 1.06667 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.00229 w_k1_ADM_s112
                                 + 0.904351 w_k1_AXP_s112
                                 + 0.70496 w_k1_DIS_s112
                                 + 0.950848 w_k1_FDX_s112
                                 + 0.731991 w_k1_IBM_s112
                                 + 0.993163 w_k1_INTC_s112
                                 + 0.833714 w_k1_KO_s112
                                 + 0.664118 w_k1_MAT_s112
                                 + 0.7512 w_k1_MSFT_s112
                                 + 0.785592 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.817128 w_k1_ADM_s112
                                 + 0.776924 w_k1_AXP_s112
                                 + 0.706399 w_k1_DIS_s112
                                 + 0.736611 w_k1_FDX_s112
                                 + 1.18324 w_k1_IBM_s112
                                 + 0.935641 w_k1_INTC_s112
                                 + 1.10572 w_k1_KO_s112 + 0.92977 w_k1_MAT_s112
                                 + 0.783717 w_k1_MSFT_s112
                                 + 0.889232 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.851458 w_k1_ADM_s112
                                 + 0.951977 w_k1_AXP_s112
                                 + 0.876153 w_k1_DIS_s112
                                 + 0.941573 w_k1_FDX_s112
                                 + 1.4087 w_k1_IBM_s112
                                 + 1.01479 w_k1_INTC_s112
                                 + 1.03568 w_k1_KO_s112 + 1.0065 w_k1_MAT_s112
                                 + 0.814519 w_k1_MSFT_s112
                                 + 0.9425 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.24303 w_k1_ADM_s113
                                 + 1.32505 w_k1_AXP_s113
                                 + 1.30574 w_k1_DIS_s113
                                 + 1.28882 w_k1_FDX_s113
                                 + 1.11373 w_k1_IBM_s113
                                 + 1.52559 w_k1_INTC_s113 + 1.169 w_k1_KO_s113
                                 + 1.11498 w_k1_MAT_s113
                                 + 1.41886 w_k1_MSFT_s113
                                 + 1.17847 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.998136 w_k1_ADM_s113
                                 + 1.11563 w_k1_AXP_s113
                                 + 0.806161 w_k1_DIS_s113
                                 + 1.04046 w_k1_FDX_s113
                                 + 0.887259 w_k1_IBM_s113
                                 + 1.20237 w_k1_INTC_s113
                                 + 0.872057 w_k1_KO_s113
                                 + 1.10187 w_k1_MAT_s113
                                 + 1.11661 w_k1_MSFT_s113
                                 + 0.946635 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.14699 w_k1_ADM_s113
                                 + 1.05861 w_k1_AXP_s113
                                 + 1.12044 w_k1_DIS_s113 + 1.1714 w_k1_FDX_s113
                                 + 0.739686 w_k1_IBM_s113
                                 + 1.25752 w_k1_INTC_s113
                                 + 0.835827 w_k1_KO_s113
                                 + 1.01327 w_k1_MAT_s113
                                 + 1.23837 w_k1_MSFT_s113
                                 + 0.994365 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.02629 w_k1_ADM_s113
                                 + 0.816818 w_k1_AXP_s113
                                 + 1.04075 w_k1_DIS_s113
                                 + 1.02033 w_k1_FDX_s113
                                 + 1.13116 w_k1_IBM_s113
                                 + 1.08655 w_k1_INTC_s113 + 1.0916 w_k1_KO_s113
                                 + 0.905259 w_k1_MAT_s113
                                 + 1.27011 w_k1_MSFT_s113
                                 + 0.987145 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.859393 w_k1_ADM_s113
                                 + 0.794787 w_k1_AXP_s113
                                 + 0.818545 w_k1_DIS_s113
                                 + 0.952678 w_k1_FDX_s113
                                 + 0.689775 w_k1_IBM_s113
                                 + 1.14431 w_k1_INTC_s113
                                 + 0.749412 w_k1_KO_s113
                                 + 0.547656 w_k1_MAT_s113
                                 + 1.05032 w_k1_MSFT_s113
                                 + 0.804786 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.909187 w_k1_ADM_s113
                                 + 0.774781 w_k1_AXP_s113
                                 + 0.666145 w_k1_DIS_s113
                                 + 0.7968 w_k1_FDX_s113
                                 + 0.732971 w_k1_IBM_s113
                                 + 1.01715 w_k1_INTC_s113
                                 + 0.673604 w_k1_KO_s113
                                 + 0.660853 w_k1_MAT_s113
                                 + 0.811442 w_k1_MSFT_s113
                                 + 0.788515 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.945815 w_k1_ADM_s113
                                 + 1.09613 w_k1_AXP_s113
                                 + 0.722941 w_k1_DIS_s113
                                 + 0.903019 w_k1_FDX_s113
                                 + 1.21092 w_k1_IBM_s113
                                 + 1.01999 w_k1_INTC_s113
                                 + 0.928784 w_k1_KO_s113
                                 + 1.10411 w_k1_MAT_s113
                                 + 0.946553 w_k1_MSFT_s113
                                 + 0.905865 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.888981 w_k1_ADM_s113
                                 + 0.64545 w_k1_AXP_s113
                                 + 0.588731 w_k1_DIS_s113
                                 + 0.674815 w_k1_FDX_s113
                                 + 0.796817 w_k1_IBM_s113
                                 + 0.8748 w_k1_INTC_s113
                                 + 0.713007 w_k1_KO_s113
                                 + 0.740748 w_k1_MAT_s113
                                 + 0.862707 w_k1_MSFT_s113
                                 + 0.807393 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.09879 w_k1_ADM_s114
                                 + 1.05566 w_k1_AXP_s114
                                 + 0.924729 w_k1_DIS_s114
                                 + 0.950857 w_k1_FDX_s114
                                 + 1.16215 w_k1_IBM_s114
                                 + 1.1084 w_k1_INTC_s114 + 1.09214 w_k1_KO_s114
                                 + 1.04128 w_k1_MAT_s114
                                 + 0.96883 w_k1_MSFT_s114
                                 + 0.996396 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.49091 w_k1_ADM_s114
                                 + 1.39512 w_k1_AXP_s114
                                 + 1.53037 w_k1_DIS_s114
                                 + 1.40414 w_k1_FDX_s114 + 1.1083 w_k1_IBM_s114
                                 + 1.30109 w_k1_INTC_s114
                                 + 1.11366 w_k1_KO_s114 + 1.40689 w_k1_MAT_s114
                                 + 1.26055 w_k1_MSFT_s114
                                 + 1.30153 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.920501 w_k1_ADM_s114
                                 + 0.706175 w_k1_AXP_s114
                                 + 0.729023 w_k1_DIS_s114
                                 + 0.861009 w_k1_FDX_s114
                                 + 0.659927 w_k1_IBM_s114
                                 + 1.02125 w_k1_INTC_s114
                                 + 0.763431 w_k1_KO_s114
                                 + 0.376363 w_k1_MAT_s114
                                 + 0.842885 w_k1_MSFT_s114
                                 + 0.819934 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.07611 w_k1_ADM_s114
                                 + 0.713321 w_k1_AXP_s114
                                 + 0.627916 w_k1_DIS_s114
                                 + 0.747362 w_k1_FDX_s114
                                 + 0.95142 w_k1_IBM_s114
                                 + 0.674677 w_k1_INTC_s114
                                 + 0.702062 w_k1_KO_s114
                                 + 0.667532 w_k1_MAT_s114
                                 + 0.658275 w_k1_MSFT_s114
                                 + 0.896466 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.2679 w_k1_ADM_s114
                                 + 1.0477 w_k1_AXP_s114 + 1.32747 w_k1_DIS_s114
                                 + 1.08838 w_k1_FDX_s114
                                 + 1.38585 w_k1_IBM_s114
                                 + 1.02698 w_k1_INTC_s114 + 1.148 w_k1_KO_s114
                                 + 1.30534 w_k1_MAT_s114
                                 + 1.17579 w_k1_MSFT_s114
                                 + 1.11802 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.26538 w_k1_ADM_s114
                                 + 1.03922 w_k1_AXP_s114
                                 + 1.04924 w_k1_DIS_s114
                                 + 1.09819 w_k1_FDX_s114
                                 + 0.870038 w_k1_IBM_s114
                                 + 0.918572 w_k1_INTC_s114
                                 + 0.826308 w_k1_KO_s114
                                 + 0.938706 w_k1_MAT_s114
                                 + 1.0931 w_k1_MSFT_s114 + 1.05859 w_k1_PG_s114
                                 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.37399 w_k1_ADM_s114
                                 + 0.851106 w_k1_AXP_s114
                                 + 1.18332 w_k1_DIS_s114
                                 + 1.10554 w_k1_FDX_s114
                                 + 0.661685 w_k1_IBM_s114
                                 + 1.09419 w_k1_INTC_s114
                                 + 0.853836 w_k1_KO_s114
                                 + 0.510892 w_k1_MAT_s114
                                 + 1.17416 w_k1_MSFT_s114
                                 + 1.00445 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.05618 w_k1_ADM_s114
                                 + 0.782798 w_k1_AXP_s114
                                 + 0.720141 w_k1_DIS_s114
                                 + 0.812171 w_k1_FDX_s114
                                 + 0.858137 w_k1_IBM_s114
                                 + 0.749886 w_k1_INTC_s114
                                 + 0.810381 w_k1_KO_s114
                                 + 0.906387 w_k1_MAT_s114
                                 + 0.783008 w_k1_MSFT_s114
                                 + 0.943892 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.00439 w_k1_ADM_s115
                                 + 0.780769 w_k1_AXP_s115
                                 + 0.808012 w_k1_DIS_s115
                                 + 0.842082 w_k1_FDX_s115
                                 + 0.919593 w_k1_IBM_s115
                                 + 0.78186 w_k1_INTC_s115
                                 + 0.807188 w_k1_KO_s115
                                 + 0.971927 w_k1_MAT_s115
                                 + 0.821985 w_k1_MSFT_s115
                                 + 0.947035 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.44256 w_k1_ADM_s115
                                 + 1.30951 w_k1_AXP_s115 + 1.6935 w_k1_DIS_s115
                                 + 1.2256 w_k1_FDX_s115 + 1.55626 w_k1_IBM_s115
                                 + 1.12462 w_k1_INTC_s115
                                 + 1.50053 w_k1_KO_s115 + 2.12448 w_k1_MAT_s115
                                 + 1.31291 w_k1_MSFT_s115
                                 + 1.34681 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.1868 w_k1_ADM_s115
                                 + 1.1602 w_k1_AXP_s115 + 1.15507 w_k1_DIS_s115
                                 + 1.12799 w_k1_FDX_s115
                                 + 0.795151 w_k1_IBM_s115
                                 + 1.07588 w_k1_INTC_s115
                                 + 0.945132 w_k1_KO_s115
                                 + 0.989644 w_k1_MAT_s115
                                 + 1.08629 w_k1_MSFT_s115
                                 + 1.19837 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.31284 w_k1_ADM_s115
                                 + 1.70434 w_k1_AXP_s115 + 1.5862 w_k1_DIS_s115
                                 + 1.39975 w_k1_FDX_s115
                                 + 1.16224 w_k1_IBM_s115
                                 + 1.13029 w_k1_INTC_s115 + 1.0408 w_k1_KO_s115
                                 + 1.11224 w_k1_MAT_s115
                                 + 1.11872 w_k1_MSFT_s115
                                 + 1.20806 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.32053 w_k1_ADM_s115
                                 + 1.7575 w_k1_AXP_s115 + 1.08772 w_k1_DIS_s115
                                 + 1.15371 w_k1_FDX_s115
                                 + 1.67803 w_k1_IBM_s115
                                 + 1.11142 w_k1_INTC_s115
                                 + 1.15772 w_k1_KO_s115 + 1.66813 w_k1_MAT_s115
                                 + 0.914227 w_k1_MSFT_s115
                                 + 1.1339 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.82916 w_k1_ADM_s115
                                 + 0.753853 w_k1_AXP_s115
                                 + 0.747233 w_k1_DIS_s115
                                 + 0.801568 w_k1_FDX_s115
                                 + 0.672289 w_k1_IBM_s115
                                 + 0.828608 w_k1_INTC_s115
                                 + 0.641413 w_k1_KO_s115
                                 + 0.962159 w_k1_MAT_s115
                                 + 0.801933 w_k1_MSFT_s115
                                 + 0.821194 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.14005 w_k1_ADM_s115
                                 + 1.19442 w_k1_AXP_s115
                                 + 1.00398 w_k1_DIS_s115
                                 + 1.02978 w_k1_FDX_s115
                                 + 1.10295 w_k1_IBM_s115
                                 + 0.915188 w_k1_INTC_s115
                                 + 1.0224 w_k1_KO_s115 + 1.07092 w_k1_MAT_s115
                                 + 0.934238 w_k1_MSFT_s115
                                 + 1.11649 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.996806 w_k1_ADM_s115
                                 + 1.41836 w_k1_AXP_s115
                                 + 1.25036 w_k1_DIS_s115
                                 + 1.12148 w_k1_FDX_s115
                                 + 1.19082 w_k1_IBM_s115
                                 + 0.9291 w_k1_INTC_s115 + 1.14377 w_k1_KO_s115
                                 + 1.76315 w_k1_MAT_s115
                                 + 1.00755 w_k1_MSFT_s115
                                 + 1.27428 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.174 w_k1_ADM_s116
                                 + 0.797321 w_k1_AXP_s116
                                 + 0.819617 w_k1_DIS_s116
                                 + 0.985828 w_k1_FDX_s116
                                 + 0.561941 w_k1_IBM_s116
                                 + 0.942804 w_k1_INTC_s116
                                 + 0.819648 w_k1_KO_s116
                                 + 0.471461 w_k1_MAT_s116
                                 + 1.00744 w_k1_MSFT_s116
                                 + 0.868512 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.3122 w_k1_ADM_s116
                                 + 1.18685 w_k1_AXP_s116
                                 + 0.915988 w_k1_DIS_s116
                                 + 0.82728 w_k1_FDX_s116
                                 + 1.64046 w_k1_IBM_s116
                                 + 0.870972 w_k1_INTC_s116
                                 + 1.23667 w_k1_KO_s116 + 1.42555 w_k1_MAT_s116
                                 + 0.855954 w_k1_MSFT_s116
                                 + 1.14813 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.1659 w_k1_ADM_s116
                                 + 0.699687 w_k1_AXP_s116
                                 + 0.785372 w_k1_DIS_s116
                                 + 0.858636 w_k1_FDX_s116
                                 + 0.886984 w_k1_IBM_s116
                                 + 1.04626 w_k1_INTC_s116
                                 + 0.99881 w_k1_KO_s116
                                 + 0.731437 w_k1_MAT_s116
                                 + 0.876357 w_k1_MSFT_s116
                                 + 0.916464 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.00489 w_k1_ADM_s116
                                 + 0.624837 w_k1_AXP_s116
                                 + 0.588581 w_k1_DIS_s116
                                 + 0.710783 w_k1_FDX_s116
                                 + 0.719834 w_k1_IBM_s116
                                 + 0.70577 w_k1_INTC_s116
                                 + 0.712763 w_k1_KO_s116
                                 + 0.720726 w_k1_MAT_s116
                                 + 0.672177 w_k1_MSFT_s116
                                 + 0.777601 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.31563 w_k1_ADM_s116
                                 + 1.12812 w_k1_AXP_s116
                                 + 1.18391 w_k1_DIS_s116
                                 + 1.25599 w_k1_FDX_s116
                                 + 0.743493 w_k1_IBM_s116
                                 + 1.24594 w_k1_INTC_s116
                                 + 0.983646 w_k1_KO_s116
                                 + 0.813379 w_k1_MAT_s116
                                 + 1.29052 w_k1_MSFT_s116
                                 + 1.16472 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.94606 w_k1_ADM_s116
                                 + 1.42772 w_k1_AXP_s116
                                 + 1.82535 w_k1_DIS_s116
                                 + 1.47501 w_k1_FDX_s116 + 1.3841 w_k1_IBM_s116
                                 + 1.3308 w_k1_INTC_s116 + 1.44897 w_k1_KO_s116
                                 + 1.5638 w_k1_MAT_s116
                                 + 1.38192 w_k1_MSFT_s116 + 1.1472 w_k1_PG_s116
                                 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.11805 w_k1_ADM_s116
                                 + 1.0384 w_k1_AXP_s116
                                 + 0.854655 w_k1_DIS_s116
                                 + 0.920777 w_k1_FDX_s116
                                 + 0.796193 w_k1_IBM_s116
                                 + 0.949737 w_k1_INTC_s116
                                 + 0.884932 w_k1_KO_s116
                                 + 0.751708 w_k1_MAT_s116
                                 + 0.743644 w_k1_MSFT_s116
                                 + 0.883573 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.20006 w_k1_ADM_s116
                                 + 0.877504 w_k1_AXP_s116
                                 + 1.11888 w_k1_DIS_s116
                                 + 1.00869 w_k1_FDX_s116
                                 + 1.11051 w_k1_IBM_s116
                                 + 0.888858 w_k1_INTC_s116
                                 + 1.07937 w_k1_KO_s116 + 1.23427 w_k1_MAT_s116
                                 + 0.863201 w_k1_MSFT_s116
                                 + 1.03751 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.277 w_k1_ADM_s117
                                 + 1.50037 w_k1_AXP_s117
                                 + 1.40927 w_k1_DIS_s117
                                 + 1.09684 w_k1_FDX_s117
                                 + 1.89036 w_k1_IBM_s117
                                 + 1.04367 w_k1_INTC_s117
                                 + 1.31135 w_k1_KO_s117 + 2.14691 w_k1_MAT_s117
                                 + 1.13407 w_k1_MSFT_s117
                                 + 1.00082 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.01902 w_k1_ADM_s117
                                 + 0.826335 w_k1_AXP_s117
                                 + 0.784079 w_k1_DIS_s117
                                 + 0.780806 w_k1_FDX_s117
                                 + 1.21352 w_k1_IBM_s117
                                 + 0.920083 w_k1_INTC_s117
                                 + 0.950989 w_k1_KO_s117
                                 + 1.09814 w_k1_MAT_s117
                                 + 0.861467 w_k1_MSFT_s117
                                 + 0.849395 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.796343 w_k1_ADM_s117
                                 + 0.97369 w_k1_AXP_s117 + 1.1649 w_k1_DIS_s117
                                 + 1.07908 w_k1_FDX_s117
                                 + 0.993321 w_k1_IBM_s117
                                 + 0.982187 w_k1_INTC_s117
                                 + 1.00637 w_k1_KO_s117 + 1.14365 w_k1_MAT_s117
                                 + 1.03239 w_k1_MSFT_s117
                                 + 1.06892 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.73836 w_k1_ADM_s117
                                 + 1.03679 w_k1_AXP_s117
                                 + 0.687986 w_k1_DIS_s117
                                 + 0.90345 w_k1_FDX_s117 + 0.9251 w_k1_IBM_s117
                                 + 0.939655 w_k1_INTC_s117
                                 + 0.881121 w_k1_KO_s117
                                 + 0.882201 w_k1_MAT_s117
                                 + 0.810866 w_k1_MSFT_s117
                                 + 0.871022 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.18866 w_k1_ADM_s117
                                 + 1.26491 w_k1_AXP_s117
                                 + 1.13913 w_k1_DIS_s117
                                 + 1.35259 w_k1_FDX_s117
                                 + 0.736958 w_k1_IBM_s117
                                 + 0.951572 w_k1_INTC_s117
                                 + 0.881571 w_k1_KO_s117
                                 + 0.910138 w_k1_MAT_s117
                                 + 1.08692 w_k1_MSFT_s117
                                 + 1.03454 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.05497 w_k1_ADM_s117
                                 + 1.23651 w_k1_AXP_s117
                                 + 1.07932 w_k1_DIS_s117
                                 + 1.08835 w_k1_FDX_s117
                                 + 1.26519 w_k1_IBM_s117
                                 + 0.923764 w_k1_INTC_s117
                                 + 1.13584 w_k1_KO_s117 + 1.27745 w_k1_MAT_s117
                                 + 0.906875 w_k1_MSFT_s117
                                 + 1.0102 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.889996 w_k1_ADM_s117
                                 + 0.834993 w_k1_AXP_s117
                                 + 0.702305 w_k1_DIS_s117
                                 + 0.906249 w_k1_FDX_s117
                                 + 0.704502 w_k1_IBM_s117
                                 + 0.764136 w_k1_INTC_s117
                                 + 0.751944 w_k1_KO_s117
                                 + 0.829084 w_k1_MAT_s117
                                 + 0.792586 w_k1_MSFT_s117
                                 + 0.810463 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.0816 w_k1_ADM_s117
                                 + 1.31411 w_k1_AXP_s117
                                 + 1.32327 w_k1_DIS_s117
                                 + 1.16907 w_k1_FDX_s117
                                 + 1.35801 w_k1_IBM_s117
                                 + 0.954213 w_k1_INTC_s117
                                 + 1.11801 w_k1_KO_s117 + 1.37274 w_k1_MAT_s117
                                 + 1.07524 w_k1_MSFT_s117 + 1.1214 w_k1_PG_s117
                                 >= 1
 k1_return_constraint:           0.0123959180936531 w_k1_ADM_s000
                                 + 0.0395648378376483 w_k1_ADM_s001
                                 + 0.00895621965252898 w_k1_ADM_s002
                                 + 0.0190307921236913 w_k1_ADM_s003
                                 + 0.0427804606388799 w_k1_ADM_s004
                                 + 0.0182651711282474 w_k1_ADM_s005
                                 + 0.0146645995002239 w_k1_ADM_s006
                                 + 0.0377429738630282 w_k1_ADM_s007
                                 + 0.0112587554824739 w_k1_ADM_s010
                                 + 0.00742881284619226 w_k1_ADM_s011
                                 + 0.0200955039421779 w_k1_ADM_s012
                                 + 0.0196472811750234 w_k1_ADM_s013
                                 + 0.0183119745318439 w_k1_ADM_s014
                                 + 0.016790483364473 w_k1_ADM_s015
                                 + 0.00585203607559943 w_k1_ADM_s016
                                 + 0.0136028791447057 w_k1_ADM_s017
                                 + 0.0280135395785873 w_k1_ADM_s100
                                 + 0.0974112655142206 w_k1_ADM_s101
                                 + 0.0619686289799407 w_k1_ADM_s102
                                 + 0.0472927950214019 w_k1_ADM_s103
                                 + 0.0708624110585476 w_k1_ADM_s104
                                 + 0.0774856661023455 w_k1_ADM_s105
                                 + 0.0394470342380736 w_k1_ADM_s106
                                 + 0.0236078904159377 w_k1_ADM_s107
                                 + 0.0477901637583694 w_k1_ADM_s110
                                 + 0.0536767399878231 w_k1_ADM_s111
                                 + 0.0152955456266523 w_k1_ADM_s112
                                 + 0.0279148091966782 w_k1_ADM_s113
                                 + 0.0538937074381444 w_k1_ADM_s114
                                 + 0.0173085873897272 w_k1_ADM_s115
                                 + 0.0146545816807276 w_k1_ADM_s116
                                 + 0.0398393945617133 w_k1_ADM_s117
                                 + 0.0116306159847143 w_k1_AXP_s000
                                 + 0.04441636386013 w_k1_AXP_s001
                                 + 0.00922587326934533 w_k1_AXP_s002
                                 + 0.0196676124802939 w_k1_AXP_s003
                                 + 0.0395070598700816 w_k1_AXP_s004
                                 + 0.0194227468140502 w_k1_AXP_s005
                                 + 0.0148673481057861 w_k1_AXP_s006
                                 + 0.0391967013993078 w_k1_AXP_s007
                                 + 0.0133475402694218 w_k1_AXP_s010
                                 + 0.00820300995428854 w_k1_AXP_s011
                                 + 0.0179236461712957 w_k1_AXP_s012
                                 + 0.0198242453778785 w_k1_AXP_s013
                                 + 0.0192921332134139 w_k1_AXP_s014
                                 + 0.0134863254387442 w_k1_AXP_s015
                                 + 0.00576566492182125 w_k1_AXP_s016
                                 + 0.01425750448352 w_k1_AXP_s017
                                 + 0.0353307401277831 w_k1_AXP_s100
                                 + 0.0973657036811522 w_k1_AXP_s101
                                 + 0.0662301365729251 w_k1_AXP_s102
                                 + 0.0430938568017066 w_k1_AXP_s103
                                 + 0.0758298841400313 w_k1_AXP_s104
                                 + 0.0720272199124658 w_k1_AXP_s105
                                 + 0.0401001699232656 w_k1_AXP_s106
                                 + 0.0267562214653215 w_k1_AXP_s107
                                 + 0.0511761503254568 w_k1_AXP_s110
                                 + 0.0447267509494613 w_k1_AXP_s111
                                 + 0.016232179805934 w_k1_AXP_s112
                                 + 0.0285014402550864 w_k1_AXP_s113
                                 + 0.0448458679128581 w_k1_AXP_s114
                                 + 0.017813680604228 w_k1_AXP_s115
                                 + 0.0112684585431141 w_k1_AXP_s116
                                 + 0.0451328368779682 w_k1_AXP_s117
                                 + 0.011581577764766 w_k1_DIS_s000
                                 + 0.0489037799145826 w_k1_DIS_s001
                                 + 0.00858993491552613 w_k1_DIS_s002
                                 + 0.0203960964221929 w_k1_DIS_s003
                                 + 0.0415888111911177 w_k1_DIS_s004
                                 + 0.0202504166615296 w_k1_DIS_s005
                                 + 0.0152982384500147 w_k1_DIS_s006
                                 + 0.0451712505783301 w_k1_DIS_s007
                                 + 0.0142568418571248 w_k1_DIS_s010
                                 + 0.00753708688890336 w_k1_DIS_s011
                                 + 0.0192015970753147 w_k1_DIS_s012
                                 + 0.0186340644869532 w_k1_DIS_s013
                                 + 0.0202540133541967 w_k1_DIS_s014
                                 + 0.0152877967539647 w_k1_DIS_s015
                                 + 0.00613207493459628 w_k1_DIS_s016
                                 + 0.0149795587412172 w_k1_DIS_s017
                                 + 0.0319719599069857 w_k1_DIS_s100
                                 + 0.0933740470329553 w_k1_DIS_s101
                                 + 0.0657454063858667 w_k1_DIS_s102
                                 + 0.044029427453381 w_k1_DIS_s103
                                 + 0.077346556574127 w_k1_DIS_s104
                                 + 0.0723189120448892 w_k1_DIS_s105
                                 + 0.0413040142872707 w_k1_DIS_s106
                                 + 0.0264211191274608 w_k1_DIS_s107
                                 + 0.0542621047167887 w_k1_DIS_s110
                                 + 0.0476555222839166 w_k1_DIS_s111
                                 + 0.0131496583600265 w_k1_DIS_s112
                                 + 0.0252116498063796 w_k1_DIS_s113
                                 + 0.0461586100051086 w_k1_DIS_s114
                                 + 0.0166921832330861 w_k1_DIS_s115
                                 + 0.0120295381667127 w_k1_DIS_s116
                                 + 0.0424898588258766 w_k1_DIS_s117
                                 + 0.0110416653804715 w_k1_FDX_s000
                                 + 0.0432856428778457 w_k1_FDX_s001
                                 + 0.00907776027759963 w_k1_FDX_s002
                                 + 0.0187777632439341 w_k1_FDX_s003
                                 + 0.0398630016138048 w_k1_FDX_s004
                                 + 0.0182380265182104 w_k1_FDX_s005
                                 + 0.015590761119803 w_k1_FDX_s006
                                 + 0.0412850119753701 w_k1_FDX_s007
                                 + 0.0134647918872818 w_k1_FDX_s010
                                 + 0.00798082403156382 w_k1_FDX_s011
                                 + 0.0187278189146901 w_k1_FDX_s012
                                 + 0.0192668338278973 w_k1_FDX_s013
                                 + 0.0195562329540221 w_k1_FDX_s014
                                 + 0.014479673330203 w_k1_FDX_s015
                                 + 0.00601428553667993 w_k1_FDX_s016
                                 + 0.0157015569611725 w_k1_FDX_s017
                                 + 0.0314499194972913 w_k1_FDX_s100
                                 + 0.0953121012297354 w_k1_FDX_s101
                                 + 0.0640263523963069 w_k1_FDX_s102
                                 + 0.0438130352489578 w_k1_FDX_s103
                                 + 0.073656168734172 w_k1_FDX_s104
                                 + 0.0757854521723569 w_k1_FDX_s105
                                 + 0.0412296047262224 w_k1_FDX_s106
                                 + 0.0260251942836257 w_k1_FDX_s107
                                 + 0.0496993931145395 w_k1_FDX_s110
                                 + 0.0478699697059188 w_k1_FDX_s111
                                 + 0.0143256968372894 w_k1_FDX_s112
                                 + 0.0278668349079577 w_k1_FDX_s113
                                 + 0.0460612796487142 w_k1_FDX_s114
                                 + 0.0161744368659935 w_k1_FDX_s115
                                 + 0.0117951250636369 w_k1_FDX_s116
                                 + 0.0438450111002861 w_k1_FDX_s117
                                 + 0.0113556033473126 w_k1_IBM_s000
                                 + 0.042760949956387 w_k1_IBM_s001
                                 + 0.00955025610944512 w_k1_IBM_s002
                                 + 0.0201470925620722 w_k1_IBM_s003
                                 + 0.0427999640639969 w_k1_IBM_s004
                                 + 0.0195776166310886 w_k1_IBM_s005
                                 + 0.014517101771419 w_k1_IBM_s006
                                 + 0.0420576884481462 w_k1_IBM_s007
                                 + 0.0140730170526228 w_k1_IBM_s010
                                 + 0.00757080712709982 w_k1_IBM_s011
                                 + 0.0195276489172637 w_k1_IBM_s012
                                 + 0.0187884523892708 w_k1_IBM_s013
                                 + 0.0206712912565903 w_k1_IBM_s014
                                 + 0.0141425505424802 w_k1_IBM_s015
                                 + 0.00575190270278159 w_k1_IBM_s016
                                 + 0.0141815668300223 w_k1_IBM_s017
                                 + 0.0314739891877709 w_k1_IBM_s100
                                 + 0.0999246623153354 w_k1_IBM_s101
                                 + 0.0659975482736465 w_k1_IBM_s102
                                 + 0.0439231639746504 w_k1_IBM_s103
                                 + 0.0709577841288679 w_k1_IBM_s104
                                 + 0.0743727205270275 w_k1_IBM_s105
                                 + 0.0438917834826714 w_k1_IBM_s106
                                 + 0.0253416384687937 w_k1_IBM_s107
                                 + 0.0509330276902833 w_k1_IBM_s110
                                 + 0.0488341687556912 w_k1_IBM_s111
                                 + 0.0151317879629465 w_k1_IBM_s112
                                 + 0.0271734197472157 w_k1_IBM_s113
                                 + 0.0455869681408163 w_k1_IBM_s114
                                 + 0.0158973313525563 w_k1_IBM_s115
                                 + 0.0117394875032433 w_k1_IBM_s116
                                 + 0.0441099039194673 w_k1_IBM_s117
                                 + 0.0122200689401578 w_k1_INTC_s000
                                 + 0.0421022121900759 w_k1_INTC_s001
                                 + 0.00835053912383741 w_k1_INTC_s002
                                 + 0.0186887307667081 w_k1_INTC_s003
                                 + 0.0405742795463048 w_k1_INTC_s004
                                 + 0.020111388225582 w_k1_INTC_s005
                                 + 0.014238981324567 w_k1_INTC_s006
                                 + 0.0334920529882036 w_k1_INTC_s007
                                 + 0.0156489535383415 w_k1_INTC_s010
                                 + 0.00718178593854124 w_k1_INTC_s011
                                 + 0.0191481383711148 w_k1_INTC_s012
                                 + 0.020126820244357 w_k1_INTC_s013
                                 + 0.018583961606671 w_k1_INTC_s014
                                 + 0.0137479406272945 w_k1_INTC_s015
                                 + 0.0054609720743481 w_k1_INTC_s016
                                 + 0.0165538578714429 w_k1_INTC_s017
                                 + 0.0306516792758424 w_k1_INTC_s100
                                 + 0.0878046360741036 w_k1_INTC_s101
                                 + 0.0621764531436824 w_k1_INTC_s102
                                 + 0.0460172813588532 w_k1_INTC_s103
                                 + 0.0695158120853111 w_k1_INTC_s104
                                 + 0.0697374294905414 w_k1_INTC_s105
                                 + 0.0410853036881204 w_k1_INTC_s106
                                 + 0.022465232192476 w_k1_INTC_s107
                                 + 0.0535468834573843 w_k1_INTC_s110
                                 + 0.047946504510123 w_k1_INTC_s111
                                 + 0.0162062105968167 w_k1_INTC_s112
                                 + 0.0322825919349485 w_k1_INTC_s113
                                 + 0.0455847853667246 w_k1_INTC_s114
                                 + 0.0148567599833953 w_k1_INTC_s115
                                 + 0.0115675730707508 w_k1_INTC_s116
                                 + 0.0388861910030303 w_k1_INTC_s117
                                 + 0.012401135419004 w_k1_KO_s000
                                 + 0.0419559945978427 w_k1_KO_s001
                                 + 0.00913433932605894 w_k1_KO_s002
                                 + 0.0181386986705721 w_k1_KO_s003
                                 + 0.044682897226336 w_k1_KO_s004
                                 + 0.0211148881833923 w_k1_KO_s005
                                 + 0.0162282978114828 w_k1_KO_s006
                                 + 0.0382580099803336 w_k1_KO_s007
                                 + 0.0138843718322714 w_k1_KO_s010
                                 + 0.00773635778491531 w_k1_KO_s011
                                 + 0.0192843354270228 w_k1_KO_s012
                                 + 0.0180681006177348 w_k1_KO_s013
                                 + 0.0211900991860689 w_k1_KO_s014
                                 + 0.0161076016986705 w_k1_KO_s015
                                 + 0.00594403880873709 w_k1_KO_s016
                                 + 0.013263981603322 w_k1_KO_s017
                                 + 0.0310721397034412 w_k1_KO_s100
                                 + 0.0839840182279373 w_k1_KO_s101
                                 + 0.0664742557076771 w_k1_KO_s102
                                 + 0.042241218639102 w_k1_KO_s103
                                 + 0.0708513114065122 w_k1_KO_s104
                                 + 0.0777648844084411 w_k1_KO_s105
                                 + 0.0441099602081014 w_k1_KO_s106
                                 + 0.0242834082206688 w_k1_KO_s107
                                 + 0.0522410579283933 w_k1_KO_s110
                                 + 0.0455053985911453 w_k1_KO_s111
                                 + 0.0153113087839264 w_k1_KO_s112
                                 + 0.0256220938289846 w_k1_KO_s113
                                 + 0.0429851321636601 w_k1_KO_s114
                                 + 0.014869151635113 w_k1_KO_s115
                                 + 0.0120715382439209 w_k1_KO_s116
                                 + 0.0409118885459737 w_k1_KO_s117
                                 + 0.0105543852346294 w_k1_MAT_s000
                                 + 0.04701987266369 w_k1_MAT_s001
                                 + 0.01247815189968 w_k1_MAT_s002
                                 + 0.0189794073173437 w_k1_MAT_s003
                                 + 0.0380368413097548 w_k1_MAT_s004
                                 + 0.0250968203208639 w_k1_MAT_s005
                                 + 0.0143628627616527 w_k1_MAT_s006
                                 + 0.0452490441914194 w_k1_MAT_s007
                                 + 0.0133480922272609 w_k1_MAT_s010
                                 + 0.00719426422531956 w_k1_MAT_s011
                                 + 0.0172661917524599 w_k1_MAT_s012
                                 + 0.0166681813169974 w_k1_MAT_s013
                                 + 0.022533520243625 w_k1_MAT_s014
                                 + 0.0183005172604301 w_k1_MAT_s015
                                 + 0.00477869768070954 w_k1_MAT_s016
                                 + 0.012832565289508 w_k1_MAT_s017
                                 + 0.0372631581403875 w_k1_MAT_s100
                                 + 0.0952502712549727 w_k1_MAT_s101
                                 + 0.0676479285998801 w_k1_MAT_s102
                                 + 0.0409073573322666 w_k1_MAT_s103
                                 + 0.0700113543181229 w_k1_MAT_s104
                                 + 0.085738562535707 w_k1_MAT_s105
                                 + 0.0360949564636725 w_k1_MAT_s106
                                 + 0.0273464871597662 w_k1_MAT_s107
                                 + 0.0450958861544492 w_k1_MAT_s110
                                 + 0.0478804345701201 w_k1_MAT_s111
                                 + 0.0132990305415365 w_k1_MAT_s112
                                 + 0.0266440651470232 w_k1_MAT_s113
                                 + 0.0435215386142669 w_k1_MAT_s114
                                 + 0.0182964799829243 w_k1_MAT_s115
                                 + 0.0117940432095205 w_k1_MAT_s116
                                 + 0.0473688470389249 w_k1_MAT_s117
                                 + 0.0119860309342307 w_k1_MSFT_s000
                                 + 0.0437121945968411 w_k1_MSFT_s001
                                 + 0.00860431259840851 w_k1_MSFT_s002
                                 + 0.0187168275226578 w_k1_MSFT_s003
                                 + 0.0401074929238558 w_k1_MSFT_s004
                                 + 0.0197381979871417 w_k1_MSFT_s005
                                 + 0.0151174454535058 w_k1_MSFT_s006
                                 + 0.0384673019846399 w_k1_MSFT_s007
                                 + 0.0138816047750665 w_k1_MSFT_s010
                                 + 0.00722391284633188 w_k1_MSFT_s011
                                 + 0.0185465593989009 w_k1_MSFT_s012
                                 + 0.0175078449051165 w_k1_MSFT_s013
                                 + 0.0202487199411115 w_k1_MSFT_s014
                                 + 0.0151551320623187 w_k1_MSFT_s015
                                 + 0.00613724897192543 w_k1_MSFT_s016
                                 + 0.0148291710619101 w_k1_MSFT_s017
                                 + 0.0308141424321936 w_k1_MSFT_s100
                                 + 0.0889898661337665 w_k1_MSFT_s101
                                 + 0.0628708224297808 w_k1_MSFT_s102
                                 + 0.0422360026402146 w_k1_MSFT_s103
                                 + 0.0723710068353826 w_k1_MSFT_s104
                                 + 0.07573207153042 w_k1_MSFT_s105
                                 + 0.0405505457010689 w_k1_MSFT_s106
                                 + 0.0245159510010489 w_k1_MSFT_s107
                                 + 0.0503243104920085 w_k1_MSFT_s110
                                 + 0.0462243152648786 w_k1_MSFT_s111
                                 + 0.0126796144567656 w_k1_MSFT_s112
                                 + 0.0306682846510764 w_k1_MSFT_s113
                                 + 0.0455088267148508 w_k1_MSFT_s114
                                 + 0.0150469117547291 w_k1_MSFT_s115
                                 + 0.0111174401080869 w_k1_MSFT_s116
                                 + 0.0400648509751721 w_k1_MSFT_s117
                                 + 0.0114547976647416 w_k1_PG_s000
                                 + 0.0407606711588509 w_k1_PG_s001
                                 + 0.00922305461532406 w_k1_PG_s002
                                 + 0.0195151911235625 w_k1_PG_s003
                                 + 0.0403551209257703 w_k1_PG_s004
                                 + 0.018123168382188 w_k1_PG_s005
                                 + 0.0155291141216908 w_k1_PG_s006
                                 + 0.0409574998506251 w_k1_PG_s007
                                 + 0.0135734890366617 w_k1_PG_s010
                                 + 0.00777169731720621 w_k1_PG_s011
                                 + 0.0187172535421869 w_k1_PG_s012
                                 + 0.0195270535754505 w_k1_PG_s013
                                 + 0.0202545946080909 w_k1_PG_s014
                                 + 0.013902876363589 w_k1_PG_s015
                                 + 0.00584410803947563 w_k1_PG_s016
                                 + 0.0148940968202346 w_k1_PG_s017
                                 + 0.0292178042771483 w_k1_PG_s100
                                 + 0.088258099594052 w_k1_PG_s101
                                 + 0.0627619554491451 w_k1_PG_s102
                                 + 0.0464230971589437 w_k1_PG_s103
                                 + 0.0694908262960257 w_k1_PG_s104
                                 + 0.0767396722801978 w_k1_PG_s105
                                 + 0.0439541978743075 w_k1_PG_s106
                                 + 0.0238372068031785 w_k1_PG_s107
                                 + 0.0488969348628651 w_k1_PG_s110
                                 + 0.0443316554783338 w_k1_PG_s111
                                 + 0.0136685283341786 w_k1_PG_s112
                                 + 0.0262378945209208 w_k1_PG_s113
                                 + 0.046643021323513 w_k1_PG_s114
                                 + 0.0169505170213585 w_k1_PG_s115
                                 + 0.0118618153455787 w_k1_PG_s116
                                 + 0.0411504119637007 w_k1_PG_s117 >= 1.05
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
