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
 w_k1_ADM_s0_return_constraint:  1.16078 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.91642 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.943344 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.10941 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.05767 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.911219 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.13356 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.752554 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.805285 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.27964 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.15919 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.901066 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.12958 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.768237 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.880614 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.869726 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.12637 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.17895 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.10742 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.51329 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.894024 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.24937 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.779422 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.910522 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.985608 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.12155 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.10789 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.30294 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.15002 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.37842 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.917634 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.19219 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.02312 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.737403 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.762619 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.09141 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.954069 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.18285 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.27346 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.916676 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.926156 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.21757 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.13787 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.785594 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.893156 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.693675 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.595896 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.43371 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.35088 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.981067 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.0363 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.706488 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.692328 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.969265 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.1992 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.417 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.949889 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.55688 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.958511 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.14106 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.574504 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.841811 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.803517 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.990821 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.23932 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.33417 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.30964 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.8165 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.921368 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.19482 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.00724 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.705116 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.840523 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.40479 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.927713 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.38923 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.34373 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.886203 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.87918 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.28593 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.21867 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.699747 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.83565 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.650129 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.575131 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.59206 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.4153 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.961416 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.01491 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.902108 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.631373 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.867518 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.43367 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.40368 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.995475 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.77501 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.813554 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.08348 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.711997 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.882896 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.71606 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.981447 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.13189 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.52385 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.26155 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 2.03979 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.870507 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.24903 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.08642 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.643081 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.12435 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.51231 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.763241 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.15989 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.20902 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.916193 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.913292 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.21371 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.12375 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.802542 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.01839 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.692745 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.752537 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.33243 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.15006 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.939591 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.05024 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.87269 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.795118 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.782739 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.26494 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.22037 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.12431 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.28119 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.910761 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.978948 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.802847 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.841142 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.882468 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.11738 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.06017 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.37744 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.995252 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.47309 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.870427 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.24184 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.27657 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.777723 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.942262 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.05491 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.923447 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.17319 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.08797 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.971031 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.04406 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.09089 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.14756 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.876944 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.617612 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.848144 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.509337 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.02651 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.77693 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.17733 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.880959 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.638417 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.639495 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.35826 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.963234 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.25607 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.667185 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.92916 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.903543 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.3161 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.548438 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.14165 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.693592 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.76585 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.39846 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.03643 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 2.14393 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.74409 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.30432 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.898992 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.526942 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.657198 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.03244 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.78828 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.86753 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.32463 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.0679 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.994201 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.913309 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.18514 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.17052 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.815737 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.15599 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.827114 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.958229 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.29301 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.17729 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.08686 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.07832 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.07065 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.896374 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.886177 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.14044 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.06227 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.979235 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.06383 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.1137 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.1044 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.737569 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.836014 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.729984 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.902501 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.968638 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.03582 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.704467 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.08523 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.846873 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.23235 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.14975 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.819294 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.951052 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.16259 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.764021 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.11443 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.18153 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.944986 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.914657 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.18666 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.18341 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.820972 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.760405 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.822191 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.653725 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.25266 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.4095 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.05975 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.00653 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.838376 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.671756 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.973376 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.10868 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.0155 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.78321 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.45353 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.821359 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.1005 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.581849 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.921948 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.692597 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.85079 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.11413 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.11412 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.36073 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.50413 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.982607 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.09814 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.739229 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.674913 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.01326 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.62944 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.90377 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.2698 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.24146 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.905311 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.06675 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.19237 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.17646 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.905953 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.487236 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.749883 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.436617 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.06359 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.82154 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.05179 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.775404 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.551585 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.40929 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.20067 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.07795 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.32353 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.602293 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 2.00092 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.76217 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.1069 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.399221 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.903943 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.624756 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.764422 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.41768 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.90347 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 2.42501 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 2.04965 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.17928 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.9617 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.632231 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.626832 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.03119 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.77774 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.02344 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.58423 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.1479 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.96026 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.90785 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.18372 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.10682 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.819721 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.942311 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.815956 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.823265 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.33521 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.24064 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.95542 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.03005 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.04118 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.840069 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.897483 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.26511 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.27073 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.07182 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.24443 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.936386 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.982705 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.804717 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.792333 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.840657 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.03652 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.932315 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.17433 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.948762 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.29333 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.885318 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.22856 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.16449 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.757491 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.985613 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.10219 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.859477 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.04978 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.09426 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.944345 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.936693 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.13961 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.05966 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.881718 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.930449 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.863652 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.802353 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.30766 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.21423 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.982203 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.08185 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.943368 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.762428 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.963086 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.06158 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.1101 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.936039 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.2979 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.907269 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.12566 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.830986 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.959676 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.944533 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.04776 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.21853 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.1943 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.23088 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.43357 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.962329 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.06637 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.01429 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.770171 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.995331 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.15582 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.954992 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.1928 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.495997200036117 z_k1_s0000
                                 - 0.323998492086757 z_k1_s0001
                                 - 0.13199943833142 z_k1_s0002
                                 - 0.54199870638687 z_k1_s0003
                                 - 0.579997230427206 z_k1_s0004
                                 - 0.35299863800692 z_k1_s0005
                                 - 0.24999906119686 z_k1_s0006
                                 - 0.166999261395172 z_k1_s0007
                                 - 0.327000100262152 z_k1_s0010
                                 - 0.272000908671012 z_k1_s0011
                                 - 0.32500045140443 z_k1_s0012
                                 - 0.217000168679209 z_k1_s0013
                                 - 0.363000415703987 z_k1_s0014
                                 - 0.132000050619588 z_k1_s0015
                                 - 0.367999537848291 z_k1_s0016
                                 - 0.207999979218702 z_k1_s0017
                                 - 0.310000266603084 z_k1_s0020
                                 - 0.352999581499374 z_k1_s0021
                                 - 0.215000368482852 z_k1_s0022
                                 - 0.2000007839377 z_k1_s0023
                                 - 0.299999278905848 z_k1_s0024
                                 - 0.162000065889326 z_k1_s0025
                                 - 0.160999967119603 z_k1_s0026
                                 - 0.196000009458665 z_k1_s0027
                                 - 0.29500006961704 z_k1_s0030
                                 - 0.116000111569784 z_k1_s0031
                                 - 0.408000803521469 z_k1_s0032
                                 - 0.168999979261942 z_k1_s0033
                                 - 0.343999508654902 z_k1_s0034
                                 - 0.140999959183031 z_k1_s0035
                                 - 0.159000000476628 z_k1_s0036
                                 - 0.354999544928771 z_k1_s0037
                                 - 0.852000959038871 z_k1_s0040
                                 - 0.311000847105941 z_k1_s0041
                                 - 0.274000674659055 z_k1_s0042
                                 - 0.773001720304053 z_k1_s0043
                                 - 0.173000333007102 z_k1_s0044
                                 - 0.826003527056613 z_k1_s0045
                                 - 0.370000928466655 z_k1_s0046
                                 - 0.401002030303183 z_k1_s0047
                                 - 0.815996243455457 z_k1_s0050
                                 - 0.54399896496725 z_k1_s0051
                                 - 1.05499750654595 z_k1_s0052
                                 - 0.490998936160048 z_k1_s0053
                                 - 0.835997501870675 z_k1_s0054
                                 - 0.353999131997475 z_k1_s0055
                                 - 1.07799619872918 z_k1_s0056
                                 - 0.335999101421487 z_k1_s0057
                                 - 0.441002267179184 z_k1_s0060
                                 - 0.43900214372914 z_k1_s0061
                                 - 0.204000307077156 z_k1_s0062
                                 - 0.317000755690122 z_k1_s0063
                                 - 0.709003453450915 z_k1_s0064
                                 - 0.313000508790034 z_k1_s0065
                                 - 0.857001071728547 z_k1_s0066
                                 - 0.559001872714697 z_k1_s0067
                                 - 0.161999720799079 z_k1_s0070
                                 - 0.0919999197192038 z_k1_s0071
                                 - 0.076000025572355 z_k1_s0072
                                 - 0.133999941267082 z_k1_s0073
                                 - 0.21499923806681 z_k1_s0074
                                 - 0.153000440810006 z_k1_s0075
                                 - 0.218000407062696 z_k1_s0076
                                 - 0.359000398887742 z_k1_s0077
                                 - 0.245999659355198 z_k1_s0100
                                 - 0.408000943543733 z_k1_s0101
                                 - 0.61200002231696 z_k1_s0102
                                 - 0.202999741549186 z_k1_s0103
                                 - 0.504000837789637 z_k1_s0104
                                 - 0.18399979730465 z_k1_s0105
                                 - 0.319999647486348 z_k1_s0106
                                 - 0.309000530228154 z_k1_s0107
                                 - 0.179000359395036 z_k1_s0110
                                 - 0.0919997014318469 z_k1_s0111
                                 - 0.0669999984863429 z_k1_s0112
                                 - 0.122000370246958 z_k1_s0113
                                 - 0.172000416938282 z_k1_s0114
                                 - 0.0450000420221919 z_k1_s0115
                                 - 0.070000065367854 z_k1_s0116
                                 - 0.0540000023666065 z_k1_s0117
                                 - 0.0820000198612711 z_k1_s0120
                                 - 0.0850000579781583 z_k1_s0121
                                 - 0.132000100342555 z_k1_s0122
                                 - 0.188000169457586 z_k1_s0123
                                 - 0.0780000274387727 z_k1_s0124
                                 - 0.0870000541894076 z_k1_s0125
                                 - 0.0760000312275235 z_k1_s0126
                                 - 0.148000245232602 z_k1_s0127
                                 - 0.387998684323055 z_k1_s0130
                                 - 0.188999536476956 z_k1_s0131
                                 - 0.235998964568309 z_k1_s0132
                                 - 0.349999704881628 z_k1_s0133
                                 - 0.234999041451285 z_k1_s0134
                                 - 0.12299966816763 z_k1_s0135
                                 - 0.185999386926981 z_k1_s0136
                                 - 0.194000292658787 z_k1_s0137
                                 - 0.403000873031953 z_k1_s0140
                                 - 0.514000120955305 z_k1_s0141
                                 - 0.17600010633574 z_k1_s0142
                                 - 0.114000054292452 z_k1_s0143
                                 - 0.252000142530707 z_k1_s0144
                                 - 0.242000306636817 z_k1_s0145
                                 - 0.259001097157596 z_k1_s0146
                                 - 0.17900027100414 z_k1_s0147
                                 - 0.201999955558436 z_k1_s0150
                                 - 0.129000405361755 z_k1_s0151
                                 - 0.0849999871607302 z_k1_s0152
                                 - 0.173999527975899 z_k1_s0153
                                 - 0.141000250325559 z_k1_s0154
                                 - 0.0610000604330235 z_k1_s0155
                                 - 0.195000144662924 z_k1_s0156
                                 - 0.196999921490143 z_k1_s0157
                                 - 0.180999951987962 z_k1_s0160
                                 - 0.279999990644325 z_k1_s0161
                                 - 0.134999903167852 z_k1_s0162
                                 - 0.225999841377291 z_k1_s0163
                                 - 0.46499893535635 z_k1_s0164
                                 - 0.242999261702454 z_k1_s0165
                                 - 0.505999362021883 z_k1_s0166
                                 - 0.313999771294117 z_k1_s0167
                                 - 0.188999617430025 z_k1_s0170
                                 - 0.23500057831773 z_k1_s0171
                                 - 0.124000003201703 z_k1_s0172
                                 - 0.103000045709478 z_k1_s0173
                                 - 0.202000087115676 z_k1_s0174
                                 - 0.177000385720164 z_k1_s0175
                                 - 0.083000024193075 z_k1_s0176
                                 - 0.188999617430025 z_k1_s0177
                                 - 0.359999809587875 z_k1_s1000
                                 - 0.489000262939387 z_k1_s1001
                                 - 0.644001272210607 z_k1_s1002
                                 - 0.330999821227462 z_k1_s1003
                                 - 0.782000910684279 z_k1_s1004
                                 - 0.214000204938918 z_k1_s1005
                                 - 0.952001301659131 z_k1_s1006
                                 - 0.666999732289483 z_k1_s1007
                                 - 2.21600087304696 z_k1_s1010
                                 - 1.07600028388719 z_k1_s1011
                                 - 1.17600560938357 z_k1_s1012
                                 - 0.854001649226935 z_k1_s1013
                                 - 0.862001687386007 z_k1_s1014
                                 - 1.61900408606119 z_k1_s1015
                                 - 1.13299934364359 z_k1_s1016
                                 - 0.761001448053123 z_k1_s1017
                                 - 0.291001228872088 z_k1_s1020
                                 - 1.30000592391155 z_k1_s1021
                                 - 0.28600139038092 z_k1_s1022
                                 - 0.571004381230409 z_k1_s1023
                                 - 0.321001566624765 z_k1_s1024
                                 - 0.909005311604768 z_k1_s1025
                                 - 0.302001483395303 z_k1_s1026
                                 - 0.376001532435172 z_k1_s1027
                                 - 1.74900161468239 z_k1_s1030
                                 - 0.858001249920704 z_k1_s1031
                                 - 0.521001024814457 z_k1_s1032
                                 - 1.01900134306401 z_k1_s1033
                                 - 0.384000508501458 z_k1_s1034
                                 - 1.07600569634815 z_k1_s1035
                                 - 1.15200071670254 z_k1_s1036
                                 - 1.3290069981474 z_k1_s1037
                                 - 0.438999897833705 z_k1_s1040
                                 - 0.300000196123318 z_k1_s1041
                                 - 0.54000184342229 z_k1_s1042
                                 - 1.15100139001993 z_k1_s1043
                                 - 0.729998076032893 z_k1_s1044
                                 - 0.766999747941787 z_k1_s1045
                                 - 0.797001506354491 z_k1_s1046
                                 - 0.243999894553575 z_k1_s1047
                                 - 0.46699875631764 z_k1_s1050
                                 - 0.792000180558307 z_k1_s1051
                                 - 0.519000404036357 z_k1_s1052
                                 - 0.49700175477105 z_k1_s1053
                                 - 0.5459995530439 z_k1_s1054
                                 - 0.606999438098245 z_k1_s1055
                                 - 0.56600155201284 z_k1_s1056
                                 - 0.442999907155439 z_k1_s1057
                                 - 0.354000824393561 z_k1_s1060
                                 - 0.278000138795147 z_k1_s1061
                                 - 0.317999422794743 z_k1_s1062
                                 - 0.188999992796741 z_k1_s1063
                                 - 0.126999874997858 z_k1_s1064
                                 - 0.29900027439473 z_k1_s1065
                                 - 0.134999936397696 z_k1_s1066
                                 - 0.34599892159446 z_k1_s1067
                                 - 0.206999367010936 z_k1_s1070
                                 - 0.185999942531188 z_k1_s1071
                                 - 0.405000380672373 z_k1_s1072
                                 - 0.331000635364095 z_k1_s1073
                                 - 0.1939998726186 z_k1_s1074
                                 - 0.259999687839466 z_k1_s1075
                                 - 0.194999863879527 z_k1_s1076
                                 - 0.181999781487626 z_k1_s1077
                                 - 0.146000048083471 z_k1_s1100
                                 - 0.370999362781392 z_k1_s1101
                                 - 0.37800007211744 z_k1_s1102
                                 - 0.167000052089995 z_k1_s1103
                                 - 0.265001061987865 z_k1_s1104
                                 - 0.208000080140844 z_k1_s1105
                                 - 0.466999987954538 z_k1_s1106
                                 - 0.122000104190346 z_k1_s1107
                                 - 0.787000457642186 z_k1_s1110
                                 - 0.285000325334736 z_k1_s1111
                                 - 0.776002807669654 z_k1_s1112
                                 - 0.724001174515223 z_k1_s1113
                                 - 0.397000608295193 z_k1_s1114
                                 - 1.30700061617184 z_k1_s1115
                                 - 0.50700028934153 z_k1_s1116
                                 - 0.608002922329457 z_k1_s1117
                                 - 0.187000414263973 z_k1_s1120
                                 - 0.132000090539174 z_k1_s1121
                                 - 0.145000045092247 z_k1_s1122
                                 - 0.0920001103758115 z_k1_s1123
                                 - 0.14100004707591 z_k1_s1124
                                 - 0.232000196947657 z_k1_s1125
                                 - 0.248999525516752 z_k1_s1126
                                 - 0.382000512560462 z_k1_s1127
                                 - 0.325999465923999 z_k1_s1130
                                 - 0.545001394262917 z_k1_s1131
                                 - 0.639999472623194 z_k1_s1132
                                 - 0.29599987406309 z_k1_s1133
                                 - 0.791998285597439 z_k1_s1134
                                 - 0.39599961469817 z_k1_s1135
                                 - 1.11800058291814 z_k1_s1136
                                 - 0.605999117221784 z_k1_s1137
                                 - 0.205999588751601 z_k1_s1140
                                 - 0.277000578977787 z_k1_s1141
                                 - 0.0750000386815944 z_k1_s1142
                                 - 0.251999679249711 z_k1_s1143
                                 - 0.101000035397864 z_k1_s1144
                                 - 0.231000488479676 z_k1_s1145
                                 - 0.27800061496691 z_k1_s1146
                                 - 0.145000210417883 z_k1_s1147
                                 - 0.28800114503578 z_k1_s1150
                                 - 0.414003196307786 z_k1_s1151
                                 - 0.442002711031042 z_k1_s1152
                                 - 0.165000813562811 z_k1_s1153
                                 - 0.807003731560735 z_k1_s1154
                                 - 0.290001283231006 z_k1_s1155
                                 - 0.339001038999184 z_k1_s1156
                                 - 0.280001197257353 z_k1_s1157
                                 - 0.204000285219115 z_k1_s1160
                                 - 0.480000158632248 z_k1_s1161
                                 - 0.22100033318742 z_k1_s1162
                                 - 0.227000298870255 z_k1_s1163
                                 - 0.382001590347584 z_k1_s1164
                                 - 0.287000536499695 z_k1_s1165
                                 - 0.788001881824322 z_k1_s1166
                                 - 0.315000376352924 z_k1_s1167
                                 - 0.329999995139161 z_k1_s1170
                                 - 0.0750000143861704 z_k1_s1171
                                 - 0.170999780608509 z_k1_s1172
                                 - 0.280999778988229 z_k1_s1173
                                 - 0.361999406013355 z_k1_s1174
                                 - 0.100999983021382 z_k1_s1175
                                 - 0.141000013414003 z_k1_s1176
                                 - 0.243000585075106 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.15082 w_k1_ADM_s000
                                 + 1.52124 w_k1_AXP_s000
                                 + 1.62317 w_k1_DIS_s000
                                 + 1.31916 w_k1_FDX_s000
                                 + 1.17578 w_k1_IBM_s000
                                 + 1.07356 w_k1_INTC_s000 + 1.3292 w_k1_KO_s000
                                 + 1.76581 w_k1_MAT_s000
                                 + 1.41869 w_k1_MSFT_s000
                                 + 1.18915 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.917343 w_k1_ADM_s000
                                 + 0.847027 w_k1_AXP_s000
                                 + 0.813178 w_k1_DIS_s000
                                 + 0.87233 w_k1_FDX_s000
                                 + 1.13986 w_k1_IBM_s000
                                 + 0.840015 w_k1_INTC_s000
                                 + 1.07856 w_k1_KO_s000 + 1.38564 w_k1_MAT_s000
                                 + 0.973168 w_k1_MSFT_s000
                                 + 1.02338 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.748456 w_k1_ADM_s000
                                 + 0.588681 w_k1_AXP_s000
                                 + 0.645916 w_k1_DIS_s000
                                 + 0.770177 w_k1_FDX_s000
                                 + 0.579777 w_k1_IBM_s000
                                 + 0.788285 w_k1_INTC_s000
                                 + 0.680708 w_k1_KO_s000
                                 + 0.412619 w_k1_MAT_s000
                                 + 0.880165 w_k1_MSFT_s000
                                 + 0.834853 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.718125 w_k1_ADM_s000
                                 + 1.02648 w_k1_AXP_s000
                                 + 1.06216 w_k1_DIS_s000
                                 + 0.951412 w_k1_FDX_s000
                                 + 1.17336 w_k1_IBM_s000
                                 + 1.12769 w_k1_INTC_s000
                                 + 1.22591 w_k1_KO_s000 + 1.45065 w_k1_MAT_s000
                                 + 1.19734 w_k1_MSFT_s000
                                 + 1.06149 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.987417 w_k1_ADM_s000
                                 + 1.1723 w_k1_AXP_s000 + 1.18518 w_k1_DIS_s000
                                 + 1.16873 w_k1_FDX_s000
                                 + 1.02521 w_k1_IBM_s000
                                 + 0.946663 w_k1_INTC_s000
                                 + 1.17832 w_k1_KO_s000 + 1.53174 w_k1_MAT_s000
                                 + 1.14283 w_k1_MSFT_s000
                                 + 1.11874 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.747863 w_k1_ADM_s000
                                 + 0.714698 w_k1_AXP_s000
                                 + 0.716471 w_k1_DIS_s000
                                 + 0.786208 w_k1_FDX_s000
                                 + 0.928576 w_k1_IBM_s000
                                 + 0.844165 w_k1_INTC_s000
                                 + 0.79218 w_k1_KO_s000
                                 + 0.882746 w_k1_MAT_s000
                                 + 0.850893 w_k1_MSFT_s000
                                 + 0.891443 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.98556 w_k1_ADM_s000
                                 + 0.935831 w_k1_AXP_s000
                                 + 0.811845 w_k1_DIS_s000
                                 + 0.997483 w_k1_FDX_s000
                                 + 0.685796 w_k1_IBM_s000
                                 + 0.881346 w_k1_INTC_s000
                                 + 0.770443 w_k1_KO_s000
                                 + 0.620854 w_k1_MAT_s000
                                 + 0.955859 w_k1_MSFT_s000
                                 + 0.967928 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.722578 w_k1_ADM_s000
                                 + 0.69132 w_k1_AXP_s000
                                 + 0.796379 w_k1_DIS_s000
                                 + 0.768805 w_k1_FDX_s000
                                 + 0.801524 w_k1_IBM_s000
                                 + 0.663244 w_k1_INTC_s000
                                 + 0.778842 w_k1_KO_s000
                                 + 0.835922 w_k1_MAT_s000
                                 + 0.904083 w_k1_MSFT_s000
                                 + 0.888971 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.00374 w_k1_ADM_s001
                                 + 1.11818 w_k1_AXP_s001
                                 + 1.21317 w_k1_DIS_s001
                                 + 1.06883 w_k1_FDX_s001
                                 + 1.68419 w_k1_IBM_s001
                                 + 1.18466 w_k1_INTC_s001
                                 + 1.27693 w_k1_KO_s001 + 1.50886 w_k1_MAT_s001
                                 + 0.989443 w_k1_MSFT_s001
                                 + 1.24182 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.23247 w_k1_ADM_s001
                                 + 1.16034 w_k1_AXP_s001
                                 + 1.25118 w_k1_DIS_s001 + 1.2581 w_k1_FDX_s001
                                 + 1.21856 w_k1_IBM_s001
                                 + 1.11465 w_k1_INTC_s001
                                 + 1.21168 w_k1_KO_s001 + 1.46236 w_k1_MAT_s001
                                 + 1.2054 w_k1_MSFT_s001 + 1.11191 w_k1_PG_s001
                                 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.02985 w_k1_ADM_s001
                                 + 0.956019 w_k1_AXP_s001
                                 + 0.891241 w_k1_DIS_s001
                                 + 1.09661 w_k1_FDX_s001
                                 + 0.763695 w_k1_IBM_s001
                                 + 1.10672 w_k1_INTC_s001
                                 + 0.914781 w_k1_KO_s001
                                 + 0.687878 w_k1_MAT_s001
                                 + 1.08815 w_k1_MSFT_s001
                                 + 0.929338 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.930749 w_k1_ADM_s001
                                 + 1.06093 w_k1_AXP_s001
                                 + 1.42807 w_k1_DIS_s001
                                 + 1.20476 w_k1_FDX_s001
                                 + 1.03324 w_k1_IBM_s001
                                 + 1.32368 w_k1_INTC_s001
                                 + 1.16902 w_k1_KO_s001
                                 + 0.849439 w_k1_MAT_s001
                                 + 1.21874 w_k1_MSFT_s001
                                 + 1.03937 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.727126 w_k1_ADM_s001
                                 + 0.784578 w_k1_AXP_s001
                                 + 0.725922 w_k1_DIS_s001
                                 + 0.853022 w_k1_FDX_s001
                                 + 1.07688 w_k1_IBM_s001
                                 + 0.997864 w_k1_INTC_s001
                                 + 0.8739 w_k1_KO_s001 + 0.827602 w_k1_MAT_s001
                                 + 0.870533 w_k1_MSFT_s001
                                 + 0.883515 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.957064 w_k1_ADM_s001
                                 + 0.733215 w_k1_AXP_s001
                                 + 1.08119 w_k1_DIS_s001 + 1.0177 w_k1_FDX_s001
                                 + 1.12462 w_k1_IBM_s001
                                 + 1.12013 w_k1_INTC_s001
                                 + 0.978047 w_k1_KO_s001
                                 + 0.724272 w_k1_MAT_s001
                                 + 0.948144 w_k1_MSFT_s001
                                 + 1.01691 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.923929 w_k1_ADM_s001
                                 + 0.732362 w_k1_AXP_s001
                                 + 0.737749 w_k1_DIS_s001
                                 + 0.822818 w_k1_FDX_s001
                                 + 0.772808 w_k1_IBM_s001
                                 + 0.838237 w_k1_INTC_s001
                                 + 0.733101 w_k1_KO_s001
                                 + 0.525755 w_k1_MAT_s001
                                 + 0.753143 w_k1_MSFT_s001
                                 + 0.841612 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.19263 w_k1_ADM_s001
                                 + 1.09048 w_k1_AXP_s001
                                 + 1.04312 w_k1_DIS_s001
                                 + 1.09316 w_k1_FDX_s001
                                 + 1.08854 w_k1_IBM_s001
                                 + 1.13107 w_k1_INTC_s001
                                 + 0.956732 w_k1_KO_s001
                                 + 1.08401 w_k1_MAT_s001
                                 + 0.961124 w_k1_MSFT_s001
                                 + 0.985434 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.08385 w_k1_ADM_s002
                                 + 1.26075 w_k1_AXP_s002
                                 + 0.855837 w_k1_DIS_s002
                                 + 0.97392 w_k1_FDX_s002
                                 + 0.945312 w_k1_IBM_s002
                                 + 0.817805 w_k1_INTC_s002
                                 + 0.970812 w_k1_KO_s002
                                 + 0.930124 w_k1_MAT_s002
                                 + 0.799512 w_k1_MSFT_s002
                                 + 0.940971 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.687987 w_k1_ADM_s002
                                 + 0.653861 w_k1_AXP_s002
                                 + 0.760099 w_k1_DIS_s002
                                 + 0.806219 w_k1_FDX_s002
                                 + 0.792965 w_k1_IBM_s002
                                 + 0.717244 w_k1_INTC_s002
                                 + 0.821122 w_k1_KO_s002 + 1.0267 w_k1_MAT_s002
                                 + 0.877355 w_k1_MSFT_s002
                                 + 0.871737 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.750499 w_k1_ADM_s002
                                 + 0.826215 w_k1_AXP_s002
                                 + 0.873896 w_k1_DIS_s002
                                 + 0.893562 w_k1_FDX_s002
                                 + 1.12979 w_k1_IBM_s002
                                 + 0.929503 w_k1_INTC_s002
                                 + 1.03366 w_k1_KO_s002 + 0.77222 w_k1_MAT_s002
                                 + 0.972097 w_k1_MSFT_s002
                                 + 0.932869 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.949436 w_k1_ADM_s002
                                 + 0.923377 w_k1_AXP_s002
                                 + 1.13059 w_k1_DIS_s002
                                 + 1.13228 w_k1_FDX_s002
                                 + 0.788024 w_k1_IBM_s002
                                 + 0.834028 w_k1_INTC_s002
                                 + 0.762413 w_k1_KO_s002
                                 + 0.665009 w_k1_MAT_s002
                                 + 1.00575 w_k1_MSFT_s002
                                 + 0.883911 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.955943 w_k1_ADM_s002
                                 + 1.33207 w_k1_AXP_s002
                                 + 1.35191 w_k1_DIS_s002
                                 + 1.20388 w_k1_FDX_s002
                                 + 1.54185 w_k1_IBM_s002
                                 + 1.05462 w_k1_INTC_s002
                                 + 1.28537 w_k1_KO_s002 + 1.73548 w_k1_MAT_s002
                                 + 1.02569 w_k1_MSFT_s002
                                 + 1.17303 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.983602 w_k1_ADM_s002
                                 + 1.12238 w_k1_AXP_s002
                                 + 0.781581 w_k1_DIS_s002
                                 + 0.896017 w_k1_FDX_s002
                                 + 1.27737 w_k1_IBM_s002
                                 + 0.773814 w_k1_INTC_s002
                                 + 0.886923 w_k1_KO_s002
                                 + 1.21939 w_k1_MAT_s002
                                 + 0.877727 w_k1_MSFT_s002
                                 + 1.02795 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.69697 w_k1_ADM_s002
                                 + 0.588841 w_k1_AXP_s002
                                 + 0.629485 w_k1_DIS_s002
                                 + 0.793875 w_k1_FDX_s002
                                 + 0.570214 w_k1_IBM_s002
                                 + 0.731033 w_k1_INTC_s002
                                 + 0.536676 w_k1_KO_s002
                                 + 0.53713 w_k1_MAT_s002
                                 + 0.805491 w_k1_MSFT_s002
                                 + 0.716911 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.33838 w_k1_ADM_s002
                                 + 1.46922 w_k1_AXP_s002 + 1.2372 w_k1_DIS_s002
                                 + 1.27404 w_k1_FDX_s002
                                 + 0.929284 w_k1_IBM_s002
                                 + 1.07593 w_k1_INTC_s002
                                 + 1.04007 w_k1_KO_s002 + 1.1554 w_k1_MAT_s002
                                 + 1.10007 w_k1_MSFT_s002
                                 + 1.13759 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.912553 w_k1_ADM_s003
                                 + 0.698072 w_k1_AXP_s003
                                 + 0.779931 w_k1_DIS_s003
                                 + 0.942966 w_k1_FDX_s003
                                 + 0.588498 w_k1_IBM_s003
                                 + 1.01138 w_k1_INTC_s003
                                 + 0.699266 w_k1_KO_s003
                                 + 0.505333 w_k1_MAT_s003
                                 + 0.933794 w_k1_MSFT_s003
                                 + 0.819476 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.930448 w_k1_ADM_s003
                                 + 1.25277 w_k1_AXP_s003
                                 + 0.867239 w_k1_DIS_s003
                                 + 1.06208 w_k1_FDX_s003
                                 + 0.91748 w_k1_IBM_s003
                                 + 0.952833 w_k1_INTC_s003
                                 + 0.840971 w_k1_KO_s003
                                 + 0.881741 w_k1_MAT_s003
                                 + 0.945621 w_k1_MSFT_s003
                                 + 0.905106 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.29554 w_k1_ADM_s003
                                 + 1.42317 w_k1_AXP_s003 + 1.4987 w_k1_DIS_s003
                                 + 1.39151 w_k1_FDX_s003
                                 + 1.30636 w_k1_IBM_s003
                                 + 1.15796 w_k1_INTC_s003
                                 + 1.47513 w_k1_KO_s003 + 1.70848 w_k1_MAT_s003
                                 + 1.34727 w_k1_MSFT_s003
                                 + 1.12069 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.790922 w_k1_ADM_s003
                                 + 0.619752 w_k1_AXP_s003
                                 + 0.551485 w_k1_DIS_s003
                                 + 0.702507 w_k1_FDX_s003
                                 + 0.966654 w_k1_IBM_s003
                                 + 0.730861 w_k1_INTC_s003
                                 + 0.769075 w_k1_KO_s003
                                 + 0.737352 w_k1_MAT_s003
                                 + 0.776813 w_k1_MSFT_s003
                                 + 0.709872 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.00736 w_k1_ADM_s003
                                 + 1.072 w_k1_AXP_s003 + 1.10963 w_k1_DIS_s003
                                 + 1.09606 w_k1_FDX_s003 + 1.0734 w_k1_IBM_s003
                                 + 1.20992 w_k1_INTC_s003
                                 + 1.22994 w_k1_KO_s003 + 1.00956 w_k1_MAT_s003
                                 + 1.19468 w_k1_MSFT_s003 + 1.0074 w_k1_PG_s003
                                 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.909353 w_k1_ADM_s003
                                 + 0.895678 w_k1_AXP_s003
                                 + 0.641028 w_k1_DIS_s003
                                 + 0.863708 w_k1_FDX_s003
                                 + 1.12314 w_k1_IBM_s003
                                 + 1.06011 w_k1_INTC_s003
                                 + 0.886771 w_k1_KO_s003 + 1.2161 w_k1_MAT_s003
                                 + 1.0014 w_k1_MSFT_s003
                                 + 0.839019 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.02604 w_k1_ADM_s003
                                 + 0.755155 w_k1_AXP_s003
                                 + 0.937744 w_k1_DIS_s003
                                 + 0.916524 w_k1_FDX_s003
                                 + 1.39943 w_k1_IBM_s003
                                 + 0.865985 w_k1_INTC_s003
                                 + 1.19628 w_k1_KO_s003 + 1.26687 w_k1_MAT_s003
                                 + 0.924687 w_k1_MSFT_s003
                                 + 0.953899 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.967374 w_k1_ADM_s003
                                 + 0.763186 w_k1_AXP_s003
                                 + 0.956705 w_k1_DIS_s003
                                 + 0.981101 w_k1_FDX_s003
                                 + 0.82056 w_k1_IBM_s003
                                 + 0.880854 w_k1_INTC_s003
                                 + 0.84163 w_k1_KO_s003 + 0.5654 w_k1_MAT_s003
                                 + 1.08241 w_k1_MSFT_s003
                                 + 0.90132 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.966235 w_k1_ADM_s004
                                 + 0.805623 w_k1_AXP_s004
                                 + 0.886938 w_k1_DIS_s004
                                 + 0.888305 w_k1_FDX_s004
                                 + 0.977645 w_k1_IBM_s004
                                 + 0.885996 w_k1_INTC_s004
                                 + 0.987843 w_k1_KO_s004
                                 + 0.651187 w_k1_MAT_s004
                                 + 0.819581 w_k1_MSFT_s004
                                 + 0.813349 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.10802 w_k1_ADM_s004
                                 + 1.59954 w_k1_AXP_s004
                                 + 1.92445 w_k1_DIS_s004
                                 + 1.44559 w_k1_FDX_s004
                                 + 1.18901 w_k1_IBM_s004
                                 + 1.50908 w_k1_INTC_s004
                                 + 1.49002 w_k1_KO_s004 + 1.20889 w_k1_MAT_s004
                                 + 1.39302 w_k1_MSFT_s004 + 1.21 w_k1_PG_s004
                                 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.10185 w_k1_ADM_s004
                                 + 1.1093 w_k1_AXP_s004 + 1.43154 w_k1_DIS_s004
                                 + 1.29855 w_k1_FDX_s004
                                 + 0.936031 w_k1_IBM_s004
                                 + 1.29182 w_k1_INTC_s004
                                 + 1.15442 w_k1_KO_s004
                                 + 0.899631 w_k1_MAT_s004
                                 + 1.17985 w_k1_MSFT_s004
                                 + 1.08244 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.960086 w_k1_ADM_s004
                                 + 0.885356 w_k1_AXP_s004
                                 + 1.06706 w_k1_DIS_s004
                                 + 1.07095 w_k1_FDX_s004
                                 + 0.794329 w_k1_IBM_s004
                                 + 1.28874 w_k1_INTC_s004
                                 + 1.03566 w_k1_KO_s004
                                 + 0.667042 w_k1_MAT_s004
                                 + 1.17147 w_k1_MSFT_s004
                                 + 1.09471 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.33476 w_k1_ADM_s004
                                 + 1.65097 w_k1_AXP_s004
                                 + 1.25051 w_k1_DIS_s004
                                 + 1.28004 w_k1_FDX_s004
                                 + 1.32394 w_k1_IBM_s004
                                 + 1.29549 w_k1_INTC_s004
                                 + 1.27236 w_k1_KO_s004 + 1.33233 w_k1_MAT_s004
                                 + 1.21385 w_k1_MSFT_s004
                                 + 1.24024 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.807063 w_k1_ADM_s004
                                 + 0.694822 w_k1_AXP_s004
                                 + 0.671593 w_k1_DIS_s004
                                 + 0.781599 w_k1_FDX_s004
                                 + 1.07655 w_k1_IBM_s004
                                 + 0.749302 w_k1_INTC_s004
                                 + 0.919203 w_k1_KO_s004
                                 + 0.992119 w_k1_MAT_s004
                                 + 0.846086 w_k1_MSFT_s004
                                 + 0.95511 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.00379 w_k1_ADM_s004
                                 + 1.2573 w_k1_AXP_s004 + 1.70464 w_k1_DIS_s004
                                 + 1.14141 w_k1_FDX_s004
                                 + 1.65395 w_k1_IBM_s004
                                 + 1.36748 w_k1_INTC_s004
                                 + 1.72503 w_k1_KO_s004 + 1.52762 w_k1_MAT_s004
                                 + 1.2297 w_k1_MSFT_s004 + 1.19709 w_k1_PG_s004
                                 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.41143 w_k1_ADM_s004
                                 + 1.23945 w_k1_AXP_s004
                                 + 0.956943 w_k1_DIS_s004
                                 + 1.07174 w_k1_FDX_s004
                                 + 0.890689 w_k1_IBM_s004
                                 + 1.08104 w_k1_INTC_s004
                                 + 1.04796 w_k1_KO_s004 + 1.0759 w_k1_MAT_s004
                                 + 0.960395 w_k1_MSFT_s004
                                 + 1.12417 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.06507 w_k1_ADM_s005
                                 + 0.911991 w_k1_AXP_s005
                                 + 1.1173 w_k1_DIS_s005
                                 + 0.979565 w_k1_FDX_s005
                                 + 1.10464 w_k1_IBM_s005
                                 + 0.905375 w_k1_INTC_s005
                                 + 1.19379 w_k1_KO_s005 + 1.1032 w_k1_MAT_s005
                                 + 1.05615 w_k1_MSFT_s005
                                 + 1.16505 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.933153 w_k1_ADM_s005
                                 + 1.01217 w_k1_AXP_s005
                                 + 0.912983 w_k1_DIS_s005
                                 + 0.967214 w_k1_FDX_s005
                                 + 1.49338 w_k1_IBM_s005
                                 + 0.975884 w_k1_INTC_s005
                                 + 1.1213 w_k1_KO_s005 + 1.51404 w_k1_MAT_s005
                                 + 0.917074 w_k1_MSFT_s005
                                 + 1.00912 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.830383 w_k1_ADM_s005
                                 + 0.874525 w_k1_AXP_s005
                                 + 0.739942 w_k1_DIS_s005
                                 + 0.83973 w_k1_FDX_s005
                                 + 1.02101 w_k1_IBM_s005
                                 + 0.892487 w_k1_INTC_s005
                                 + 0.880148 w_k1_KO_s005
                                 + 0.657683 w_k1_MAT_s005
                                 + 0.884833 w_k1_MSFT_s005
                                 + 0.917705 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.40067 w_k1_ADM_s005
                                 + 1.24272 w_k1_AXP_s005
                                 + 1.59357 w_k1_DIS_s005
                                 + 1.26743 w_k1_FDX_s005
                                 + 1.45127 w_k1_IBM_s005
                                 + 1.00144 w_k1_INTC_s005
                                 + 1.48968 w_k1_KO_s005 + 1.67467 w_k1_MAT_s005
                                 + 1.29432 w_k1_MSFT_s005
                                 + 1.26746 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.26518 w_k1_ADM_s005
                                 + 1.35798 w_k1_AXP_s005
                                 + 1.42471 w_k1_DIS_s005 + 1.3245 w_k1_FDX_s005
                                 + 1.03438 w_k1_IBM_s005
                                 + 1.29162 w_k1_INTC_s005
                                 + 1.18335 w_k1_KO_s005 + 1.18199 w_k1_MAT_s005
                                 + 1.34349 w_k1_MSFT_s005
                                 + 1.24931 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.777441 w_k1_ADM_s005
                                 + 0.643368 w_k1_AXP_s005
                                 + 0.605242 w_k1_DIS_s005
                                 + 0.711316 w_k1_FDX_s005
                                 + 0.692803 w_k1_IBM_s005
                                 + 0.820792 w_k1_INTC_s005
                                 + 0.754082 w_k1_KO_s005
                                 + 0.830905 w_k1_MAT_s005
                                 + 0.818226 w_k1_MSFT_s005
                                 + 0.828169 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.945979 w_k1_ADM_s005
                                 + 0.829773 w_k1_AXP_s005
                                 + 0.932498 w_k1_DIS_s005
                                 + 0.944057 w_k1_FDX_s005
                                 + 0.754611 w_k1_IBM_s005
                                 + 1.00122 w_k1_INTC_s005
                                 + 0.98394 w_k1_KO_s005
                                 + 0.777924 w_k1_MAT_s005
                                 + 1.06096 w_k1_MSFT_s005
                                 + 1.02431 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.12253 w_k1_ADM_s005
                                 + 1.30359 w_k1_AXP_s005
                                 + 1.11921 w_k1_DIS_s005
                                 + 1.05111 w_k1_FDX_s005
                                 + 1.02454 w_k1_IBM_s005
                                 + 0.993629 w_k1_INTC_s005
                                 + 0.97381 w_k1_KO_s005 + 1.40613 w_k1_MAT_s005
                                 + 1.07974 w_k1_MSFT_s005
                                 + 1.16342 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.02309 w_k1_ADM_s006
                                 + 0.984066 w_k1_AXP_s006
                                 + 0.939076 w_k1_DIS_s006
                                 + 0.930936 w_k1_FDX_s006
                                 + 1.25764 w_k1_IBM_s006
                                 + 1.08232 w_k1_INTC_s006
                                 + 1.10712 w_k1_KO_s006 + 1.14141 w_k1_MAT_s006
                                 + 1.00053 w_k1_MSFT_s006
                                 + 0.97098 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.699599 w_k1_ADM_s006
                                 + 0.771119 w_k1_AXP_s006
                                 + 0.717514 w_k1_DIS_s006
                                 + 0.8116 w_k1_FDX_s006 + 1.05817 w_k1_IBM_s006
                                 + 0.899755 w_k1_INTC_s006
                                 + 0.738483 w_k1_KO_s006
                                 + 1.08948 w_k1_MAT_s006
                                 + 0.82431 w_k1_MSFT_s006
                                 + 0.849313 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.0922 w_k1_ADM_s006
                                 + 1.40376 w_k1_AXP_s006
                                 + 1.54452 w_k1_DIS_s006
                                 + 1.20875 w_k1_FDX_s006
                                 + 1.58651 w_k1_IBM_s006
                                 + 1.30195 w_k1_INTC_s006 + 1.553 w_k1_KO_s006
                                 + 1.65279 w_k1_MAT_s006
                                 + 1.25474 w_k1_MSFT_s006
                                 + 1.23105 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.52885 w_k1_ADM_s006
                                 + 1.34541 w_k1_AXP_s006
                                 + 1.65798 w_k1_DIS_s006
                                 + 1.40048 w_k1_FDX_s006 + 1.1681 w_k1_IBM_s006
                                 + 1.21059 w_k1_INTC_s006
                                 + 1.25644 w_k1_KO_s006 + 1.43343 w_k1_MAT_s006
                                 + 1.50141 w_k1_MSFT_s006
                                 + 1.23059 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.11727 w_k1_ADM_s006
                                 + 1.04008 w_k1_AXP_s006
                                 + 1.01087 w_k1_DIS_s006
                                 + 1.11306 w_k1_FDX_s006
                                 + 0.849115 w_k1_IBM_s006
                                 + 1.15219 w_k1_INTC_s006
                                 + 0.961232 w_k1_KO_s006
                                 + 0.973012 w_k1_MAT_s006
                                 + 1.05794 w_k1_MSFT_s006
                                 + 1.00102 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.19032 w_k1_ADM_s006
                                 + 1.47034 w_k1_AXP_s006
                                 + 1.12426 w_k1_DIS_s006 + 1.1145 w_k1_FDX_s006
                                 + 1.17248 w_k1_IBM_s006
                                 + 1.17099 w_k1_INTC_s006
                                 + 1.06975 w_k1_KO_s006 + 2.02636 w_k1_MAT_s006
                                 + 1.06367 w_k1_MSFT_s006
                                 + 1.12028 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.853529 w_k1_ADM_s006
                                 + 0.670375 w_k1_AXP_s006
                                 + 0.692528 w_k1_DIS_s006
                                 + 0.813618 w_k1_FDX_s006
                                 + 0.789579 w_k1_IBM_s006
                                 + 0.797536 w_k1_INTC_s006
                                 + 0.757384 w_k1_KO_s006
                                 + 0.750266 w_k1_MAT_s006
                                 + 0.869469 w_k1_MSFT_s006
                                 + 0.82531 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.810395 w_k1_ADM_s006
                                 + 0.873853 w_k1_AXP_s006
                                 + 1.35347 w_k1_DIS_s006
                                 + 1.00984 w_k1_FDX_s006
                                 + 1.01862 w_k1_IBM_s006
                                 + 1.1145 w_k1_INTC_s006 + 1.18428 w_k1_KO_s006
                                 + 1.0894 w_k1_MAT_s006
                                 + 1.18064 w_k1_MSFT_s006
                                 + 1.12417 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.922207 w_k1_ADM_s007
                                 + 0.836146 w_k1_AXP_s007
                                 + 0.8021 w_k1_DIS_s007
                                 + 0.849559 w_k1_FDX_s007
                                 + 0.976191 w_k1_IBM_s007
                                 + 0.82959 w_k1_INTC_s007
                                 + 0.909515 w_k1_KO_s007
                                 + 1.01677 w_k1_MAT_s007
                                 + 0.786362 w_k1_MSFT_s007
                                 + 0.865654 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.13879 w_k1_ADM_s007
                                 + 0.64879 w_k1_AXP_s007
                                 + 0.691895 w_k1_DIS_s007
                                 + 0.887241 w_k1_FDX_s007
                                 + 0.619357 w_k1_IBM_s007
                                 + 0.860162 w_k1_INTC_s007
                                 + 0.682461 w_k1_KO_s007
                                 + 0.39319 w_k1_MAT_s007
                                 + 0.779643 w_k1_MSFT_s007
                                 + 0.835709 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.986131 w_k1_ADM_s007
                                 + 0.775526 w_k1_AXP_s007
                                 + 0.94091 w_k1_DIS_s007
                                 + 0.955163 w_k1_FDX_s007
                                 + 0.906888 w_k1_IBM_s007
                                 + 0.928981 w_k1_INTC_s007
                                 + 0.835787 w_k1_KO_s007
                                 + 0.738761 w_k1_MAT_s007
                                 + 0.908522 w_k1_MSFT_s007
                                 + 0.792981 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.19298 w_k1_ADM_s007
                                 + 1.34552 w_k1_AXP_s007
                                 + 1.63459 w_k1_DIS_s007
                                 + 1.26784 w_k1_FDX_s007
                                 + 1.40995 w_k1_IBM_s007
                                 + 1.05005 w_k1_INTC_s007
                                 + 1.42187 w_k1_KO_s007 + 1.69847 w_k1_MAT_s007
                                 + 1.19626 w_k1_MSFT_s007
                                 + 1.12231 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.21098 w_k1_ADM_s007
                                 + 1.06248 w_k1_AXP_s007
                                 + 1.21644 w_k1_DIS_s007 + 1.1785 w_k1_FDX_s007
                                 + 1.10066 w_k1_IBM_s007
                                 + 0.943581 w_k1_INTC_s007
                                 + 1.05756 w_k1_KO_s007 + 1.31435 w_k1_MAT_s007
                                 + 0.978542 w_k1_MSFT_s007
                                 + 1.07227 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.691097 w_k1_ADM_s007
                                 + 0.539848 w_k1_AXP_s007
                                 + 0.626374 w_k1_DIS_s007
                                 + 0.693932 w_k1_FDX_s007
                                 + 0.81433 w_k1_IBM_s007
                                 + 0.725278 w_k1_INTC_s007
                                 + 0.756994 w_k1_KO_s007
                                 + 0.507015 w_k1_MAT_s007
                                 + 0.687509 w_k1_MSFT_s007
                                 + 0.78223 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.08989 w_k1_ADM_s007
                                 + 0.918816 w_k1_AXP_s007
                                 + 0.959075 w_k1_DIS_s007
                                 + 0.878665 w_k1_FDX_s007
                                 + 1.39314 w_k1_IBM_s007
                                 + 0.689952 w_k1_INTC_s007
                                 + 1.18706 w_k1_KO_s007 + 1.02292 w_k1_MAT_s007
                                 + 0.859881 w_k1_MSFT_s007
                                 + 0.965842 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.22789 w_k1_ADM_s007
                                 + 0.999606 w_k1_AXP_s007
                                 + 1.00559 w_k1_DIS_s007
                                 + 1.05918 w_k1_FDX_s007
                                 + 0.845809 w_k1_IBM_s007
                                 + 0.867533 w_k1_INTC_s007
                                 + 0.902785 w_k1_KO_s007
                                 + 0.879924 w_k1_MAT_s007
                                 + 0.944699 w_k1_MSFT_s007
                                 + 0.966416 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.07341 w_k1_ADM_s010
                                 + 1.09108 w_k1_AXP_s010
                                 + 1.37825 w_k1_DIS_s010 + 1.0558 w_k1_FDX_s010
                                 + 1.80008 w_k1_IBM_s010
                                 + 1.0725 w_k1_INTC_s010 + 1.62802 w_k1_KO_s010
                                 + 2.66201 w_k1_MAT_s010
                                 + 1.21941 w_k1_MSFT_s010
                                 + 1.28214 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.42977 w_k1_ADM_s010
                                 + 1.49562 w_k1_AXP_s010
                                 + 1.63721 w_k1_DIS_s010
                                 + 1.30476 w_k1_FDX_s010
                                 + 1.28946 w_k1_IBM_s010
                                 + 1.11303 w_k1_INTC_s010 + 1.3154 w_k1_KO_s010
                                 + 1.86123 w_k1_MAT_s010
                                 + 1.22334 w_k1_MSFT_s010
                                 + 1.24531 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.13205 w_k1_ADM_s010
                                 + 1.21486 w_k1_AXP_s010
                                 + 1.08333 w_k1_DIS_s010 + 1.0699 w_k1_FDX_s010
                                 + 1.02084 w_k1_IBM_s010
                                 + 0.915065 w_k1_INTC_s010
                                 + 1.0224 w_k1_KO_s010 + 1.26502 w_k1_MAT_s010
                                 + 0.887671 w_k1_MSFT_s010
                                 + 1.03641 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.708073 w_k1_ADM_s010
                                 + 0.728053 w_k1_AXP_s010
                                 + 0.712461 w_k1_DIS_s010
                                 + 0.82421 w_k1_FDX_s010
                                 + 0.514534 w_k1_IBM_s010
                                 + 0.979365 w_k1_INTC_s010
                                 + 0.622537 w_k1_KO_s010
                                 + 0.533494 w_k1_MAT_s010
                                 + 0.848419 w_k1_MSFT_s010
                                 + 0.90362 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.843651 w_k1_ADM_s010
                                 + 0.759211 w_k1_AXP_s010
                                 + 0.694999 w_k1_DIS_s010
                                 + 0.766145 w_k1_FDX_s010
                                 + 0.983246 w_k1_IBM_s010
                                 + 0.792284 w_k1_INTC_s010
                                 + 0.841673 w_k1_KO_s010
                                 + 1.15677 w_k1_MAT_s010
                                 + 0.701258 w_k1_MSFT_s010
                                 + 0.890949 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.20382 w_k1_ADM_s010
                                 + 0.974553 w_k1_AXP_s010
                                 + 1.08697 w_k1_DIS_s010
                                 + 1.18967 w_k1_FDX_s010
                                 + 0.65278 w_k1_IBM_s010
                                 + 1.02718 w_k1_INTC_s010
                                 + 0.787966 w_k1_KO_s010
                                 + 0.680431 w_k1_MAT_s010
                                 + 1.21535 w_k1_MSFT_s010
                                 + 0.99966 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.0142 w_k1_ADM_s010
                                 + 0.882762 w_k1_AXP_s010
                                 + 0.933206 w_k1_DIS_s010
                                 + 0.932595 w_k1_FDX_s010
                                 + 0.777036 w_k1_IBM_s010
                                 + 1.03862 w_k1_INTC_s010
                                 + 0.860649 w_k1_KO_s010
                                 + 0.649329 w_k1_MAT_s010
                                 + 0.939593 w_k1_MSFT_s010
                                 + 0.978242 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.14007 w_k1_ADM_s010
                                 + 1.12476 w_k1_AXP_s010
                                 + 1.04734 w_k1_DIS_s010
                                 + 1.01187 w_k1_FDX_s010
                                 + 1.34066 w_k1_IBM_s010
                                 + 0.847745 w_k1_INTC_s010
                                 + 1.15815 w_k1_KO_s010 + 2.10897 w_k1_MAT_s010
                                 + 0.906325 w_k1_MSFT_s010
                                 + 1.14951 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.21005 w_k1_ADM_s011
                                 + 1.10219 w_k1_AXP_s011
                                 + 1.04752 w_k1_DIS_s011
                                 + 1.21048 w_k1_FDX_s011
                                 + 0.98226 w_k1_IBM_s011
                                 + 0.946631 w_k1_INTC_s011
                                 + 0.916056 w_k1_KO_s011
                                 + 0.973556 w_k1_MAT_s011
                                 + 1.01381 w_k1_MSFT_s011
                                 + 0.956063 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.09345 w_k1_ADM_s011
                                 + 1.09349 w_k1_AXP_s011
                                 + 1.34847 w_k1_DIS_s011
                                 + 1.25728 w_k1_FDX_s011
                                 + 1.01638 w_k1_IBM_s011
                                 + 1.25851 w_k1_INTC_s011
                                 + 1.14377 w_k1_KO_s011
                                 + 0.815703 w_k1_MAT_s011
                                 + 1.2718 w_k1_MSFT_s011 + 1.1334 w_k1_PG_s011
                                 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.772143 w_k1_ADM_s011
                                 + 0.797084 w_k1_AXP_s011
                                 + 0.641921 w_k1_DIS_s011
                                 + 0.904664 w_k1_FDX_s011
                                 + 0.716232 w_k1_IBM_s011
                                 + 0.872594 w_k1_INTC_s011
                                 + 0.7123 w_k1_KO_s011 + 0.85218 w_k1_MAT_s011
                                 + 0.764587 w_k1_MSFT_s011
                                 + 0.842338 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.731977 w_k1_ADM_s011
                                 + 1.0559 w_k1_AXP_s011
                                 + 0.817864 w_k1_DIS_s011
                                 + 0.967708 w_k1_FDX_s011
                                 + 1.3956 w_k1_IBM_s011
                                 + 0.874434 w_k1_INTC_s011
                                 + 1.13226 w_k1_KO_s011 + 1.10039 w_k1_MAT_s011
                                 + 0.888004 w_k1_MSFT_s011
                                 + 0.967101 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.20844 w_k1_ADM_s011
                                 + 0.779301 w_k1_AXP_s011
                                 + 0.801642 w_k1_DIS_s011
                                 + 0.875667 w_k1_FDX_s011
                                 + 0.945663 w_k1_IBM_s011
                                 + 0.90899 w_k1_INTC_s011
                                 + 0.983983 w_k1_KO_s011
                                 + 0.859369 w_k1_MAT_s011
                                 + 0.941595 w_k1_MSFT_s011
                                 + 0.972622 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.03172 w_k1_ADM_s011
                                 + 1.61075 w_k1_AXP_s011
                                 + 0.983913 w_k1_DIS_s011
                                 + 1.30695 w_k1_FDX_s011
                                 + 1.34113 w_k1_IBM_s011
                                 + 1.20152 w_k1_INTC_s011
                                 + 1.22063 w_k1_KO_s011 + 2.09508 w_k1_MAT_s011
                                 + 1.16146 w_k1_MSFT_s011 + 1.0718 w_k1_PG_s011
                                 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.23807 w_k1_ADM_s011
                                 + 1.33166 w_k1_AXP_s011
                                 + 1.74551 w_k1_DIS_s011
                                 + 1.32272 w_k1_FDX_s011
                                 + 1.50813 w_k1_IBM_s011
                                 + 1.31377 w_k1_INTC_s011
                                 + 1.52239 w_k1_KO_s011 + 1.22546 w_k1_MAT_s011
                                 + 1.11321 w_k1_MSFT_s011 + 1.338 w_k1_PG_s011
                                 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.1901 w_k1_ADM_s011
                                 + 0.692473 w_k1_AXP_s011
                                 + 0.782896 w_k1_DIS_s011
                                 + 0.996726 w_k1_FDX_s011
                                 + 0.625136 w_k1_IBM_s011
                                 + 1.07537 w_k1_INTC_s011
                                 + 0.701095 w_k1_KO_s011
                                 + 0.269976 w_k1_MAT_s011
                                 + 0.893675 w_k1_MSFT_s011
                                 + 0.847176 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.19745 w_k1_ADM_s012
                                 + 1.28267 w_k1_AXP_s012
                                 + 1.08047 w_k1_DIS_s012 + 1.1799 w_k1_FDX_s012
                                 + 1.12844 w_k1_IBM_s012
                                 + 1.1817 w_k1_INTC_s012 + 1.0214 w_k1_KO_s012
                                 + 1.37692 w_k1_MAT_s012
                                 + 1.08888 w_k1_MSFT_s012
                                 + 1.16508 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.264 w_k1_ADM_s012
                                 + 0.957724 w_k1_AXP_s012
                                 + 1.39524 w_k1_DIS_s012 + 1.0806 w_k1_FDX_s012
                                 + 1.02877 w_k1_IBM_s012
                                 + 1.09487 w_k1_INTC_s012
                                 + 1.24706 w_k1_KO_s012 + 1.05732 w_k1_MAT_s012
                                 + 1.13645 w_k1_MSFT_s012
                                 + 1.18238 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.31312 w_k1_ADM_s012
                                 + 1.231 w_k1_AXP_s012 + 1.55987 w_k1_DIS_s012
                                 + 1.44851 w_k1_FDX_s012
                                 + 1.10331 w_k1_IBM_s012
                                 + 1.51197 w_k1_INTC_s012
                                 + 1.22364 w_k1_KO_s012
                                 + 0.919299 w_k1_MAT_s012
                                 + 1.21142 w_k1_MSFT_s012
                                 + 1.32678 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.723616 w_k1_ADM_s012
                                 + 0.634679 w_k1_AXP_s012
                                 + 0.726622 w_k1_DIS_s012
                                 + 0.741813 w_k1_FDX_s012
                                 + 0.825695 w_k1_IBM_s012
                                 + 1.1188 w_k1_INTC_s012
                                 + 0.827493 w_k1_KO_s012
                                 + 0.547356 w_k1_MAT_s012
                                 + 0.767885 w_k1_MSFT_s012
                                 + 0.897966 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.869412 w_k1_ADM_s012
                                 + 0.631301 w_k1_AXP_s012
                                 + 0.826049 w_k1_DIS_s012
                                 + 0.944954 w_k1_FDX_s012
                                 + 0.792321 w_k1_IBM_s012
                                 + 1.02251 w_k1_INTC_s012
                                 + 0.825085 w_k1_KO_s012
                                 + 0.790671 w_k1_MAT_s012
                                 + 0.965739 w_k1_MSFT_s012
                                 + 1.0128 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.07067 w_k1_ADM_s012
                                 + 1.01563 w_k1_AXP_s012
                                 + 0.820804 w_k1_DIS_s012
                                 + 0.980653 w_k1_FDX_s012
                                 + 0.9926 w_k1_IBM_s012
                                 + 1.05277 w_k1_INTC_s012
                                 + 0.990915 w_k1_KO_s012
                                 + 0.751005 w_k1_MAT_s012
                                 + 0.853305 w_k1_MSFT_s012
                                 + 1.00673 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.712707 w_k1_ADM_s012
                                 + 0.823887 w_k1_AXP_s012
                                 + 0.935462 w_k1_DIS_s012
                                 + 0.874587 w_k1_FDX_s012
                                 + 1.38057 w_k1_IBM_s012
                                 + 1.14698 w_k1_INTC_s012
                                 + 1.17248 w_k1_KO_s012
                                 + 0.975168 w_k1_MAT_s012
                                 + 0.989251 w_k1_MSFT_s012
                                 + 0.986506 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.03487 w_k1_ADM_s012
                                 + 0.831396 w_k1_AXP_s012
                                 + 1.06065 w_k1_DIS_s012
                                 + 1.05114 w_k1_FDX_s012
                                 + 0.842946 w_k1_IBM_s012
                                 + 1.29334 w_k1_INTC_s012
                                 + 1.00537 w_k1_KO_s012
                                 + 0.581597 w_k1_MAT_s012
                                 + 1.04417 w_k1_MSFT_s012
                                 + 0.979327 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.16357 w_k1_ADM_s013
                                 + 0.95043 w_k1_AXP_s013
                                 + 0.946522 w_k1_DIS_s013
                                 + 1.03059 w_k1_FDX_s013
                                 + 1.04541 w_k1_IBM_s013
                                 + 1.31498 w_k1_INTC_s013
                                 + 1.15166 w_k1_KO_s013
                                 + 0.984446 w_k1_MAT_s013
                                 + 1.0366 w_k1_MSFT_s013
                                 + 0.984212 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.902627 w_k1_ADM_s013
                                 + 0.605357 w_k1_AXP_s013
                                 + 0.929283 w_k1_DIS_s013
                                 + 0.84042 w_k1_FDX_s013
                                 + 1.09485 w_k1_IBM_s013
                                 + 1.03755 w_k1_INTC_s013
                                 + 1.31368 w_k1_KO_s013
                                 + 0.841294 w_k1_MAT_s013
                                 + 0.990118 w_k1_MSFT_s013
                                 + 1.05007 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.04898 w_k1_ADM_s013
                                 + 1.07833 w_k1_AXP_s013
                                 + 0.867218 w_k1_DIS_s013
                                 + 1.0868 w_k1_FDX_s013
                                 + 0.790186 w_k1_IBM_s013
                                 + 1.09113 w_k1_INTC_s013
                                 + 0.985455 w_k1_KO_s013
                                 + 0.943405 w_k1_MAT_s013
                                 + 0.896037 w_k1_MSFT_s013
                                 + 1.0659 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.680217 w_k1_ADM_s013
                                 + 0.675509 w_k1_AXP_s013
                                 + 0.64566 w_k1_DIS_s013
                                 + 0.76247 w_k1_FDX_s013
                                 + 0.704581 w_k1_IBM_s013
                                 + 0.960289 w_k1_INTC_s013
                                 + 0.874979 w_k1_KO_s013
                                 + 0.78957 w_k1_MAT_s013
                                 + 0.781675 w_k1_MSFT_s013
                                 + 0.923184 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.13263 w_k1_ADM_s013
                                 + 0.968385 w_k1_AXP_s013
                                 + 1.09007 w_k1_DIS_s013
                                 + 0.948288 w_k1_FDX_s013
                                 + 1.23498 w_k1_IBM_s013
                                 + 1.26993 w_k1_INTC_s013
                                 + 1.41199 w_k1_KO_s013 + 1.48851 w_k1_MAT_s013
                                 + 1.07465 w_k1_MSFT_s013 + 1.1892 w_k1_PG_s013
                                 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.50543 w_k1_ADM_s013
                                 + 1.41865 w_k1_AXP_s013 + 1.7682 w_k1_DIS_s013
                                 + 1.56169 w_k1_FDX_s013
                                 + 1.10279 w_k1_IBM_s013
                                 + 1.77432 w_k1_INTC_s013
                                 + 1.51633 w_k1_KO_s013
                                 + 0.848919 w_k1_MAT_s013
                                 + 1.30326 w_k1_MSFT_s013
                                 + 1.19823 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.35366 w_k1_ADM_s013
                                 + 1.37142 w_k1_AXP_s013
                                 + 1.54722 w_k1_DIS_s013
                                 + 1.26431 w_k1_FDX_s013
                                 + 1.11276 w_k1_IBM_s013
                                 + 1.40052 w_k1_INTC_s013
                                 + 1.34998 w_k1_KO_s013 + 1.12515 w_k1_MAT_s013
                                 + 1.29876 w_k1_MSFT_s013
                                 + 1.29165 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.03827 w_k1_ADM_s013
                                 + 1.07611 w_k1_AXP_s013
                                 + 1.43955 w_k1_DIS_s013
                                 + 1.12193 w_k1_FDX_s013
                                 + 1.22192 w_k1_IBM_s013
                                 + 1.1645 w_k1_INTC_s013 + 1.32098 w_k1_KO_s013
                                 + 0.8796 w_k1_MAT_s013
                                 + 1.11384 w_k1_MSFT_s013
                                 + 1.04881 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.10237 w_k1_ADM_s014
                                 + 1.1055 w_k1_AXP_s014 + 1.22646 w_k1_DIS_s014
                                 + 1.07738 w_k1_FDX_s014 + 1.0087 w_k1_IBM_s014
                                 + 1.16351 w_k1_INTC_s014
                                 + 1.17902 w_k1_KO_s014 + 1.21336 w_k1_MAT_s014
                                 + 1.17105 w_k1_MSFT_s014
                                 + 1.19112 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.706095 w_k1_ADM_s014
                                 + 0.768476 w_k1_AXP_s014
                                 + 0.840404 w_k1_DIS_s014
                                 + 0.809023 w_k1_FDX_s014
                                 + 0.871693 w_k1_IBM_s014
                                 + 0.910038 w_k1_INTC_s014
                                 + 0.808847 w_k1_KO_s014
                                 + 0.682292 w_k1_MAT_s014
                                 + 0.879132 w_k1_MSFT_s014
                                 + 0.941986 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.37587 w_k1_ADM_s014
                                 + 1.71013 w_k1_AXP_s014
                                 + 1.89299 w_k1_DIS_s014
                                 + 1.56876 w_k1_FDX_s014
                                 + 1.31195 w_k1_IBM_s014
                                 + 1.54641 w_k1_INTC_s014
                                 + 1.23364 w_k1_KO_s014
                                 + 0.868649 w_k1_MAT_s014
                                 + 1.29142 w_k1_MSFT_s014
                                 + 1.22043 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.04491 w_k1_ADM_s014
                                 + 1.28573 w_k1_AXP_s014
                                 + 0.953798 w_k1_DIS_s014
                                 + 1.20946 w_k1_FDX_s014
                                 + 0.643228 w_k1_IBM_s014
                                 + 1.12654 w_k1_INTC_s014
                                 + 0.806449 w_k1_KO_s014
                                 + 0.891296 w_k1_MAT_s014
                                 + 0.996777 w_k1_MSFT_s014
                                 + 1.06486 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.905305 w_k1_ADM_s014
                                 + 1.10003 w_k1_AXP_s014
                                 + 0.929331 w_k1_DIS_s014
                                 + 0.884634 w_k1_FDX_s014
                                 + 1.32851 w_k1_IBM_s014
                                 + 1.32545 w_k1_INTC_s014
                                 + 1.08832 w_k1_KO_s014 + 1.13994 w_k1_MAT_s014
                                 + 0.947403 w_k1_MSFT_s014
                                 + 1.02339 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.05381 w_k1_ADM_s014
                                 + 0.872731 w_k1_AXP_s014
                                 + 0.951494 w_k1_DIS_s014
                                 + 1.00657 w_k1_FDX_s014
                                 + 0.699471 w_k1_IBM_s014
                                 + 0.936689 w_k1_INTC_s014
                                 + 0.990732 w_k1_KO_s014
                                 + 0.70055 w_k1_MAT_s014
                                 + 1.05133 w_k1_MSFT_s014
                                 + 0.980448 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.924905 w_k1_ADM_s014
                                 + 1.08748 w_k1_AXP_s014
                                 + 0.782226 w_k1_DIS_s014
                                 + 1.00295 w_k1_FDX_s014
                                 + 0.812571 w_k1_IBM_s014
                                 + 0.785701 w_k1_INTC_s014
                                 + 0.764169 w_k1_KO_s014 + 1.0363 w_k1_MAT_s014
                                 + 0.81973 w_k1_MSFT_s014
                                 + 0.953504 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.27193 w_k1_ADM_s014
                                 + 1.4485 w_k1_AXP_s014 + 1.50881 w_k1_DIS_s014
                                 + 1.21042 w_k1_FDX_s014
                                 + 1.49088 w_k1_IBM_s014
                                 + 1.46758 w_k1_INTC_s014 + 1.4244 w_k1_KO_s014
                                 + 1.37095 w_k1_MAT_s014
                                 + 1.18201 w_k1_MSFT_s014
                                 + 1.25458 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.613051 w_k1_ADM_s015
                                 + 0.64074 w_k1_AXP_s015
                                 + 0.615241 w_k1_DIS_s015
                                 + 0.695844 w_k1_FDX_s015
                                 + 0.766359 w_k1_IBM_s015
                                 + 1.04635 w_k1_INTC_s015
                                 + 0.841896 w_k1_KO_s015
                                 + 0.707531 w_k1_MAT_s015
                                 + 0.793107 w_k1_MSFT_s015
                                 + 0.757687 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.23348 w_k1_ADM_s015
                                 + 1.13911 w_k1_AXP_s015
                                 + 1.20013 w_k1_DIS_s015
                                 + 1.11971 w_k1_FDX_s015
                                 + 1.72479 w_k1_IBM_s015
                                 + 1.44056 w_k1_INTC_s015
                                 + 1.41459 w_k1_KO_s015 + 1.68116 w_k1_MAT_s015
                                 + 1.14435 w_k1_MSFT_s015
                                 + 1.28206 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.713605 w_k1_ADM_s015
                                 + 0.935641 w_k1_AXP_s015
                                 + 0.825754 w_k1_DIS_s015
                                 + 0.783739 w_k1_FDX_s015
                                 + 1.31209 w_k1_IBM_s015
                                 + 0.958906 w_k1_INTC_s015
                                 + 0.975838 w_k1_KO_s015
                                 + 1.05146 w_k1_MAT_s015
                                 + 0.769492 w_k1_MSFT_s015
                                 + 0.748942 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.597017 w_k1_ADM_s015
                                 + 0.928359 w_k1_AXP_s015
                                 + 0.719903 w_k1_DIS_s015
                                 + 0.872296 w_k1_FDX_s015
                                 + 0.935225 w_k1_IBM_s015
                                 + 0.909331 w_k1_INTC_s015
                                 + 0.891614 w_k1_KO_s015
                                 + 0.976917 w_k1_MAT_s015
                                 + 0.897765 w_k1_MSFT_s015
                                 + 0.87281 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.846679 w_k1_ADM_s015
                                 + 0.849857 w_k1_AXP_s015
                                 + 0.981136 w_k1_DIS_s015
                                 + 0.994891 w_k1_FDX_s015
                                 + 0.767321 w_k1_IBM_s015
                                 + 1.31072 w_k1_INTC_s015
                                 + 0.949101 w_k1_KO_s015
                                 + 0.747887 w_k1_MAT_s015
                                 + 1.06923 w_k1_MSFT_s015
                                 + 1.04489 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.10428 w_k1_ADM_s015
                                 + 1.47211 w_k1_AXP_s015
                                 + 0.950515 w_k1_DIS_s015
                                 + 1.05636 w_k1_FDX_s015
                                 + 1.19182 w_k1_IBM_s015
                                 + 1.25277 w_k1_INTC_s015
                                 + 1.19185 w_k1_KO_s015 + 1.641 w_k1_MAT_s015
                                 + 1.03527 w_k1_MSFT_s015
                                 + 1.14977 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.34919 w_k1_ADM_s015
                                 + 1.56628 w_k1_AXP_s015
                                 + 1.60101 w_k1_DIS_s015
                                 + 1.55577 w_k1_FDX_s015
                                 + 0.915198 w_k1_IBM_s015
                                 + 1.3787 w_k1_INTC_s015 + 1.08754 w_k1_KO_s015
                                 + 1.04293 w_k1_MAT_s015
                                 + 1.40931 w_k1_MSFT_s015
                                 + 1.17072 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.94619 w_k1_ADM_s015
                                 + 1.10777 w_k1_AXP_s015
                                 + 1.06126 w_k1_DIS_s015
                                 + 1.06163 w_k1_FDX_s015
                                 + 1.11815 w_k1_IBM_s015
                                 + 1.3259 w_k1_INTC_s015 + 1.18012 w_k1_KO_s015
                                 + 0.695759 w_k1_MAT_s015
                                 + 1.14994 w_k1_MSFT_s015
                                 + 0.962284 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.963858 w_k1_ADM_s016
                                 + 0.81716 w_k1_AXP_s016
                                 + 0.880195 w_k1_DIS_s016
                                 + 0.959292 w_k1_FDX_s016
                                 + 0.582289 w_k1_IBM_s016
                                 + 0.798668 w_k1_INTC_s016
                                 + 0.86838 w_k1_KO_s016
                                 + 0.635165 w_k1_MAT_s016
                                 + 0.868975 w_k1_MSFT_s016
                                 + 0.975735 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.993269 w_k1_ADM_s016
                                 + 1.19897 w_k1_AXP_s016
                                 + 1.04194 w_k1_DIS_s016
                                 + 1.13558 w_k1_FDX_s016
                                 + 1.07387 w_k1_IBM_s016
                                 + 1.0014 w_k1_INTC_s016 + 1.02497 w_k1_KO_s016
                                 + 1.08589 w_k1_MAT_s016
                                 + 0.97345 w_k1_MSFT_s016
                                 + 1.01502 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.792857 w_k1_ADM_s016
                                 + 0.719037 w_k1_AXP_s016
                                 + 0.729334 w_k1_DIS_s016
                                 + 0.754813 w_k1_FDX_s016
                                 + 0.867289 w_k1_IBM_s016
                                 + 0.856401 w_k1_INTC_s016
                                 + 0.852605 w_k1_KO_s016
                                 + 0.787951 w_k1_MAT_s016
                                 + 0.762574 w_k1_MSFT_s016
                                 + 0.902081 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.06371 w_k1_ADM_s016
                                 + 0.867498 w_k1_AXP_s016
                                 + 0.784599 w_k1_DIS_s016
                                 + 0.861923 w_k1_FDX_s016
                                 + 0.761594 w_k1_IBM_s016
                                 + 0.741276 w_k1_INTC_s016
                                 + 0.764379 w_k1_KO_s016
                                 + 0.662637 w_k1_MAT_s016
                                 + 0.797033 w_k1_MSFT_s016
                                 + 0.790276 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.24467 w_k1_ADM_s016
                                 + 1.10478 w_k1_AXP_s016
                                 + 1.45198 w_k1_DIS_s016
                                 + 1.16841 w_k1_FDX_s016
                                 + 1.41139 w_k1_IBM_s016
                                 + 0.891371 w_k1_INTC_s016
                                 + 1.24495 w_k1_KO_s016 + 1.24969 w_k1_MAT_s016
                                 + 1.03262 w_k1_MSFT_s016
                                 + 1.27172 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.13772 w_k1_ADM_s016
                                 + 0.801789 w_k1_AXP_s016
                                 + 1.24784 w_k1_DIS_s016
                                 + 1.08527 w_k1_FDX_s016
                                 + 0.738441 w_k1_IBM_s016
                                 + 0.930275 w_k1_INTC_s016
                                 + 1.12774 w_k1_KO_s016
                                 + 0.897506 w_k1_MAT_s016
                                 + 1.09215 w_k1_MSFT_s016 + 1.0758 w_k1_PG_s016
                                 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.950793 w_k1_ADM_s016
                                 + 0.919954 w_k1_AXP_s016
                                 + 0.875264 w_k1_DIS_s016
                                 + 0.928079 w_k1_FDX_s016
                                 + 0.996696 w_k1_IBM_s016
                                 + 0.944984 w_k1_INTC_s016
                                 + 0.936884 w_k1_KO_s016
                                 + 1.14077 w_k1_MAT_s016
                                 + 0.997092 w_k1_MSFT_s016
                                 + 1.03275 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.35226 w_k1_ADM_s016
                                 + 1.38263 w_k1_AXP_s016
                                 + 1.29814 w_k1_DIS_s016
                                 + 1.38308 w_k1_FDX_s016
                                 + 1.01811 w_k1_IBM_s016
                                 + 1.07683 w_k1_INTC_s016 + 1.0766 w_k1_KO_s016
                                 + 1.29059 w_k1_MAT_s016
                                 + 1.11279 w_k1_MSFT_s016 + 1.2679 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.772621 w_k1_ADM_s017
                                 + 0.933478 w_k1_AXP_s017
                                 + 0.847791 w_k1_DIS_s017
                                 + 0.914088 w_k1_FDX_s017
                                 + 1.08878 w_k1_IBM_s017
                                 + 0.90647 w_k1_INTC_s017
                                 + 0.914822 w_k1_KO_s017
                                 + 1.16453 w_k1_MAT_s017
                                 + 0.961623 w_k1_MSFT_s017
                                 + 0.85206 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.12027 w_k1_ADM_s017
                                 + 1.09725 w_k1_AXP_s017
                                 + 1.03805 w_k1_DIS_s017
                                 + 1.13376 w_k1_FDX_s017
                                 + 0.938744 w_k1_IBM_s017
                                 + 0.851324 w_k1_INTC_s017
                                 + 0.901148 w_k1_KO_s017
                                 + 1.08599 w_k1_MAT_s017
                                 + 0.958257 w_k1_MSFT_s017
                                 + 0.91676 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.995271 w_k1_ADM_s017
                                 + 1.31466 w_k1_AXP_s017
                                 + 1.14143 w_k1_DIS_s017
                                 + 1.06014 w_k1_FDX_s017
                                 + 1.52784 w_k1_IBM_s017
                                 + 1.00548 w_k1_INTC_s017
                                 + 1.36954 w_k1_KO_s017 + 1.65317 w_k1_MAT_s017
                                 + 1.06102 w_k1_MSFT_s017
                                 + 1.03808 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.877054 w_k1_ADM_s017
                                 + 1.41863 w_k1_AXP_s017 + 1.0844 w_k1_DIS_s017
                                 + 1.22542 w_k1_FDX_s017
                                 + 1.14082 w_k1_IBM_s017
                                 + 1.06331 w_k1_INTC_s017
                                 + 1.12698 w_k1_KO_s017 + 1.7591 w_k1_MAT_s017
                                 + 1.03311 w_k1_MSFT_s017
                                 + 1.01743 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.826459 w_k1_ADM_s017
                                 + 0.814667 w_k1_AXP_s017
                                 + 0.786378 w_k1_DIS_s017
                                 + 0.946255 w_k1_FDX_s017
                                 + 0.819959 w_k1_IBM_s017
                                 + 0.876035 w_k1_INTC_s017
                                 + 0.842342 w_k1_KO_s017
                                 + 0.745217 w_k1_MAT_s017
                                 + 0.885281 w_k1_MSFT_s017
                                 + 0.835923 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.667781 w_k1_ADM_s017
                                 + 0.659127 w_k1_AXP_s017
                                 + 0.792814 w_k1_DIS_s017
                                 + 0.881647 w_k1_FDX_s017
                                 + 0.723096 w_k1_IBM_s017
                                 + 0.920701 w_k1_INTC_s017
                                 + 0.809868 w_k1_KO_s017
                                 + 0.76003 w_k1_MAT_s017
                                 + 0.922266 w_k1_MSFT_s017
                                 + 0.853299 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.590785 w_k1_ADM_s017
                                 + 0.696286 w_k1_AXP_s017
                                 + 1.08865 w_k1_DIS_s017
                                 + 0.884048 w_k1_FDX_s017
                                 + 1.24807 w_k1_IBM_s017
                                 + 0.898057 w_k1_INTC_s017
                                 + 1.27134 w_k1_KO_s017 + 1.01426 w_k1_MAT_s017
                                 + 1.03395 w_k1_MSFT_s017
                                 + 0.918164 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.00732 w_k1_ADM_s017
                                 + 1.18291 w_k1_AXP_s017
                                 + 1.51412 w_k1_DIS_s017
                                 + 1.29708 w_k1_FDX_s017 + 1.0679 w_k1_IBM_s017
                                 + 1.07727 w_k1_INTC_s017
                                 + 1.15544 w_k1_KO_s017 + 1.35711 w_k1_MAT_s017
                                 + 1.20039 w_k1_MSFT_s017
                                 + 1.13497 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.39877 w_k1_ADM_s100
                                 + 1.66139 w_k1_AXP_s100
                                 + 1.53646 w_k1_DIS_s100
                                 + 1.29772 w_k1_FDX_s100
                                 + 1.55018 w_k1_IBM_s100
                                 + 1.35915 w_k1_INTC_s100
                                 + 1.52677 w_k1_KO_s100 + 1.83513 w_k1_MAT_s100
                                 + 1.22753 w_k1_MSFT_s100
                                 + 1.21604 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.975807 w_k1_ADM_s100
                                 + 0.983497 w_k1_AXP_s100
                                 + 0.740396 w_k1_DIS_s100
                                 + 0.897838 w_k1_FDX_s100
                                 + 0.769268 w_k1_IBM_s100
                                 + 0.910929 w_k1_INTC_s100
                                 + 0.798561 w_k1_KO_s100
                                 + 0.77493 w_k1_MAT_s100
                                 + 0.870689 w_k1_MSFT_s100
                                 + 0.862277 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.08765 w_k1_ADM_s100
                                 + 1.11111 w_k1_AXP_s100
                                 + 0.915121 w_k1_DIS_s100
                                 + 1.00117 w_k1_FDX_s100
                                 + 1.01425 w_k1_IBM_s100
                                 + 0.939185 w_k1_INTC_s100
                                 + 1.05902 w_k1_KO_s100 + 1.50761 w_k1_MAT_s100
                                 + 0.915113 w_k1_MSFT_s100
                                 + 0.999479 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.3831 w_k1_ADM_s100
                                 + 1.02757 w_k1_AXP_s100
                                 + 1.00345 w_k1_DIS_s100
                                 + 1.10503 w_k1_FDX_s100
                                 + 0.687531 w_k1_IBM_s100
                                 + 0.841939 w_k1_INTC_s100
                                 + 1.01314 w_k1_KO_s100
                                 + 0.754772 w_k1_MAT_s100
                                 + 1.0312 w_k1_MSFT_s100
                                 + 0.939106 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.16625 w_k1_ADM_s100
                                 + 0.942201 w_k1_AXP_s100
                                 + 0.838283 w_k1_DIS_s100
                                 + 0.865408 w_k1_FDX_s100
                                 + 1.42419 w_k1_IBM_s100
                                 + 0.773709 w_k1_INTC_s100
                                 + 1.26252 w_k1_KO_s100 + 1.37643 w_k1_MAT_s100
                                 + 0.841012 w_k1_MSFT_s100
                                 + 0.973144 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.14936 w_k1_ADM_s100
                                 + 1.15025 w_k1_AXP_s100
                                 + 0.800532 w_k1_DIS_s100
                                 + 1.15855 w_k1_FDX_s100
                                 + 0.644644 w_k1_IBM_s100
                                 + 0.965248 w_k1_INTC_s100
                                 + 0.787391 w_k1_KO_s100
                                 + 0.644115 w_k1_MAT_s100
                                 + 0.90967 w_k1_MSFT_s100
                                 + 0.964085 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.41883 w_k1_ADM_s100
                                 + 1.61555 w_k1_AXP_s100
                                 + 1.23309 w_k1_DIS_s100
                                 + 1.30248 w_k1_FDX_s100
                                 + 0.976379 w_k1_IBM_s100
                                 + 1.31597 w_k1_INTC_s100
                                 + 1.12862 w_k1_KO_s100 + 1.27237 w_k1_MAT_s100
                                 + 1.26783 w_k1_MSFT_s100
                                 + 1.10139 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.790525 w_k1_ADM_s100
                                 + 0.745842 w_k1_AXP_s100
                                 + 0.524794 w_k1_DIS_s100
                                 + 0.700332 w_k1_FDX_s100
                                 + 0.763724 w_k1_IBM_s100
                                 + 0.77442 w_k1_INTC_s100
                                 + 0.757125 w_k1_KO_s100
                                 + 0.989294 w_k1_MAT_s100
                                 + 0.740719 w_k1_MSFT_s100
                                 + 0.743872 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.11079 w_k1_ADM_s101
                                 + 1.24887 w_k1_AXP_s101 + 1.2656 w_k1_DIS_s101
                                 + 1.15542 w_k1_FDX_s101
                                 + 1.13876 w_k1_IBM_s101
                                 + 0.937434 w_k1_INTC_s101
                                 + 1.32637 w_k1_KO_s101 + 1.57267 w_k1_MAT_s101
                                 + 1.24706 w_k1_MSFT_s101
                                 + 1.14967 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.0087 w_k1_ADM_s101
                                 + 0.894188 w_k1_AXP_s101
                                 + 0.606015 w_k1_DIS_s101
                                 + 0.826845 w_k1_FDX_s101
                                 + 0.785424 w_k1_IBM_s101
                                 + 0.74297 w_k1_INTC_s101
                                 + 0.71554 w_k1_KO_s101
                                 + 0.967057 w_k1_MAT_s101
                                 + 0.909507 w_k1_MSFT_s101
                                 + 0.900404 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.854556 w_k1_ADM_s101
                                 + 0.860145 w_k1_AXP_s101
                                 + 0.842384 w_k1_DIS_s101
                                 + 0.803086 w_k1_FDX_s101
                                 + 1.35827 w_k1_IBM_s101
                                 + 0.863757 w_k1_INTC_s101
                                 + 1.01705 w_k1_KO_s101 + 1.49988 w_k1_MAT_s101
                                 + 0.838079 w_k1_MSFT_s101
                                 + 0.955142 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.88865 w_k1_ADM_s101
                                 + 0.61489 w_k1_AXP_s101
                                 + 0.632466 w_k1_DIS_s101
                                 + 0.729527 w_k1_FDX_s101
                                 + 0.619679 w_k1_IBM_s101
                                 + 0.755899 w_k1_INTC_s101
                                 + 0.735104 w_k1_KO_s101
                                 + 0.798806 w_k1_MAT_s101
                                 + 0.811437 w_k1_MSFT_s101
                                 + 0.840099 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.19716 w_k1_ADM_s101
                                 + 0.945823 w_k1_AXP_s101
                                 + 1.21966 w_k1_DIS_s101
                                 + 1.11955 w_k1_FDX_s101
                                 + 0.752882 w_k1_IBM_s101
                                 + 1.17334 w_k1_INTC_s101
                                 + 1.05245 w_k1_KO_s101
                                 + 0.751705 w_k1_MAT_s101
                                 + 1.22213 w_k1_MSFT_s101
                                 + 1.05754 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.880407 w_k1_ADM_s101
                                 + 0.94871 w_k1_AXP_s101
                                 + 0.89808 w_k1_DIS_s101
                                 + 0.920862 w_k1_FDX_s101
                                 + 0.944766 w_k1_IBM_s101
                                 + 1.00489 w_k1_INTC_s101
                                 + 0.927335 w_k1_KO_s101
                                 + 0.948734 w_k1_MAT_s101
                                 + 1.01623 w_k1_MSFT_s101
                                 + 0.925979 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.63993 w_k1_ADM_s101
                                 + 1.61392 w_k1_AXP_s101
                                 + 1.61104 w_k1_DIS_s101
                                 + 1.37146 w_k1_FDX_s101 + 1.4029 w_k1_IBM_s101
                                 + 1.21812 w_k1_INTC_s101
                                 + 1.61345 w_k1_KO_s101 + 1.61413 w_k1_MAT_s101
                                 + 1.43642 w_k1_MSFT_s101
                                 + 1.37081 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.899428 w_k1_ADM_s101
                                 + 1.06216 w_k1_AXP_s101
                                 + 0.898416 w_k1_DIS_s101
                                 + 1.09855 w_k1_FDX_s101
                                 + 0.734773 w_k1_IBM_s101
                                 + 0.941356 w_k1_INTC_s101
                                 + 0.743242 w_k1_KO_s101
                                 + 1.02947 w_k1_MAT_s101
                                 + 0.992818 w_k1_MSFT_s101
                                 + 0.986729 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.921872 w_k1_ADM_s102
                                 + 1.28404 w_k1_AXP_s102
                                 + 1.46038 w_k1_DIS_s102
                                 + 1.13523 w_k1_FDX_s102
                                 + 1.41171 w_k1_IBM_s102
                                 + 1.35355 w_k1_INTC_s102
                                 + 1.49835 w_k1_KO_s102 + 1.62671 w_k1_MAT_s102
                                 + 1.19407 w_k1_MSFT_s102
                                 + 1.06234 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.897293 w_k1_ADM_s102
                                 + 0.880336 w_k1_AXP_s102
                                 + 0.669388 w_k1_DIS_s102
                                 + 0.808265 w_k1_FDX_s102
                                 + 0.871248 w_k1_IBM_s102
                                 + 0.89659 w_k1_INTC_s102
                                 + 0.831645 w_k1_KO_s102
                                 + 1.14434 w_k1_MAT_s102
                                 + 0.848133 w_k1_MSFT_s102
                                 + 0.803029 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.955643 w_k1_ADM_s102
                                 + 1.54176 w_k1_AXP_s102
                                 + 1.12821 w_k1_DIS_s102
                                 + 1.20409 w_k1_FDX_s102 + 1.1631 w_k1_IBM_s102
                                 + 1.04132 w_k1_INTC_s102
                                 + 1.11818 w_k1_KO_s102 + 1.61587 w_k1_MAT_s102
                                 + 1.05742 w_k1_MSFT_s102
                                 + 0.985356 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.23417 w_k1_ADM_s102
                                 + 1.16197 w_k1_AXP_s102
                                 + 1.17269 w_k1_DIS_s102
                                 + 1.14892 w_k1_FDX_s102
                                 + 0.831966 w_k1_IBM_s102
                                 + 1.12694 w_k1_INTC_s102
                                 + 0.920941 w_k1_KO_s102
                                 + 0.830501 w_k1_MAT_s102
                                 + 1.07356 w_k1_MSFT_s102
                                 + 0.85885 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.26305 w_k1_ADM_s102
                                 + 1.69552 w_k1_AXP_s102
                                 + 1.77409 w_k1_DIS_s102
                                 + 1.37208 w_k1_FDX_s102 + 1.2041 w_k1_IBM_s102
                                 + 1.37605 w_k1_INTC_s102
                                 + 1.31822 w_k1_KO_s102 + 1.72614 w_k1_MAT_s102
                                 + 1.31007 w_k1_MSFT_s102
                                 + 1.19945 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.05039 w_k1_ADM_s102
                                 + 1.23958 w_k1_AXP_s102
                                 + 0.943723 w_k1_DIS_s102
                                 + 0.970048 w_k1_FDX_s102
                                 + 1.01613 w_k1_IBM_s102
                                 + 1.08299 w_k1_INTC_s102
                                 + 1.01339 w_k1_KO_s102 + 1.1239 w_k1_MAT_s102
                                 + 0.916461 w_k1_MSFT_s102
                                 + 0.919211 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.17608 w_k1_ADM_s102
                                 + 1.32858 w_k1_AXP_s102
                                 + 1.30584 w_k1_DIS_s102
                                 + 1.07181 w_k1_FDX_s102
                                 + 1.61474 w_k1_IBM_s102
                                 + 1.03344 w_k1_INTC_s102
                                 + 1.46099 w_k1_KO_s102 + 1.70036 w_k1_MAT_s102
                                 + 1.03032 w_k1_MSFT_s102
                                 + 1.11633 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.815399 w_k1_ADM_s102
                                 + 0.951014 w_k1_AXP_s102
                                 + 1.02576 w_k1_DIS_s102
                                 + 0.815164 w_k1_FDX_s102
                                 + 1.50309 w_k1_IBM_s102
                                 + 0.966723 w_k1_INTC_s102
                                 + 1.34837 w_k1_KO_s102 + 1.67875 w_k1_MAT_s102
                                 + 0.911228 w_k1_MSFT_s102
                                 + 1.04453 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.17088 w_k1_ADM_s103
                                 + 1.27174 w_k1_AXP_s103
                                 + 1.37278 w_k1_DIS_s103
                                 + 1.17623 w_k1_FDX_s103
                                 + 1.04722 w_k1_IBM_s103
                                 + 1.12271 w_k1_INTC_s103
                                 + 1.11434 w_k1_KO_s103 + 1.21027 w_k1_MAT_s103
                                 + 1.1212 w_k1_MSFT_s103 + 1.02766 w_k1_PG_s103
                                 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.12498 w_k1_ADM_s103
                                 + 1.05139 w_k1_AXP_s103
                                 + 1.35606 w_k1_DIS_s103
                                 + 0.983012 w_k1_FDX_s103
                                 + 1.53116 w_k1_IBM_s103
                                 + 1.08304 w_k1_INTC_s103
                                 + 1.44396 w_k1_KO_s103 + 2.01525 w_k1_MAT_s103
                                 + 1.15333 w_k1_MSFT_s103
                                 + 1.06717 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.24227 w_k1_ADM_s103
                                 + 1.56436 w_k1_AXP_s103
                                 + 1.68925 w_k1_DIS_s103
                                 + 1.42011 w_k1_FDX_s103
                                 + 1.05552 w_k1_IBM_s103
                                 + 1.13294 w_k1_INTC_s103
                                 + 1.25266 w_k1_KO_s103 + 1.24056 w_k1_MAT_s103
                                 + 1.2414 w_k1_MSFT_s103 + 1.20884 w_k1_PG_s103
                                 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.790948 w_k1_ADM_s103
                                 + 0.731688 w_k1_AXP_s103
                                 + 0.714026 w_k1_DIS_s103
                                 + 0.751215 w_k1_FDX_s103
                                 + 0.885681 w_k1_IBM_s103
                                 + 0.885497 w_k1_INTC_s103
                                 + 0.881673 w_k1_KO_s103
                                 + 0.856326 w_k1_MAT_s103
                                 + 0.849249 w_k1_MSFT_s103
                                 + 0.889921 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.847436 w_k1_ADM_s103
                                 + 0.653237 w_k1_AXP_s103
                                 + 0.678099 w_k1_DIS_s103
                                 + 0.749568 w_k1_FDX_s103
                                 + 0.531552 w_k1_IBM_s103
                                 + 0.729491 w_k1_INTC_s103
                                 + 0.668749 w_k1_KO_s103
                                 + 0.620201 w_k1_MAT_s103
                                 + 0.794455 w_k1_MSFT_s103
                                 + 0.790433 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.855719 w_k1_ADM_s103
                                 + 0.753559 w_k1_AXP_s103
                                 + 1.03449 w_k1_DIS_s103
                                 + 0.969728 w_k1_FDX_s103
                                 + 0.741296 w_k1_IBM_s103
                                 + 0.911248 w_k1_INTC_s103
                                 + 0.979451 w_k1_KO_s103
                                 + 0.705693 w_k1_MAT_s103
                                 + 1.00637 w_k1_MSFT_s103
                                 + 0.960101 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.05648 w_k1_ADM_s103
                                 + 1.22609 w_k1_AXP_s103
                                 + 1.01456 w_k1_DIS_s103
                                 + 1.08034 w_k1_FDX_s103
                                 + 0.861087 w_k1_IBM_s103
                                 + 1.17449 w_k1_INTC_s103
                                 + 0.81483 w_k1_KO_s103
                                 + 0.861437 w_k1_MAT_s103
                                 + 1.00522 w_k1_MSFT_s103
                                 + 0.936332 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.846881 w_k1_ADM_s103
                                 + 0.948288 w_k1_AXP_s103
                                 + 1.02154 w_k1_DIS_s103
                                 + 0.862035 w_k1_FDX_s103
                                 + 1.15125 w_k1_IBM_s103
                                 + 1.01022 w_k1_INTC_s103
                                 + 1.05289 w_k1_KO_s103 + 1.00889 w_k1_MAT_s103
                                 + 0.892194 w_k1_MSFT_s103
                                 + 0.900237 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.930811 w_k1_ADM_s104
                                 + 1.42867 w_k1_AXP_s104 + 1.046 w_k1_DIS_s104
                                 + 1.1166 w_k1_FDX_s104
                                 + 0.998791 w_k1_IBM_s104
                                 + 1.1383 w_k1_INTC_s104 + 1.1619 w_k1_KO_s104
                                 + 1.44208 w_k1_MAT_s104
                                 + 1.31459 w_k1_MSFT_s104
                                 + 1.07937 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.26518 w_k1_ADM_s104
                                 + 1.70225 w_k1_AXP_s104
                                 + 1.44882 w_k1_DIS_s104
                                 + 1.38948 w_k1_FDX_s104
                                 + 1.20681 w_k1_IBM_s104
                                 + 1.18265 w_k1_INTC_s104
                                 + 1.29208 w_k1_KO_s104 + 1.7606 w_k1_MAT_s104
                                 + 1.29083 w_k1_MSFT_s104
                                 + 1.16322 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.22882 w_k1_ADM_s104
                                 + 1.12077 w_k1_AXP_s104 + 1.1682 w_k1_DIS_s104
                                 + 0.946727 w_k1_FDX_s104
                                 + 1.56502 w_k1_IBM_s104
                                 + 1.02284 w_k1_INTC_s104
                                 + 1.27087 w_k1_KO_s104 + 1.54428 w_k1_MAT_s104
                                 + 1.17396 w_k1_MSFT_s104
                                 + 0.926285 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.879104 w_k1_ADM_s104
                                 + 0.799894 w_k1_AXP_s104
                                 + 0.846204 w_k1_DIS_s104
                                 + 0.863186 w_k1_FDX_s104
                                 + 0.880856 w_k1_IBM_s104
                                 + 0.851389 w_k1_INTC_s104
                                 + 0.899328 w_k1_KO_s104
                                 + 0.945376 w_k1_MAT_s104
                                 + 0.974845 w_k1_MSFT_s104
                                 + 0.898715 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.768281 w_k1_ADM_s104
                                 + 0.976539 w_k1_AXP_s104
                                 + 0.893279 w_k1_DIS_s104
                                 + 0.853319 w_k1_FDX_s104
                                 + 1.18041 w_k1_IBM_s104
                                 + 0.800522 w_k1_INTC_s104
                                 + 0.930177 w_k1_KO_s104
                                 + 1.12453 w_k1_MAT_s104
                                 + 0.94767 w_k1_MSFT_s104
                                 + 0.928754 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.07976 w_k1_ADM_s104
                                 + 1.11375 w_k1_AXP_s104
                                 + 1.36338 w_k1_DIS_s104
                                 + 1.20909 w_k1_FDX_s104
                                 + 0.950994 w_k1_IBM_s104
                                 + 1.14598 w_k1_INTC_s104 + 1.1798 w_k1_KO_s104
                                 + 0.946112 w_k1_MAT_s104
                                 + 1.31607 w_k1_MSFT_s104
                                 + 0.955723 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.06578 w_k1_ADM_s104
                                 + 0.979328 w_k1_AXP_s104
                                 + 0.805464 w_k1_DIS_s104
                                 + 0.955768 w_k1_FDX_s104
                                 + 0.649285 w_k1_IBM_s104
                                 + 0.840218 w_k1_INTC_s104
                                 + 0.821321 w_k1_KO_s104
                                 + 0.706959 w_k1_MAT_s104
                                 + 0.935388 w_k1_MSFT_s104
                                 + 0.92072 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.72668 w_k1_ADM_s104
                                 + 1.30404 w_k1_AXP_s104
                                 + 1.67398 w_k1_DIS_s104
                                 + 1.43275 w_k1_FDX_s104
                                 + 1.21071 w_k1_IBM_s104
                                 + 1.11556 w_k1_INTC_s104
                                 + 1.39383 w_k1_KO_s104 + 1.22398 w_k1_MAT_s104
                                 + 1.60605 w_k1_MSFT_s104
                                 + 1.18077 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.16339 w_k1_ADM_s105
                                 + 1.20917 w_k1_AXP_s105 + 1.3382 w_k1_DIS_s105
                                 + 1.1792 w_k1_FDX_s105 + 1.31567 w_k1_IBM_s105
                                 + 1.38268 w_k1_INTC_s105
                                 + 1.32456 w_k1_KO_s105 + 1.90546 w_k1_MAT_s105
                                 + 1.1768 w_k1_MSFT_s105 + 1.13241 w_k1_PG_s105
                                 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.852951 w_k1_ADM_s105
                                 + 0.775477 w_k1_AXP_s105
                                 + 0.772515 w_k1_DIS_s105
                                 + 0.823825 w_k1_FDX_s105
                                 + 0.622512 w_k1_IBM_s105
                                 + 0.909377 w_k1_INTC_s105
                                 + 0.723151 w_k1_KO_s105
                                 + 0.743349 w_k1_MAT_s105
                                 + 0.899858 w_k1_MSFT_s105
                                 + 0.760609 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.25216 w_k1_ADM_s105
                                 + 1.28756 w_k1_AXP_s105
                                 + 1.08444 w_k1_DIS_s105
                                 + 1.10945 w_k1_FDX_s105
                                 + 0.821263 w_k1_IBM_s105
                                 + 1.00337 w_k1_INTC_s105
                                 + 0.972199 w_k1_KO_s105
                                 + 1.13408 w_k1_MAT_s105
                                 + 1.11019 w_k1_MSFT_s105
                                 + 1.00781 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.710894 w_k1_ADM_s105
                                 + 0.836493 w_k1_AXP_s105
                                 + 1.03907 w_k1_DIS_s105
                                 + 0.881768 w_k1_FDX_s105
                                 + 1.0585 w_k1_IBM_s105
                                 + 1.08733 w_k1_INTC_s105
                                 + 1.00666 w_k1_KO_s105 + 1.18435 w_k1_MAT_s105
                                 + 0.99443 w_k1_MSFT_s105
                                 + 0.979865 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.824035 w_k1_ADM_s105
                                 + 0.833061 w_k1_AXP_s105
                                 + 0.679708 w_k1_DIS_s105
                                 + 0.785946 w_k1_FDX_s105
                                 + 1.02646 w_k1_IBM_s105
                                 + 0.845301 w_k1_INTC_s105
                                 + 0.883691 w_k1_KO_s105
                                 + 1.38134 w_k1_MAT_s105
                                 + 0.898681 w_k1_MSFT_s105
                                 + 0.878542 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.991991 w_k1_ADM_s105
                                 + 1.08558 w_k1_AXP_s105
                                 + 1.03197 w_k1_DIS_s105
                                 + 0.957282 w_k1_FDX_s105
                                 + 1.24919 w_k1_IBM_s105
                                 + 1.03072 w_k1_INTC_s105
                                 + 1.09666 w_k1_KO_s105 + 2.13368 w_k1_MAT_s105
                                 + 1.01396 w_k1_MSFT_s105
                                 + 0.997897 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.12593 w_k1_ADM_s105
                                 + 0.980804 w_k1_AXP_s105
                                 + 1.17328 w_k1_DIS_s105
                                 + 1.03189 w_k1_FDX_s105
                                 + 0.771094 w_k1_IBM_s105
                                 + 1.02169 w_k1_INTC_s105
                                 + 0.953598 w_k1_KO_s105
                                 + 0.789896 w_k1_MAT_s105
                                 + 1.03046 w_k1_MSFT_s105
                                 + 1.00797 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.56261 w_k1_ADM_s105
                                 + 1.69367 w_k1_AXP_s105
                                 + 1.51882 w_k1_DIS_s105
                                 + 1.42204 w_k1_FDX_s105
                                 + 0.989327 w_k1_IBM_s105
                                 + 1.3854 w_k1_INTC_s105 + 1.10713 w_k1_KO_s105
                                 + 1.2195 w_k1_MAT_s105
                                 + 1.35665 w_k1_MSFT_s105
                                 + 1.17255 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.16184 w_k1_ADM_s106
                                 + 1.3407 w_k1_AXP_s106 + 1.35266 w_k1_DIS_s106
                                 + 1.25084 w_k1_FDX_s106
                                 + 1.15174 w_k1_IBM_s106
                                 + 1.38334 w_k1_INTC_s106
                                 + 1.11186 w_k1_KO_s106
                                 + 0.875494 w_k1_MAT_s106
                                 + 1.26137 w_k1_MSFT_s106
                                 + 1.20049 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.02187 w_k1_ADM_s106
                                 + 0.937209 w_k1_AXP_s106
                                 + 1.3599 w_k1_DIS_s106 + 1.14156 w_k1_FDX_s106
                                 + 0.976286 w_k1_IBM_s106
                                 + 1.48704 w_k1_INTC_s106
                                 + 1.11171 w_k1_KO_s106
                                 + 0.760787 w_k1_MAT_s106
                                 + 1.12668 w_k1_MSFT_s106
                                 + 0.88581 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.970079 w_k1_ADM_s106
                                 + 1.40219 w_k1_AXP_s106
                                 + 1.29882 w_k1_DIS_s106
                                 + 1.09185 w_k1_FDX_s106
                                 + 1.58495 w_k1_IBM_s106
                                 + 1.33668 w_k1_INTC_s106
                                 + 1.29278 w_k1_KO_s106 + 1.5098 w_k1_MAT_s106
                                 + 1.03891 w_k1_MSFT_s106
                                 + 0.998284 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.909808 w_k1_ADM_s106
                                 + 0.914684 w_k1_AXP_s106
                                 + 0.664366 w_k1_DIS_s106
                                 + 0.809716 w_k1_FDX_s106
                                 + 0.83323 w_k1_IBM_s106
                                 + 1.09613 w_k1_INTC_s106
                                 + 0.789896 w_k1_KO_s106
                                 + 0.808759 w_k1_MAT_s106
                                 + 0.935303 w_k1_MSFT_s106
                                 + 0.822283 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.973976 w_k1_ADM_s106
                                 + 0.587566 w_k1_AXP_s106
                                 + 0.795568 w_k1_DIS_s106
                                 + 0.856265 w_k1_FDX_s106
                                 + 0.687346 w_k1_IBM_s106
                                 + 1.17665 w_k1_INTC_s106
                                 + 0.870864 w_k1_KO_s106
                                 + 0.368797 w_k1_MAT_s106
                                 + 0.988479 w_k1_MSFT_s106
                                 + 0.881076 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.811022 w_k1_ADM_s106
                                 + 0.845379 w_k1_AXP_s106
                                 + 0.69083 w_k1_DIS_s106
                                 + 0.773596 w_k1_FDX_s106
                                 + 1.07572 w_k1_IBM_s106
                                 + 1.22556 w_k1_INTC_s106
                                 + 0.964923 w_k1_KO_s106
                                 + 1.05831 w_k1_MAT_s106
                                 + 0.773814 w_k1_MSFT_s106
                                 + 0.87653 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.827633 w_k1_ADM_s106
                                 + 0.604212 w_k1_AXP_s106
                                 + 0.486875 w_k1_DIS_s106
                                 + 0.723438 w_k1_FDX_s106
                                 + 0.617772 w_k1_IBM_s106
                                 + 0.98115 w_k1_INTC_s106
                                 + 0.591873 w_k1_KO_s106
                                 + 0.426677 w_k1_MAT_s106
                                 + 0.677965 w_k1_MSFT_s106
                                 + 0.710898 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.974211 w_k1_ADM_s106
                                 + 0.965245 w_k1_AXP_s106
                                 + 0.886052 w_k1_DIS_s106
                                 + 1.03106 w_k1_FDX_s106
                                 + 0.804689 w_k1_IBM_s106
                                 + 1.36694 w_k1_INTC_s106
                                 + 0.857932 w_k1_KO_s106
                                 + 0.718772 w_k1_MAT_s106
                                 + 1.04888 w_k1_MSFT_s106
                                 + 0.981515 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.47909 w_k1_ADM_s107
                                 + 1.53086 w_k1_AXP_s107
                                 + 1.59299 w_k1_DIS_s107
                                 + 1.25426 w_k1_FDX_s107 + 1.6047 w_k1_IBM_s107
                                 + 1.17515 w_k1_INTC_s107
                                 + 1.61916 w_k1_KO_s107 + 1.7488 w_k1_MAT_s107
                                 + 1.30674 w_k1_MSFT_s107
                                 + 1.15747 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.20888 w_k1_ADM_s107
                                 + 0.923489 w_k1_AXP_s107
                                 + 1.30166 w_k1_DIS_s107
                                 + 1.15412 w_k1_FDX_s107
                                 + 1.06382 w_k1_IBM_s107
                                 + 1.07328 w_k1_INTC_s107
                                 + 1.29582 w_k1_KO_s107
                                 + 0.897137 w_k1_MAT_s107
                                 + 1.20422 w_k1_MSFT_s107
                                 + 1.11794 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.829243 w_k1_ADM_s107
                                 + 0.977044 w_k1_AXP_s107
                                 + 0.893255 w_k1_DIS_s107
                                 + 0.908639 w_k1_FDX_s107
                                 + 0.9678 w_k1_IBM_s107
                                 + 1.11041 w_k1_INTC_s107 + 1.0202 w_k1_KO_s107
                                 + 1.0908 w_k1_MAT_s107
                                 + 1.05641 w_k1_MSFT_s107
                                 + 0.986486 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.14967 w_k1_ADM_s107
                                 + 1.29866 w_k1_AXP_s107
                                 + 1.16532 w_k1_DIS_s107
                                 + 1.26885 w_k1_FDX_s107
                                 + 0.795939 w_k1_IBM_s107
                                 + 1.3475 w_k1_INTC_s107
                                 + 0.953252 w_k1_KO_s107
                                 + 0.88269 w_k1_MAT_s107
                                 + 1.33897 w_k1_MSFT_s107
                                 + 0.92257 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.39647 w_k1_ADM_s107
                                 + 1.01831 w_k1_AXP_s107 + 1.1402 w_k1_DIS_s107
                                 + 1.08096 w_k1_FDX_s107
                                 + 1.32247 w_k1_IBM_s107
                                 + 1.26902 w_k1_INTC_s107
                                 + 1.28703 w_k1_KO_s107 + 1.32929 w_k1_MAT_s107
                                 + 1.12288 w_k1_MSFT_s107
                                 + 1.06174 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.922479 w_k1_ADM_s107
                                 + 0.706547 w_k1_AXP_s107
                                 + 0.737764 w_k1_DIS_s107
                                 + 0.890125 w_k1_FDX_s107
                                 + 0.58007 w_k1_IBM_s107
                                 + 1.0473 w_k1_INTC_s107
                                 + 0.768653 w_k1_KO_s107
                                 + 0.635151 w_k1_MAT_s107
                                 + 1.02615 w_k1_MSFT_s107
                                 + 0.829205 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.08554 w_k1_ADM_s107
                                 + 0.640248 w_k1_AXP_s107
                                 + 0.916803 w_k1_DIS_s107
                                 + 0.819116 w_k1_FDX_s107
                                 + 0.882711 w_k1_IBM_s107
                                 + 0.886392 w_k1_INTC_s107
                                 + 0.920975 w_k1_KO_s107
                                 + 0.706532 w_k1_MAT_s107
                                 + 0.923057 w_k1_MSFT_s107
                                 + 0.816152 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.874197 w_k1_ADM_s107
                                 + 0.637702 w_k1_AXP_s107
                                 + 0.697152 w_k1_DIS_s107
                                 + 0.735629 w_k1_FDX_s107
                                 + 0.97003 w_k1_IBM_s107
                                 + 0.898428 w_k1_INTC_s107
                                 + 0.957702 w_k1_KO_s107
                                 + 0.892987 w_k1_MAT_s107
                                 + 0.954481 w_k1_MSFT_s107
                                 + 0.874049 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.914851 w_k1_ADM_s110
                                 + 1.31189 w_k1_AXP_s110
                                 + 0.978437 w_k1_DIS_s110
                                 + 1.06961 w_k1_FDX_s110
                                 + 1.12117 w_k1_IBM_s110
                                 + 1.35023 w_k1_INTC_s110
                                 + 1.10893 w_k1_KO_s110 + 1.20197 w_k1_MAT_s110
                                 + 1.03575 w_k1_MSFT_s110
                                 + 0.998238 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.12248 w_k1_ADM_s110
                                 + 1.27744 w_k1_AXP_s110
                                 + 1.25267 w_k1_DIS_s110
                                 + 1.18581 w_k1_FDX_s110
                                 + 0.927671 w_k1_IBM_s110
                                 + 1.13737 w_k1_INTC_s110
                                 + 1.07162 w_k1_KO_s110 + 1.12498 w_k1_MAT_s110
                                 + 1.16592 w_k1_MSFT_s110
                                 + 1.13476 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.06928 w_k1_ADM_s110
                                 + 0.915428 w_k1_AXP_s110
                                 + 0.872613 w_k1_DIS_s110
                                 + 0.830131 w_k1_FDX_s110
                                 + 1.10438 w_k1_IBM_s110
                                 + 0.801598 w_k1_INTC_s110
                                 + 1.01675 w_k1_KO_s110
                                 + 0.926149 w_k1_MAT_s110
                                 + 0.862582 w_k1_MSFT_s110
                                 + 0.995513 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.651755 w_k1_ADM_s110
                                 + 0.604644 w_k1_AXP_s110
                                 + 0.729316 w_k1_DIS_s110
                                 + 0.672137 w_k1_FDX_s110
                                 + 0.880192 w_k1_IBM_s110
                                 + 0.778984 w_k1_INTC_s110
                                 + 0.930972 w_k1_KO_s110
                                 + 1.03466 w_k1_MAT_s110
                                 + 0.861042 w_k1_MSFT_s110
                                 + 0.804426 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.87715 w_k1_ADM_s110
                                 + 1.59113 w_k1_AXP_s110
                                 + 1.50804 w_k1_DIS_s110
                                 + 1.16571 w_k1_FDX_s110
                                 + 1.35642 w_k1_IBM_s110
                                 + 1.17915 w_k1_INTC_s110
                                 + 1.51953 w_k1_KO_s110 + 1.62264 w_k1_MAT_s110
                                 + 0.982709 w_k1_MSFT_s110
                                 + 1.17642 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.998472 w_k1_ADM_s110
                                 + 1.28298 w_k1_AXP_s110
                                 + 1.10597 w_k1_DIS_s110
                                 + 1.10183 w_k1_FDX_s110
                                 + 0.981891 w_k1_IBM_s110
                                 + 1.05496 w_k1_INTC_s110
                                 + 1.06081 w_k1_KO_s110 + 1.56653 w_k1_MAT_s110
                                 + 1.00008 w_k1_MSFT_s110
                                 + 1.08887 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.856773 w_k1_ADM_s110
                                 + 0.932646 w_k1_AXP_s110
                                 + 0.966716 w_k1_DIS_s110
                                 + 1.00837 w_k1_FDX_s110
                                 + 0.710821 w_k1_IBM_s110
                                 + 1.02421 w_k1_INTC_s110
                                 + 0.925399 w_k1_KO_s110
                                 + 0.697488 w_k1_MAT_s110
                                 + 1.082 w_k1_MSFT_s110 + 0.972815 w_k1_PG_s110
                                 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.820549 w_k1_ADM_s110
                                 + 0.927054 w_k1_AXP_s110
                                 + 1.05343 w_k1_DIS_s110
                                 + 0.85489 w_k1_FDX_s110
                                 + 1.62849 w_k1_IBM_s110
                                 + 1.13753 w_k1_INTC_s110
                                 + 1.25704 w_k1_KO_s110 + 1.66787 w_k1_MAT_s110
                                 + 0.941033 w_k1_MSFT_s110
                                 + 1.07987 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.811027 w_k1_ADM_s111
                                 + 0.758931 w_k1_AXP_s111
                                 + 0.896566 w_k1_DIS_s111
                                 + 0.924793 w_k1_FDX_s111
                                 + 0.867082 w_k1_IBM_s111
                                 + 0.879189 w_k1_INTC_s111
                                 + 0.910348 w_k1_KO_s111
                                 + 0.600397 w_k1_MAT_s111
                                 + 0.805522 w_k1_MSFT_s111
                                 + 0.936047 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.01176 w_k1_ADM_s111
                                 + 0.955963 w_k1_AXP_s111
                                 + 0.966612 w_k1_DIS_s111
                                 + 1.02727 w_k1_FDX_s111
                                 + 0.810423 w_k1_IBM_s111
                                 + 0.774444 w_k1_INTC_s111
                                 + 0.798108 w_k1_KO_s111
                                 + 0.820874 w_k1_MAT_s111
                                 + 0.949407 w_k1_MSFT_s111
                                 + 0.897253 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.23111 w_k1_ADM_s111
                                 + 1.43241 w_k1_AXP_s111
                                 + 1.65678 w_k1_DIS_s111
                                 + 1.31412 w_k1_FDX_s111
                                 + 1.30958 w_k1_IBM_s111
                                 + 1.28513 w_k1_INTC_s111
                                 + 1.23297 w_k1_KO_s111 + 1.22504 w_k1_MAT_s111
                                 + 1.22067 w_k1_MSFT_s111
                                 + 1.21881 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.684175 w_k1_ADM_s111
                                 + 0.780312 w_k1_AXP_s111
                                 + 0.6515 w_k1_DIS_s111
                                 + 0.802355 w_k1_FDX_s111
                                 + 0.647627 w_k1_IBM_s111
                                 + 0.727058 w_k1_INTC_s111
                                 + 0.674623 w_k1_KO_s111
                                 + 0.628116 w_k1_MAT_s111
                                 + 0.698087 w_k1_MSFT_s111
                                 + 0.828755 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.969964 w_k1_ADM_s111
                                 + 1.30181 w_k1_AXP_s111
                                 + 0.941941 w_k1_DIS_s111
                                 + 1.07185 w_k1_FDX_s111
                                 + 0.687056 w_k1_IBM_s111
                                 + 0.84293 w_k1_INTC_s111
                                 + 0.834159 w_k1_KO_s111
                                 + 0.972453 w_k1_MAT_s111
                                 + 0.894126 w_k1_MSFT_s111
                                 + 1.04194 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.937764 w_k1_ADM_s111
                                 + 1.0651 w_k1_AXP_s111 + 1.08246 w_k1_DIS_s111
                                 + 1.04363 w_k1_FDX_s111
                                 + 1.05525 w_k1_IBM_s111
                                 + 0.995761 w_k1_INTC_s111
                                 + 1.01618 w_k1_KO_s111 + 1.04783 w_k1_MAT_s111
                                 + 0.949086 w_k1_MSFT_s111
                                 + 1.03736 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.671316 w_k1_ADM_s111
                                 + 0.875653 w_k1_AXP_s111
                                 + 0.72338 w_k1_DIS_s111
                                 + 0.730898 w_k1_FDX_s111
                                 + 0.962067 w_k1_IBM_s111
                                 + 0.740295 w_k1_INTC_s111
                                 + 0.792969 w_k1_KO_s111
                                 + 1.18498 w_k1_MAT_s111
                                 + 0.751308 w_k1_MSFT_s111
                                 + 0.833505 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.942976 w_k1_ADM_s111
                                 + 1.15615 w_k1_AXP_s111
                                 + 0.986819 w_k1_DIS_s111
                                 + 0.885566 w_k1_FDX_s111
                                 + 1.41432 w_k1_IBM_s111
                                 + 0.862155 w_k1_INTC_s111 + 1.104 w_k1_KO_s111
                                 + 1.77265 w_k1_MAT_s111
                                 + 0.784125 w_k1_MSFT_s111
                                 + 1.12268 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.13916 w_k1_ADM_s112
                                 + 1.37503 w_k1_AXP_s112
                                 + 0.988608 w_k1_DIS_s112
                                 + 1.13111 w_k1_FDX_s112
                                 + 0.931392 w_k1_IBM_s112
                                 + 0.953421 w_k1_INTC_s112
                                 + 1.06702 w_k1_KO_s112 + 1.28715 w_k1_MAT_s112
                                 + 0.845174 w_k1_MSFT_s112
                                 + 1.06575 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.42233 w_k1_ADM_s112
                                 + 1.55577 w_k1_AXP_s112
                                 + 2.14071 w_k1_DIS_s112
                                 + 1.46604 w_k1_FDX_s112
                                 + 1.23243 w_k1_IBM_s112
                                 + 1.06837 w_k1_INTC_s112
                                 + 1.53788 w_k1_KO_s112 + 1.05502 w_k1_MAT_s112
                                 + 1.36954 w_k1_MSFT_s112
                                 + 1.19035 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.779622 w_k1_ADM_s112
                                 + 0.731297 w_k1_AXP_s112
                                 + 0.504923 w_k1_DIS_s112
                                 + 0.671087 w_k1_FDX_s112
                                 + 0.786761 w_k1_IBM_s112
                                 + 0.586061 w_k1_INTC_s112
                                 + 0.653199 w_k1_KO_s112
                                 + 0.766573 w_k1_MAT_s112
                                 + 0.630685 w_k1_MSFT_s112
                                 + 0.822655 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.970047 w_k1_ADM_s112
                                 + 0.686152 w_k1_AXP_s112
                                 + 0.74009 w_k1_DIS_s112
                                 + 0.90961 w_k1_FDX_s112
                                 + 0.607842 w_k1_IBM_s112
                                 + 0.731633 w_k1_INTC_s112
                                 + 0.792868 w_k1_KO_s112
                                 + 0.52367 w_k1_MAT_s112
                                 + 0.851883 w_k1_MSFT_s112
                                 + 0.909263 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.07873 w_k1_ADM_s112
                                 + 0.877491 w_k1_AXP_s112
                                 + 0.937928 w_k1_DIS_s112
                                 + 1.03384 w_k1_FDX_s112
                                 + 0.709303 w_k1_IBM_s112
                                 + 0.845348 w_k1_INTC_s112
                                 + 0.948446 w_k1_KO_s112
                                 + 0.805225 w_k1_MAT_s112
                                 + 0.951857 w_k1_MSFT_s112
                                 + 0.918531 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.00046 w_k1_ADM_s112
                                 + 0.891836 w_k1_AXP_s112
                                 + 0.737966 w_k1_DIS_s112
                                 + 0.832226 w_k1_FDX_s112
                                 + 0.965105 w_k1_IBM_s112
                                 + 0.724913 w_k1_INTC_s112
                                 + 0.87306 w_k1_KO_s112 + 1.08073 w_k1_MAT_s112
                                 + 0.837705 w_k1_MSFT_s112
                                 + 0.912458 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.849428 w_k1_ADM_s112
                                 + 0.865283 w_k1_AXP_s112
                                 + 0.982952 w_k1_DIS_s112
                                 + 0.868338 w_k1_FDX_s112
                                 + 1.11745 w_k1_IBM_s112
                                 + 1.03806 w_k1_INTC_s112
                                 + 1.16221 w_k1_KO_s112
                                 + 0.767228 w_k1_MAT_s112
                                 + 1.03481 w_k1_MSFT_s112
                                 + 0.939194 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.1885 w_k1_ADM_s112
                                 + 1.40192 w_k1_AXP_s112
                                 + 1.34429 w_k1_DIS_s112
                                 + 1.18805 w_k1_FDX_s112
                                 + 1.33315 w_k1_IBM_s112
                                 + 1.00772 w_k1_INTC_s112
                                 + 1.61809 w_k1_KO_s112 + 1.44887 w_k1_MAT_s112
                                 + 1.11342 w_k1_MSFT_s112
                                 + 1.23378 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.2575 w_k1_ADM_s113
                                 + 1.47121 w_k1_AXP_s113
                                 + 1.41137 w_k1_DIS_s113
                                 + 1.29099 w_k1_FDX_s113
                                 + 1.53727 w_k1_IBM_s113
                                 + 0.925038 w_k1_INTC_s113
                                 + 1.27484 w_k1_KO_s113 + 1.62395 w_k1_MAT_s113
                                 + 1.16196 w_k1_MSFT_s113
                                 + 1.23788 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.02774 w_k1_ADM_s113
                                 + 1.21637 w_k1_AXP_s113
                                 + 1.42868 w_k1_DIS_s113 + 1.3273 w_k1_FDX_s113
                                 + 0.76088 w_k1_IBM_s113
                                 + 1.19683 w_k1_INTC_s113 + 1.0472 w_k1_KO_s113
                                 + 0.81879 w_k1_MAT_s113
                                 + 1.25087 w_k1_MSFT_s113
                                 + 1.14424 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.10481 w_k1_ADM_s113
                                 + 0.925245 w_k1_AXP_s113
                                 + 1.18711 w_k1_DIS_s113
                                 + 1.10207 w_k1_FDX_s113
                                 + 1.17511 w_k1_IBM_s113
                                 + 1.04764 w_k1_INTC_s113
                                 + 1.19918 w_k1_KO_s113
                                 + 0.986064 w_k1_MAT_s113
                                 + 1.14303 w_k1_MSFT_s113
                                 + 1.11791 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.20425 w_k1_ADM_s113
                                 + 1.13534 w_k1_AXP_s113
                                 + 1.81686 w_k1_DIS_s113 + 1.2602 w_k1_FDX_s113
                                 + 1.25104 w_k1_IBM_s113
                                 + 1.33561 w_k1_INTC_s113
                                 + 1.48931 w_k1_KO_s113 + 1.0556 w_k1_MAT_s113
                                 + 1.36295 w_k1_MSFT_s113
                                 + 1.28744 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.767223 w_k1_ADM_s113
                                 + 0.752187 w_k1_AXP_s113
                                 + 0.621978 w_k1_DIS_s113
                                 + 0.704231 w_k1_FDX_s113
                                 + 0.86563 w_k1_IBM_s113
                                 + 0.773965 w_k1_INTC_s113
                                 + 0.815161 w_k1_KO_s113
                                 + 0.637766 w_k1_MAT_s113
                                 + 0.766987 w_k1_MSFT_s113
                                 + 0.934811 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.09856 w_k1_ADM_s113
                                 + 0.963868 w_k1_AXP_s113
                                 + 1.1194 w_k1_DIS_s113
                                 + 0.911459 w_k1_FDX_s113
                                 + 1.94418 w_k1_IBM_s113
                                 + 1.10565 w_k1_INTC_s113
                                 + 1.50874 w_k1_KO_s113 + 1.8068 w_k1_MAT_s113
                                 + 1.01788 w_k1_MSFT_s113
                                 + 1.05135 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.04052 w_k1_ADM_s113
                                 + 0.801917 w_k1_AXP_s113
                                 + 0.961641 w_k1_DIS_s113
                                 + 0.985421 w_k1_FDX_s113
                                 + 0.728433 w_k1_IBM_s113
                                 + 1.06513 w_k1_INTC_s113
                                 + 0.941404 w_k1_KO_s113
                                 + 0.648158 w_k1_MAT_s113
                                 + 1.02447 w_k1_MSFT_s113
                                 + 0.982049 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.872241 w_k1_ADM_s113
                                 + 1.01528 w_k1_AXP_s113
                                 + 1.07187 w_k1_DIS_s113
                                 + 0.941616 w_k1_FDX_s113
                                 + 1.24154 w_k1_IBM_s113
                                 + 0.89314 w_k1_INTC_s113
                                 + 1.10583 w_k1_KO_s113 + 1.3554 w_k1_MAT_s113
                                 + 1.07539 w_k1_MSFT_s113 + 1.0574 w_k1_PG_s113
                                 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.41602 w_k1_ADM_s114
                                 + 1.07448 w_k1_AXP_s114
                                 + 1.14454 w_k1_DIS_s114
                                 + 1.05873 w_k1_FDX_s114
                                 + 1.53553 w_k1_IBM_s114
                                 + 1.16372 w_k1_INTC_s114
                                 + 1.29159 w_k1_KO_s114
                                 + 0.988109 w_k1_MAT_s114
                                 + 1.16312 w_k1_MSFT_s114
                                 + 1.00696 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.06489 w_k1_ADM_s114
                                 + 0.79199 w_k1_AXP_s114
                                 + 0.860099 w_k1_DIS_s114
                                 + 0.898939 w_k1_FDX_s114
                                 + 0.745187 w_k1_IBM_s114
                                 + 0.822784 w_k1_INTC_s114
                                 + 0.941294 w_k1_KO_s114
                                 + 0.652001 w_k1_MAT_s114
                                 + 0.89786 w_k1_MSFT_s114
                                 + 0.908743 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.22557 w_k1_ADM_s114
                                 + 1.0912 w_k1_AXP_s114 + 1.19118 w_k1_DIS_s114
                                 + 1.20994 w_k1_FDX_s114
                                 + 0.668858 w_k1_IBM_s114
                                 + 1.12729 w_k1_INTC_s114
                                 + 0.957171 w_k1_KO_s114
                                 + 0.447742 w_k1_MAT_s114
                                 + 1.17676 w_k1_MSFT_s114
                                 + 1.00127 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.12907 w_k1_ADM_s114
                                 + 1.26617 w_k1_AXP_s114
                                 + 1.46522 w_k1_DIS_s114
                                 + 1.23563 w_k1_FDX_s114
                                 + 1.20508 w_k1_IBM_s114
                                 + 1.43527 w_k1_INTC_s114
                                 + 1.35153 w_k1_KO_s114
                                 + 0.875518 w_k1_MAT_s114
                                 + 1.2773 w_k1_MSFT_s114 + 1.19843 w_k1_PG_s114
                                 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.05263 w_k1_ADM_s114
                                 + 1.03328 w_k1_AXP_s114 + 1.131 w_k1_DIS_s114
                                 + 1.12251 w_k1_FDX_s114
                                 + 0.833822 w_k1_IBM_s114
                                 + 1.13067 w_k1_INTC_s114
                                 + 1.09995 w_k1_KO_s114
                                 + 0.769445 w_k1_MAT_s114
                                 + 1.12606 w_k1_MSFT_s114
                                 + 1.08815 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.47517 w_k1_ADM_s114
                                 + 1.23206 w_k1_AXP_s114
                                 + 1.14462 w_k1_DIS_s114
                                 + 1.13576 w_k1_FDX_s114
                                 + 0.960112 w_k1_IBM_s114
                                 + 1.14114 w_k1_INTC_s114
                                 + 1.03782 w_k1_KO_s114
                                 + 0.844595 w_k1_MAT_s114
                                 + 1.0754 w_k1_MSFT_s114 + 1.14144 w_k1_PG_s114
                                 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.1733 w_k1_ADM_s114
                                 + 0.765748 w_k1_AXP_s114
                                 + 0.935597 w_k1_DIS_s114
                                 + 0.908492 w_k1_FDX_s114
                                 + 1.0923 w_k1_IBM_s114
                                 + 1.00838 w_k1_INTC_s114
                                 + 1.08016 w_k1_KO_s114
                                 + 0.767222 w_k1_MAT_s114
                                 + 1.00054 w_k1_MSFT_s114
                                 + 0.962387 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.0397 w_k1_ADM_s114
                                 + 0.740349 w_k1_AXP_s114
                                 + 0.662773 w_k1_DIS_s114
                                 + 0.760017 w_k1_FDX_s114
                                 + 0.8248 w_k1_IBM_s114
                                 + 0.836743 w_k1_INTC_s114
                                 + 0.839026 w_k1_KO_s114
                                 + 0.919056 w_k1_MAT_s114
                                 + 0.822917 w_k1_MSFT_s114
                                 + 0.75933 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.908886 w_k1_ADM_s115
                                 + 0.629214 w_k1_AXP_s115
                                 + 0.657325 w_k1_DIS_s115
                                 + 0.824274 w_k1_FDX_s115
                                 + 0.661979 w_k1_IBM_s115
                                 + 0.831461 w_k1_INTC_s115
                                 + 0.59678 w_k1_KO_s115
                                 + 0.562299 w_k1_MAT_s115
                                 + 0.847237 w_k1_MSFT_s115
                                 + 0.805509 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.856025 w_k1_ADM_s115
                                 + 0.715817 w_k1_AXP_s115
                                 + 0.75106 w_k1_DIS_s115
                                 + 0.875798 w_k1_FDX_s115
                                 + 1.04301 w_k1_IBM_s115
                                 + 0.816328 w_k1_INTC_s115
                                 + 0.911586 w_k1_KO_s115
                                 + 1.04704 w_k1_MAT_s115
                                 + 0.945726 w_k1_MSFT_s115
                                 + 0.893698 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.32097 w_k1_ADM_s115
                                 + 1.1289 w_k1_AXP_s115 + 1.07649 w_k1_DIS_s115
                                 + 1.20739 w_k1_FDX_s115
                                 + 1.04405 w_k1_IBM_s115
                                 + 1.12735 w_k1_INTC_s115
                                 + 1.03859 w_k1_KO_s115 + 1.49445 w_k1_MAT_s115
                                 + 1.20504 w_k1_MSFT_s115
                                 + 1.09107 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.577001 w_k1_ADM_s115
                                 + 0.694802 w_k1_AXP_s115
                                 + 0.503686 w_k1_DIS_s115
                                 + 0.725037 w_k1_FDX_s115
                                 + 0.905204 w_k1_IBM_s115
                                 + 0.872341 w_k1_INTC_s115
                                 + 0.63166 w_k1_KO_s115
                                 + 0.793909 w_k1_MAT_s115
                                 + 0.717806 w_k1_MSFT_s115
                                 + 0.778758 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.963548 w_k1_ADM_s115
                                 + 0.891001 w_k1_AXP_s115
                                 + 0.968975 w_k1_DIS_s115
                                 + 1.0682 w_k1_FDX_s115
                                 + 0.768592 w_k1_IBM_s115
                                 + 0.983033 w_k1_INTC_s115
                                 + 0.800238 w_k1_KO_s115
                                 + 0.706581 w_k1_MAT_s115
                                 + 1.10961 w_k1_MSFT_s115
                                 + 1.01937 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.899512 w_k1_ADM_s115
                                 + 1.42118 w_k1_AXP_s115
                                 + 1.19234 w_k1_DIS_s115
                                 + 1.20981 w_k1_FDX_s115 + 1.3071 w_k1_IBM_s115
                                 + 1.09766 w_k1_INTC_s115
                                 + 1.00502 w_k1_KO_s115 + 1.48205 w_k1_MAT_s115
                                 + 1.15643 w_k1_MSFT_s115
                                 + 1.19507 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.829083 w_k1_ADM_s115
                                 + 0.780401 w_k1_AXP_s115
                                 + 0.740011 w_k1_DIS_s115
                                 + 0.77942 w_k1_FDX_s115
                                 + 1.44793 w_k1_IBM_s115
                                 + 0.855293 w_k1_INTC_s115
                                 + 1.00965 w_k1_KO_s115 + 1.31851 w_k1_MAT_s115
                                 + 0.848546 w_k1_MSFT_s115
                                 + 1.07975 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.22468 w_k1_ADM_s115
                                 + 1.57773 w_k1_AXP_s115
                                 + 1.58743 w_k1_DIS_s115
                                 + 1.42771 w_k1_FDX_s115
                                 + 1.42525 w_k1_IBM_s115
                                 + 1.32265 w_k1_INTC_s115
                                 + 1.37785 w_k1_KO_s115 + 1.50076 w_k1_MAT_s115
                                 + 1.28824 w_k1_MSFT_s115
                                 + 1.47065 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.838898 w_k1_ADM_s116
                                 + 0.87289 w_k1_AXP_s116
                                 + 0.769378 w_k1_DIS_s116
                                 + 0.79401 w_k1_FDX_s116
                                 + 0.919893 w_k1_IBM_s116
                                 + 0.935622 w_k1_INTC_s116
                                 + 0.742191 w_k1_KO_s116
                                 + 0.855731 w_k1_MAT_s116
                                 + 0.762967 w_k1_MSFT_s116
                                 + 0.867357 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.779891 w_k1_ADM_s116
                                 + 1.02157 w_k1_AXP_s116
                                 + 1.09577 w_k1_DIS_s116
                                 + 0.924415 w_k1_FDX_s116
                                 + 1.28576 w_k1_IBM_s116
                                 + 1.0679 w_k1_INTC_s116 + 1.08327 w_k1_KO_s116
                                 + 1.12931 w_k1_MAT_s116
                                 + 0.973151 w_k1_MSFT_s116
                                 + 0.974529 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.725163 w_k1_ADM_s116
                                 + 0.830083 w_k1_AXP_s116
                                 + 0.743595 w_k1_DIS_s116
                                 + 0.713401 w_k1_FDX_s116
                                 + 1.03805 w_k1_IBM_s116
                                 + 1.01156 w_k1_INTC_s116
                                 + 0.741404 w_k1_KO_s116
                                 + 0.661593 w_k1_MAT_s116
                                 + 0.824623 w_k1_MSFT_s116
                                 + 0.822912 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.86414 w_k1_ADM_s116
                                 + 1.17133 w_k1_AXP_s116
                                 + 1.04768 w_k1_DIS_s116
                                 + 0.934177 w_k1_FDX_s116
                                 + 0.94759 w_k1_IBM_s116
                                 + 1.10747 w_k1_INTC_s116
                                 + 0.831558 w_k1_KO_s116
                                 + 0.964496 w_k1_MAT_s116
                                 + 0.91486 w_k1_MSFT_s116
                                 + 0.88134 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.0162 w_k1_ADM_s116
                                 + 0.919442 w_k1_AXP_s116
                                 + 0.951541 w_k1_DIS_s116
                                 + 0.967538 w_k1_FDX_s116
                                 + 0.621319 w_k1_IBM_s116
                                 + 0.968242 w_k1_INTC_s116
                                 + 0.781631 w_k1_KO_s116
                                 + 0.591297 w_k1_MAT_s116
                                 + 0.94094 w_k1_MSFT_s116
                                 + 0.936458 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.85893 w_k1_ADM_s116
                                 + 1.42432 w_k1_AXP_s116
                                 + 1.44023 w_k1_DIS_s116
                                 + 1.08168 w_k1_FDX_s116
                                 + 1.67453 w_k1_IBM_s116
                                 + 1.5658 w_k1_INTC_s116 + 1.13473 w_k1_KO_s116
                                 + 1.32367 w_k1_MAT_s116
                                 + 1.00768 w_k1_MSFT_s116
                                 + 1.11012 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.957861 w_k1_ADM_s116
                                 + 1.30218 w_k1_AXP_s116
                                 + 1.39502 w_k1_DIS_s116
                                 + 1.18318 w_k1_FDX_s116
                                 + 1.04453 w_k1_IBM_s116
                                 + 1.42488 w_k1_INTC_s116
                                 + 1.03312 w_k1_KO_s116
                                 + 0.987209 w_k1_MAT_s116
                                 + 1.12433 w_k1_MSFT_s116 + 1.0872 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.720615 w_k1_ADM_s116
                                 + 0.749 w_k1_AXP_s116 + 0.966244 w_k1_DIS_s116
                                 + 0.952651 w_k1_FDX_s116
                                 + 0.897247 w_k1_IBM_s116
                                 + 1.10101 w_k1_INTC_s116
                                 + 0.891462 w_k1_KO_s116
                                 + 0.71318 w_k1_MAT_s116
                                 + 0.96835 w_k1_MSFT_s116
                                 + 0.950246 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.05991 w_k1_ADM_s117
                                 + 1.0021 w_k1_AXP_s117
                                 + 0.945509 w_k1_DIS_s117
                                 + 1.03722 w_k1_FDX_s117
                                 + 0.787967 w_k1_IBM_s117
                                 + 0.863871 w_k1_INTC_s117
                                 + 0.797877 w_k1_KO_s117
                                 + 0.86762 w_k1_MAT_s117
                                 + 0.971403 w_k1_MSFT_s117
                                 + 0.809999 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.02273 w_k1_ADM_s117
                                 + 1.25275 w_k1_AXP_s117
                                 + 1.55546 w_k1_DIS_s117
                                 + 1.05904 w_k1_FDX_s117
                                 + 1.78784 w_k1_IBM_s117
                                 + 1.10534 w_k1_INTC_s117
                                 + 1.35064 w_k1_KO_s117 + 1.37595 w_k1_MAT_s117
                                 + 1.10805 w_k1_MSFT_s117
                                 + 1.20368 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.798659 w_k1_ADM_s117
                                 + 0.575499 w_k1_AXP_s117
                                 + 0.725209 w_k1_DIS_s117
                                 + 0.760353 w_k1_FDX_s117
                                 + 0.743745 w_k1_IBM_s117
                                 + 0.814401 w_k1_INTC_s117
                                 + 0.702209 w_k1_KO_s117
                                 + 0.594057 w_k1_MAT_s117
                                 + 0.849859 w_k1_MSFT_s117
                                 + 0.839223 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.0185 w_k1_ADM_s117
                                 + 1.41684 w_k1_AXP_s117 + 1.5481 w_k1_DIS_s117
                                 + 1.21097 w_k1_FDX_s117
                                 + 1.25629 w_k1_IBM_s117
                                 + 1.06418 w_k1_INTC_s117
                                 + 1.01267 w_k1_KO_s117 + 1.19131 w_k1_MAT_s117
                                 + 1.03384 w_k1_MSFT_s117
                                 + 1.05383 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.907384 w_k1_ADM_s117
                                 + 0.839287 w_k1_AXP_s117
                                 + 0.959365 w_k1_DIS_s117
                                 + 0.913326 w_k1_FDX_s117
                                 + 1.16585 w_k1_IBM_s117
                                 + 0.899938 w_k1_INTC_s117
                                 + 0.995814 w_k1_KO_s117
                                 + 1.07048 w_k1_MAT_s117
                                 + 1.03668 w_k1_MSFT_s117
                                 + 0.944463 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.0102 w_k1_ADM_s117
                                 + 1.2071 w_k1_AXP_s117 + 1.70553 w_k1_DIS_s117
                                 + 1.25299 w_k1_FDX_s117
                                 + 1.17287 w_k1_IBM_s117
                                 + 1.51261 w_k1_INTC_s117
                                 + 1.16359 w_k1_KO_s117
                                 + 0.945823 w_k1_MAT_s117
                                 + 1.44642 w_k1_MSFT_s117
                                 + 1.17786 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.698671 w_k1_ADM_s117
                                 + 0.901006 w_k1_AXP_s117
                                 + 0.738464 w_k1_DIS_s117
                                 + 0.780027 w_k1_FDX_s117
                                 + 0.973947 w_k1_IBM_s117
                                 + 0.944751 w_k1_INTC_s117
                                 + 0.77899 w_k1_KO_s117
                                 + 0.952836 w_k1_MAT_s117
                                 + 0.881503 w_k1_MSFT_s117
                                 + 0.901497 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.02753 w_k1_ADM_s117
                                 + 0.932451 w_k1_AXP_s117
                                 + 1.36042 w_k1_DIS_s117 + 1.1213 w_k1_FDX_s117
                                 + 0.988617 w_k1_IBM_s117
                                 + 1.07201 w_k1_INTC_s117
                                 + 1.06694 w_k1_KO_s117
                                 + 0.673295 w_k1_MAT_s117
                                 + 1.16016 w_k1_MSFT_s117
                                 + 0.890703 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0255979265185785 w_k1_ADM_s000
                                 + 0.0217848244235935 w_k1_ADM_s001
                                 + 0.0175075885710605 w_k1_ADM_s002
                                 + 0.0202069020411767 w_k1_ADM_s003
                                 + 0.0404682441493798 w_k1_ADM_s004
                                 + 0.0567033845555492 w_k1_ADM_s005
                                 + 0.0381497285545571 w_k1_ADM_s006
                                 + 0.0153347848264693 w_k1_ADM_s007
                                 + 0.0300748858843207 w_k1_ADM_s010
                                 + 0.00863441514206626 w_k1_ADM_s011
                                 + 0.00883292175365817 w_k1_ADM_s012
                                 + 0.0201223262557267 w_k1_ADM_s013
                                 + 0.021188486074806 w_k1_ADM_s014
                                 + 0.0108372748536499 w_k1_ADM_s015
                                 + 0.0256095279896706 w_k1_ADM_s016
                                 + 0.011475999290633 w_k1_ADM_s017
                                 + 0.051749604575302 w_k1_ADM_s100
                                 + 0.103105864057556 w_k1_ADM_s101
                                 + 0.0443480410035767 w_k1_ADM_s102
                                 + 0.0805505714030092 w_k1_ADM_s103
                                 + 0.0512335247630056 w_k1_ADM_s104
                                 + 0.0460360107416317 w_k1_ADM_s105
                                 + 0.019908075757129 w_k1_ADM_s106
                                 + 0.0212895060477533 w_k1_ADM_s107
                                 + 0.0200338667250182 w_k1_ADM_s110
                                 + 0.0490174286321933 w_k1_ADM_s111
                                 + 0.0165278341287337 w_k1_ADM_s112
                                 + 0.0476814760696489 w_k1_ADM_s113
                                 + 0.0188713325763274 w_k1_ADM_s114
                                 + 0.0295765358093686 w_k1_ADM_s115
                                 + 0.0251839819152513 w_k1_ADM_s116
                                 + 0.0162794951167006 w_k1_ADM_s117
                                 + 0.029446476194708 w_k1_AXP_s000
                                 + 0.0210010235412981 w_k1_AXP_s001
                                 + 0.0194817433773521 w_k1_AXP_s002
                                 + 0.0192270368983931 w_k1_AXP_s003
                                 + 0.0399394843936609 w_k1_AXP_s004
                                 + 0.0552311336436844 w_k1_AXP_s005
                                 + 0.0374599672579685 w_k1_AXP_s006
                                 + 0.0130457423044552 w_k1_AXP_s007
                                 + 0.0296190409717375 w_k1_AXP_s010
                                 + 0.0081725172267181 w_k1_AXP_s011
                                 + 0.00791661054047255 w_k1_AXP_s012
                                 + 0.0184000245620633 w_k1_AXP_s013
                                 + 0.023174198658274 w_k1_AXP_s014
                                 + 0.0125072288237214 w_k1_AXP_s015
                                 + 0.0238493914503653 w_k1_AXP_s016
                                 + 0.0130600799900075 w_k1_AXP_s017
                                 + 0.0515315000812022 w_k1_AXP_s100
                                 + 0.1025443795836 w_k1_AXP_s101
                                 + 0.050523999300248 w_k1_AXP_s102
                                 + 0.0842140852920028 w_k1_AXP_s103
                                 + 0.0532958736396003 w_k1_AXP_s104
                                 + 0.0468207108191152 w_k1_AXP_s105
                                 + 0.0209685568957007 w_k1_AXP_s106
                                 + 0.0193638033619966 w_k1_AXP_s107
                                 + 0.0234963029652169 w_k1_AXP_s110
                                 + 0.056020344177326 w_k1_AXP_s111
                                 + 0.0171327873315411 w_k1_AXP_s112
                                 + 0.0455997845752115 w_k1_AXP_s113
                                 + 0.0155083555685593 w_k1_AXP_s114
                                 + 0.0292869282850344 w_k1_AXP_s115
                                 + 0.0313982893340971 w_k1_AXP_s116
                                 + 0.0170055766437313 w_k1_AXP_s117
                                 + 0.0300577571352307 w_k1_DIS_s000
                                 + 0.022312613186106 w_k1_DIS_s001
                                 + 0.0182365765004269 w_k1_DIS_s002
                                 + 0.0199617951055701 w_k1_DIS_s003
                                 + 0.0435679105005118 w_k1_DIS_s004
                                 + 0.0575804454669377 w_k1_DIS_s005
                                 + 0.0398848207228255 w_k1_DIS_s006
                                 + 0.014115935313512 w_k1_DIS_s007
                                 + 0.0298719618262654 w_k1_DIS_s010
                                 + 0.0080097479523777 w_k1_DIS_s011
                                 + 0.0091361553646868 w_k1_DIS_s012
                                 + 0.0201423521755454 w_k1_DIS_s013
                                 + 0.0230525984880687 w_k1_DIS_s014
                                 + 0.0119213498506422 w_k1_DIS_s015
                                 + 0.0255572975350377 w_k1_DIS_s016
                                 + 0.0133310798359635 w_k1_DIS_s017
                                 + 0.0418745282899729 w_k1_DIS_s100
                                 + 0.100017656404878 w_k1_DIS_s101
                                 + 0.0449484933448682 w_k1_DIS_s102
                                 + 0.0907210519537636 w_k1_DIS_s103
                                 + 0.0524686334351636 w_k1_DIS_s104
                                 + 0.0465045506093747 w_k1_DIS_s105
                                 + 0.0207538085002303 w_k1_DIS_s106
                                 + 0.0203802191861681 w_k1_DIS_s107
                                 + 0.0226888418800894 w_k1_DIS_s110
                                 + 0.0549390502574014 w_k1_DIS_s111
                                 + 0.0167047643894908 w_k1_DIS_s112
                                 + 0.0519683351401964 w_k1_DIS_s113
                                 + 0.0166743379137406 w_k1_DIS_s114
                                 + 0.0288226506072736 w_k1_DIS_s115
                                 + 0.0326556408793787 w_k1_DIS_s116
                                 + 0.0194195817657423 w_k1_DIS_s117
                                 + 0.0288741047268249 w_k1_FDX_s000
                                 + 0.0228370142173095 w_k1_FDX_s001
                                 + 0.018889271220211 w_k1_FDX_s002
                                 + 0.0208068191985042 w_k1_FDX_s003
                                 + 0.0410920974764964 w_k1_FDX_s004
                                 + 0.0556365042363388 w_k1_FDX_s005
                                 + 0.0385715401602212 w_k1_FDX_s006
                                 + 0.0139308119365532 w_k1_FDX_s007
                                 + 0.0283029819080177 w_k1_FDX_s010
                                 + 0.00866860547420553 w_k1_FDX_s011
                                 + 0.00900328240292914 w_k1_FDX_s012
                                 + 0.0194980426022518 w_k1_FDX_s013
                                 + 0.0220695223408457 w_k1_FDX_s014
                                 + 0.0122063470375534 w_k1_FDX_s015
                                 + 0.0249921173890251 w_k1_FDX_s016
                                 + 0.0136259129804509 w_k1_FDX_s017
                                 + 0.0454850515056962 w_k1_FDX_s100
                                 + 0.0986334283584329 w_k1_FDX_s101
                                 + 0.0433392211811983 w_k1_FDX_s102
                                 + 0.0812749190019915 w_k1_FDX_s103
                                 + 0.0507342650762225 w_k1_FDX_s104
                                 + 0.0444141045405336 w_k1_FDX_s105
                                 + 0.0205485691389585 w_k1_FDX_s106
                                 + 0.0199703672701814 w_k1_FDX_s107
                                 + 0.0213543492088536 w_k1_FDX_s110
                                 + 0.0531979393887529 w_k1_FDX_s111
                                 + 0.0159492686048148 w_k1_FDX_s112
                                 + 0.0481359253586598 w_k1_FDX_s113
                                 + 0.0160772710268225 w_k1_FDX_s114
                                 + 0.0313014891295002 w_k1_FDX_s115
                                 + 0.0288789521511542 w_k1_FDX_s116
                                 + 0.0173164892047068 w_k1_FDX_s117
                                 + 0.0289269364558731 w_k1_IBM_s000
                                 + 0.0240476126896737 w_k1_IBM_s001
                                 + 0.0191690598962366 w_k1_IBM_s002
                                 + 0.0201781473297508 w_k1_IBM_s003
                                 + 0.0416063569474654 w_k1_IBM_s004
                                 + 0.0577121736043274 w_k1_IBM_s005
                                 + 0.0392819094034982 w_k1_IBM_s006
                                 + 0.0144156495791848 w_k1_IBM_s007
                                 + 0.0297670842765204 w_k1_IBM_s010
                                 + 0.00849914689896196 w_k1_IBM_s011
                                 + 0.00858682334046015 w_k1_IBM_s012
                                 + 0.0191551750264412 w_k1_IBM_s013
                                 + 0.0214017026488295 w_k1_IBM_s014
                                 + 0.0123119218452546 w_k1_IBM_s015
                                 + 0.0235503123767422 w_k1_IBM_s016
                                 + 0.013323836553879 w_k1_IBM_s017
                                 + 0.0450557729508152 w_k1_IBM_s100
                                 + 0.0982236329317365 w_k1_IBM_s101
                                 + 0.0471414492653345 w_k1_IBM_s102
                                 + 0.0812151089973021 w_k1_IBM_s103
                                 + 0.0506349597165343 w_k1_IBM_s104
                                 + 0.0425317455555896 w_k1_IBM_s105
                                 + 0.0211137234032674 w_k1_IBM_s106
                                 + 0.0194150882958013 w_k1_IBM_s107
                                 + 0.021666195485767 w_k1_IBM_s110
                                 + 0.0539813820561098 w_k1_IBM_s111
                                 + 0.0161827828256815 w_k1_IBM_s112
                                 + 0.0506043999204783 w_k1_IBM_s113
                                 + 0.0160583856609695 w_k1_IBM_s114
                                 + 0.03122531280811 w_k1_IBM_s115
                                 + 0.0307299665020723 w_k1_IBM_s116
                                 + 0.0179237248313047 w_k1_IBM_s117
                                 + 0.0269805398996693 w_k1_INTC_s000
                                 + 0.0239130866310111 w_k1_INTC_s001
                                 + 0.0164367807884016 w_k1_INTC_s002
                                 + 0.0202093174580844 w_k1_INTC_s003
                                 + 0.0435686678935442 w_k1_INTC_s004
                                 + 0.0548646057325402 w_k1_INTC_s005
                                 + 0.0401158004097668 w_k1_INTC_s006
                                 + 0.0120052896107307 w_k1_INTC_s007
                                 + 0.0265940696708797 w_k1_INTC_s010
                                 + 0.00810823979805733 w_k1_INTC_s011
                                 + 0.0104981022634457 w_k1_INTC_s012
                                 + 0.0230299570190684 w_k1_INTC_s013
                                 + 0.0236413638635534 w_k1_INTC_s014
                                 + 0.0142820542907465 w_k1_INTC_s015
                                 + 0.021649215498065 w_k1_INTC_s016
                                 + 0.0122365056641508 w_k1_INTC_s017
                                 + 0.0439920375309152 w_k1_INTC_s100
                                 + 0.0927295707866903 w_k1_INTC_s101
                                 + 0.0460251128048416 w_k1_INTC_s102
                                 + 0.0834169253502817 w_k1_INTC_s103
                                 + 0.0479199164862576 w_k1_INTC_s104
                                 + 0.0470628156806734 w_k1_INTC_s105
                                 + 0.0265662499006046 w_k1_INTC_s106
                                 + 0.021934725637693 w_k1_INTC_s107
                                 + 0.0220118426352782 w_k1_INTC_s110
                                 + 0.049719207734779 w_k1_INTC_s111
                                 + 0.014024043174195 w_k1_INTC_s112
                                 + 0.0480253929149398 w_k1_INTC_s113
                                 + 0.0169333163767081 w_k1_INTC_s114
                                 + 0.0299157575927304 w_k1_INTC_s115
                                 + 0.0346729255192429 w_k1_INTC_s116
                                 + 0.0167853424547959 w_k1_INTC_s117
                                 + 0.0304876286279007 w_k1_KO_s000
                                 + 0.0221322545627785 w_k1_KO_s001
                                 + 0.017850783855092 w_k1_KO_s002
                                 + 0.0207278435090557 w_k1_KO_s003
                                 + 0.0445979989916993 w_k1_KO_s004
                                 + 0.0588820431752921 w_k1_KO_s005
                                 + 0.0385498569589135 w_k1_KO_s006
                                 + 0.0139025261135471 w_k1_KO_s007
                                 + 0.0289172630160363 w_k1_KO_s010
                                 + 0.00823661508598639 w_k1_KO_s011
                                 + 0.0089531425409624 w_k1_KO_s012
                                 + 0.0225962656960434 w_k1_KO_s013
                                 + 0.0216685238446663 w_k1_KO_s014
                                 + 0.0124171267590017 w_k1_KO_s015
                                 + 0.0239707472593855 w_k1_KO_s016
                                 + 0.0130797233954955 w_k1_KO_s017
                                 + 0.0469273634094487 w_k1_KO_s100
                                 + 0.103352137137854 w_k1_KO_s101
                                 + 0.0465536540982134 w_k1_KO_s102
                                 + 0.0838764279265191 w_k1_KO_s103
                                 + 0.0519772032575189 w_k1_KO_s104
                                 + 0.0437455038551966 w_k1_KO_s105
                                 + 0.0203890674808775 w_k1_KO_s106
                                 + 0.0210832069712414 w_k1_KO_s107
                                 + 0.0230812407784594 w_k1_KO_s110
                                 + 0.0512170161364034 w_k1_KO_s111
                                 + 0.0181397327974985 w_k1_KO_s112
                                 + 0.0516032155695631 w_k1_KO_s113
                                 + 0.0171195498704385 w_k1_KO_s114
                                 + 0.0277788104106479 w_k1_KO_s115
                                 + 0.02743155166585 w_k1_KO_s116
                                 + 0.0161634634811456 w_k1_KO_s117
                                 + 0.0366032235320077 w_k1_MAT_s000
                                 + 0.0211402515207366 w_k1_MAT_s001
                                 + 0.0198091359634259 w_k1_MAT_s002
                                 + 0.0199393627769123 w_k1_MAT_s003
                                 + 0.0373954810671346 w_k1_MAT_s004
                                 + 0.0583329622960636 w_k1_MAT_s005
                                 + 0.043492924995518 w_k1_MAT_s006
                                 + 0.013836784501209 w_k1_MAT_s007
                                 + 0.0386439867624913 w_k1_MAT_s010
                                 + 0.00783106851562704 w_k1_MAT_s011
                                 + 0.00714229442117471 w_k1_MAT_s012
                                 + 0.018740948450953 w_k1_MAT_s013
                                 + 0.0206477503439639 w_k1_MAT_s014
                                 + 0.0116513805804102 w_k1_MAT_s015
                                 + 0.0245681486188412 w_k1_MAT_s016
                                 + 0.0148722084566107 w_k1_MAT_s017
                                 + 0.0534568656423856 w_k1_MAT_s100
                                 + 0.117678522266488 w_k1_MAT_s101
                                 + 0.0561783269308637 w_k1_MAT_s102
                                 + 0.0869546667754996 w_k1_MAT_s103
                                 + 0.0549196723564692 w_k1_MAT_s104
                                 + 0.0569246633129141 w_k1_MAT_s105
                                 + 0.0182396332108389 w_k1_MAT_s106
                                 + 0.0198613152464021 w_k1_MAT_s107
                                 + 0.0240077409376504 w_k1_MAT_s110
                                 + 0.0554599374337832 w_k1_MAT_s111
                                 + 0.016480659624581 w_k1_MAT_s112
                                 + 0.0468580033908398 w_k1_MAT_s113
                                 + 0.0125773295563874 w_k1_MAT_s114
                                 + 0.0325416800709895 w_k1_MAT_s115
                                 + 0.0269013641358705 w_k1_MAT_s116
                                 + 0.016068545277526 w_k1_MAT_s117
                                 + 0.031372565998258 w_k1_MSFT_s000
                                 + 0.0218776282712652 w_k1_MSFT_s001
                                 + 0.0176290272975928 w_k1_MSFT_s002
                                 + 0.0214957704295429 w_k1_MSFT_s003
                                 + 0.04109319180369 w_k1_MSFT_s004
                                 + 0.0584901618542266 w_k1_MSFT_s005
                                 + 0.0402315350594604 w_k1_MSFT_s006
                                 + 0.012706406731614 w_k1_MSFT_s007
                                 + 0.0267236493592521 w_k1_MSFT_s010
                                 + 0.00798445425539985 w_k1_MSFT_s011
                                 + 0.00869442341891907 w_k1_MSFT_s012
                                 + 0.0194487721992713 w_k1_MSFT_s013
                                 + 0.0218178976925442 w_k1_MSFT_s014
                                 + 0.0124471366279575 w_k1_MSFT_s015
                                 + 0.0231242973757376 w_k1_MSFT_s016
                                 + 0.0129967359154862 w_k1_MSFT_s017
                                 + 0.0435171556937549 w_k1_MSFT_s100
                                 + 0.105024274777224 w_k1_MSFT_s101
                                 + 0.0427286854739089 w_k1_MSFT_s102
                                 + 0.0819437294251764 w_k1_MSFT_s103
                                 + 0.0555894695367461 w_k1_MSFT_s104
                                 + 0.0462306470637826 w_k1_MSFT_s105
                                 + 0.0207823214537183 w_k1_MSFT_s106
                                 + 0.0220387415595192 w_k1_MSFT_s107
                                 + 0.0214216100370107 w_k1_MSFT_s110
                                 + 0.0481027485921666 w_k1_MSFT_s111
                                 + 0.0152020379207707 w_k1_MSFT_s112
                                 + 0.0500307178374927 w_k1_MSFT_s113
                                 + 0.0165807011596098 w_k1_MSFT_s114
                                 + 0.0316580064636653 w_k1_MSFT_s115
                                 + 0.0285232361843892 w_k1_MSFT_s116
                                 + 0.0176707869854905 w_k1_MSFT_s117
                                 + 0.0296089783932341 w_k1_PG_s000
                                 + 0.0220572609202058 w_k1_PG_s001
                                 + 0.018336005487743 w_k1_PG_s002
                                 + 0.0186043352887168 w_k1_PG_s003
                                 + 0.0410929218214334 w_k1_PG_s004
                                 + 0.0592283616444105 w_k1_PG_s005
                                 + 0.0383836668377341 w_k1_PG_s006
                                 + 0.0133549336404553 w_k1_PG_s007
                                 + 0.0294242380229863 w_k1_PG_s010
                                 + 0.00804761435028197 w_k1_PG_s011
                                 + 0.00926490920304971 w_k1_PG_s012
                                 + 0.0202555943579819 w_k1_PG_s013
                                 + 0.022670851100597 w_k1_PG_s014
                                 + 0.0116929382127982 w_k1_PG_s015
                                 + 0.0253465487991341 w_k1_PG_s016
                                 + 0.012206027050002 w_k1_PG_s017
                                 + 0.0432593853413672 w_k1_PG_s100
                                 + 0.100719970417819 w_k1_PG_s101
                                 + 0.0407577620859946 w_k1_PG_s102
                                 + 0.078613269933486 w_k1_PG_s103
                                 + 0.0479037828525827 w_k1_PG_s104
                                 + 0.0431664353173426 w_k1_PG_s105
                                 + 0.0195364853817733 w_k1_PG_s106
                                 + 0.0189222751660676 w_k1_PG_s107
                                 + 0.0220166764500571 w_k1_PG_s110
                                 + 0.0541286735969094 w_k1_PG_s111
                                 + 0.0160572611043332 w_k1_PG_s112
                                 + 0.0501912222782842 w_k1_PG_s113
                                 + 0.0158747852106647 w_k1_PG_s114
                                 + 0.0315976030436153 w_k1_PG_s115
                                 + 0.0285902039398878 w_k1_PG_s116
                                 + 0.0160161995357584 w_k1_PG_s117 >= 1.17
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
