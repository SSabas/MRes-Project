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
 w_k1_ADM_s0_return_constraint:  1.2519 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.95726 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.1572 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.797 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.98401 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.24507 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.940367 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.41639 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.721396 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.12798 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.23353 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.744935 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.905747 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.764859 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.836529 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.29422 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.08404 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.940933 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.07371 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.967252 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.35741 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.829827 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.897323 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.01856 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.874841 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.22134 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.698836 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.29345 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.920895 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.801286 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.13812 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.21689 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.00124 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.11642 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.29572 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.0218 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.898812 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.879155 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.27944 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.909402 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.13917 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.920263 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.04224 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.32567 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.05074 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.65243 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.913209 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.32017 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.33935 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.642059 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.782317 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.06586 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.659376 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.42166 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.912642 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.769275 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.98786 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.01188 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.48377 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.798633 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.00893 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.929544 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.4363 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.53539 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.513248 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.19973 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.754655 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.71496 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.11047 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.40548 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.02871 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.25287 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.43361 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.0378 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.667057 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.872476 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.44295 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.909305 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.13505 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.885529 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.951794 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.34869 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.995323 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.72218 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.675181 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.1498 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.41083 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.634724 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.904733 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.9248 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.618349 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.28201 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.927971 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.685184 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.0348 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.2284 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.43749 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.676174 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.13325 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.0882 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.68475 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.78513 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.468718 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.77306 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.733632 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.543496 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.960023 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.64393 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.00776 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.1189 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.63184 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.25152 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.615065 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.942957 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.26393 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.937966 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.10309 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.847657 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.956058 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.29211 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.0137 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.2857 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.805876 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.32095 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.2866 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.743541 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.965381 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.923533 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.81255 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.13243 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.984246 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.810103 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.10819 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.07649 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.39129 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.833501 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.977037 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.08612 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.24506 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.42822 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.627269 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.44432 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.838876 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.729761 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.08061 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.28768 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.906153 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.06606 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.43206 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.17895 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.761461 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.956836 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.21166 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.986894 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.951828 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.22176 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.07998 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.865624 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.974874 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.82918 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.999243 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.771755 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.1184 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.632677 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.65043 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.51174 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.522777 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.54416 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.84459 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.976899 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.687283 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.01873 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.09709 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.739357 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.31149 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.710395 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 2.11266 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.33417 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.528479 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.939393 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.878016 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.630121 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.7815 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.87226 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.37156 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.1074 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.23276 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.968326 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.681487 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.808838 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.13787 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.923374 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.08532 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.968019 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.10066 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.09454 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.999668 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.45701 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.913167 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.11024 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.22298 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.778772 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.947925 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.05257 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.818123 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.17408 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.989326 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.684988 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.942508 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.975183 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.08211 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.718873 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.01926 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.966846 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.303 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.22089 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.611563 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.05387 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.764603 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.752122 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.03052 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.34025 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.01276 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.10787 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.55052 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.31673 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.896284 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.998802 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.26704 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.909779 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.98139 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.0139 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.04038 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.07971 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.04406 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.84952 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.945436 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.94359 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.19497 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.698584 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.839688 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.12841 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.631827 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.2635 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.981106 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.910171 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.783856 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.14212 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.21977 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.689022 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.24322 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.93796 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.63767 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.42549 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.578841 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.16484 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.865045 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.634376 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.938187 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.7058 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.22356 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.05303 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.3557 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.18109 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.776041 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.804048 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.33679 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.989043 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.889136 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.08374 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.964365 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.02578 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.03221 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 2.21927 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.03658 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.903988 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.62686 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.486445 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.656214 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.63878 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.472623 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.69282 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.888855 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.08176 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.7749 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.10272 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.51721 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.793392 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.55826 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.756836 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 2.60509 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.28204 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.611498 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.03077 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.94548 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.624922 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.788061 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 2.05533 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.47984 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.36522 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.3665 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.880138 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.569864 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.693079 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.21335 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.945617 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.14449 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.91488 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.98537 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.16117 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.935801 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.25389 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.766209 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.13627 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.19276 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.704171 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.948843 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.929017 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.786787 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.16094 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.08117 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.772778 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.998837 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.06972 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.26221 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.844437 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.06637 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.06256 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.20364 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.30716 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.658018 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.27377 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.83591 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.713876 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.992338 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.27069 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.09235 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.04667 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.3274 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.31444 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.77116 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.04762 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.18098 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.957664 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.01723 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.978085 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.990638 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.12574 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.00756 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.31376 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.891504 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.04812 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.1793 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.774478 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.894835 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.02688 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.839469 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.07957 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.00793 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.848866 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.954022 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.02878 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.16075 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.795259 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.0508 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.01221 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.26864 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.29145 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.732258 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.11789 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.905413 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.803506 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.04433 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.26516 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.952382 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.03111 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.13434 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.07427 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.776395 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.889102 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.310001600048395 z_k1_s0000
                                 - 0.51399930313461 z_k1_s0001
                                 - 0.298001048807272 z_k1_s0002
                                 - 0.191000120144991 z_k1_s0003
                                 - 0.409999387383576 z_k1_s0004
                                 - 0.486999521343693 z_k1_s0005
                                 - 0.153000027516593 z_k1_s0006
                                 - 0.554000168270614 z_k1_s0007
                                 - 0.0900000800865291 z_k1_s0010
                                 - 0.0790000755648446 z_k1_s0011
                                 - 0.122000545695272 z_k1_s0012
                                 - 0.193999993564214 z_k1_s0013
                                 - 0.165999615781578 z_k1_s0014
                                 - 0.171000318064481 z_k1_s0015
                                 - 0.110000045216845 z_k1_s0016
                                 - 0.252999985498689 z_k1_s0017
                                 - 0.181000420037233 z_k1_s0020
                                 - 0.084000035021228 z_k1_s0021
                                 - 0.188000409480675 z_k1_s0022
                                 - 0.279999577737701 z_k1_s0023
                                 - 0.374000590977627 z_k1_s0024
                                 - 0.162999985182901 z_k1_s0025
                                 - 0.0880000058889212 z_k1_s0026
                                 - 0.258999609407373 z_k1_s0027
                                 - 0.0670000330603529 z_k1_s0030
                                 - 0.0769999916618911 z_k1_s0031
                                 - 0.0820000129126665 z_k1_s0032
                                 - 0.164000193625358 z_k1_s0033
                                 - 0.144000108622257 z_k1_s0034
                                 - 0.117000245568107 z_k1_s0035
                                 - 0.0970001605650054 z_k1_s0036
                                 - 0.0909996316639993 z_k1_s0037
                                 - 0.205999680953435 z_k1_s0040
                                 - 0.301000070848938 z_k1_s0041
                                 - 0.546999638706 z_k1_s0042
                                 - 0.24699969672953 z_k1_s0043
                                 - 0.281000127397125 z_k1_s0044
                                 - 0.195999709227528 z_k1_s0045
                                 - 0.195999709227528 z_k1_s0046
                                 - 0.659998133909846 z_k1_s0047
                                 - 0.245000690791417 z_k1_s0050
                                 - 0.411000133282042 z_k1_s0051
                                 - 0.225000564598053 z_k1_s0052
                                 - 0.302000142745075 z_k1_s0053
                                 - 0.392001631467674 z_k1_s0054
                                 - 0.696001339560889 z_k1_s0055
                                 - 0.151000334473241 z_k1_s0056
                                 - 0.209000621503784 z_k1_s0057
                                 - 0.0800000227447988 z_k1_s0060
                                 - 0.172000277921333 z_k1_s0061
                                 - 0.141000336772728 z_k1_s0062
                                 - 0.133000251218243 z_k1_s0063
                                 - 0.128999687940964 z_k1_s0064
                                 - 0.107000233416182 z_k1_s0065
                                 - 0.102999982438925 z_k1_s0066
                                 - 0.17599980019854 z_k1_s0067
                                 - 0.15499996076517 z_k1_s0070
                                 - 0.0709999725343683 z_k1_s0071
                                 - 0.105000353163499 z_k1_s0072
                                 - 0.201000031398704 z_k1_s0073
                                 - 0.0739999686104684 z_k1_s0074
                                 - 0.0809999921547016 z_k1_s0075
                                 - 0.222000298231402 z_k1_s0076
                                 - 0.0720000039264014 z_k1_s0077
                                 - 0.0940000473102908 z_k1_s0100
                                 - 0.0830000404782349 z_k1_s0101
                                 - 0.153000360773268 z_k1_s0102
                                 - 0.147000489919482 z_k1_s0103
                                 - 0.181999736566564 z_k1_s0104
                                 - 0.135999508686963 z_k1_s0105
                                 - 0.0890000331320783 z_k1_s0106
                                 - 0.334000118864201 z_k1_s0107
                                 - 0.613999433168328 z_k1_s0110
                                 - 0.306000872452121 z_k1_s0111
                                 - 0.140000112307468 z_k1_s0112
                                 - 0.19800009397484 z_k1_s0113
                                 - 0.256999135974283 z_k1_s0114
                                 - 0.146000095500643 z_k1_s0115
                                 - 0.306000872452121 z_k1_s0116
                                 - 0.195000102378252 z_k1_s0117
                                 - 0.0729999042979606 z_k1_s0120
                                 - 0.110000130804903 z_k1_s0121
                                 - 0.0809998964292416 z_k1_s0122
                                 - 0.183000226302695 z_k1_s0123
                                 - 0.207999604213476 z_k1_s0124
                                 - 0.098999448525004 z_k1_s0125
                                 - 0.0739999750143073 z_k1_s0126
                                 - 0.127999682900666 z_k1_s0127
                                 - 0.106000181548418 z_k1_s0130
                                 - 0.0899999843335919 z_k1_s0131
                                 - 0.0509999911223688 z_k1_s0132
                                 - 0.044999992166796 z_k1_s0133
                                 - 0.100000182592845 z_k1_s0134
                                 - 0.0550000104260805 z_k1_s0135
                                 - 0.0550000104260805 z_k1_s0136
                                 - 0.0979997829410572 z_k1_s0137
                                 - 0.0990002442612652 z_k1_s0140
                                 - 0.0930000915787026 z_k1_s0141
                                 - 0.0550000345895466 z_k1_s0142
                                 - 0.130000058120736 z_k1_s0143
                                 - 0.057000020483705 z_k1_s0144
                                 - 0.11900010320285 z_k1_s0145
                                 - 0.0679997804015039 z_k1_s0146
                                 - 0.0290000229653994 z_k1_s0147
                                 - 0.060999815318589 z_k1_s0150
                                 - 0.0409999144435335 z_k1_s0151
                                 - 0.14899968686797 z_k1_s0152
                                 - 0.155999842223706 z_k1_s0153
                                 - 0.202999075331584 z_k1_s0154
                                 - 0.0419998461374641 z_k1_s0155
                                 - 0.14899968686797 z_k1_s0156
                                 - 0.103999593149984 z_k1_s0157
                                 - 0.104000092766334 z_k1_s0160
                                 - 0.194999433687569 z_k1_s0161
                                 - 0.0799999271050073 z_k1_s0162
                                 - 0.0799999271050073 z_k1_s0163
                                 - 0.104000092766334 z_k1_s0164
                                 - 0.0749998884797347 z_k1_s0165
                                 - 0.217999473183952 z_k1_s0166
                                 - 0.1309994525236 z_k1_s0167
                                 - 0.0660001446874378 z_k1_s0170
                                 - 0.0249999733967522 z_k1_s0171
                                 - 0.069000069855044 z_k1_s0172
                                 - 0.0389999823789347 z_k1_s0173
                                 - 0.09100011808419 z_k1_s0174
                                 - 0.117000066536811 z_k1_s0175
                                 - 0.0969999684194025 z_k1_s0176
                                 - 0.0930000681959275 z_k1_s0177
                                 - 2.047996920556 z_k1_s1000
                                 - 0.709000565317958 z_k1_s1001
                                 - 1.48200053687779 z_k1_s1002
                                 - 0.715000438578062 z_k1_s1003
                                 - 1.9790029268694 z_k1_s1004
                                 - 0.912001205156445 z_k1_s1005
                                 - 2.69100456601856 z_k1_s1006
                                 - 0.836000915193217 z_k1_s1007
                                 - 1.11699731867681 z_k1_s1010
                                 - 1.29499730126287 z_k1_s1011
                                 - 2.01098893164306 z_k1_s1012
                                 - 4.33799354403784 z_k1_s1013
                                 - 1.21099753956785 z_k1_s1014
                                 - 1.73199411124695 z_k1_s1015
                                 - 1.89499706178141 z_k1_s1016
                                 - 1.02799805873227 z_k1_s1017
                                 - 0.797997952234449 z_k1_s1020
                                 - 0.263999542929931 z_k1_s1021
                                 - 1.14699728498583 z_k1_s1022
                                 - 0.330999778019305 z_k1_s1023
                                 - 0.554998814057509 z_k1_s1024
                                 - 0.299999444593146 z_k1_s1025
                                 - 0.257999823385746 z_k1_s1026
                                 - 0.307999466751586 z_k1_s1027
                                 - 0.533998357851792 z_k1_s1030
                                 - 0.468998881467113 z_k1_s1031
                                 - 0.791998124416923 z_k1_s1032
                                 - 0.99400016305203 z_k1_s1033
                                 - 1.61599294814196 z_k1_s1034
                                 - 0.781999642569366 z_k1_s1035
                                 - 0.702998398108291 z_k1_s1036
                                 - 1.89699389514425 z_k1_s1037
                                 - 0.22299982862362 z_k1_s1040
                                 - 0.369999923185386 z_k1_s1041
                                 - 0.284999983231988 z_k1_s1042
                                 - 1.10199967369701 z_k1_s1043
                                 - 0.963000067194404 z_k1_s1044
                                 - 0.351999903602312 z_k1_s1045
                                 - 0.359999999461459 z_k1_s1046
                                 - 0.266999963648914 z_k1_s1047
                                 - 0.244999728294389 z_k1_s1050
                                 - 0.372998590715279 z_k1_s1051
                                 - 0.387000915152692 z_k1_s1052
                                 - 0.424998386348516 z_k1_s1053
                                 - 0.399998654563141 z_k1_s1054
                                 - 0.311000080766751 z_k1_s1055
                                 - 0.511999574079565 z_k1_s1056
                                 - 0.292999743601795 z_k1_s1057
                                 - 2.79900303290043 z_k1_s1060
                                 - 1.41299673185525 z_k1_s1061
                                 - 1.30899858032842 z_k1_s1062
                                 - 1.07199827464237 z_k1_s1063
                                 - 1.84900344031745 z_k1_s1064
                                 - 2.2340005839461 z_k1_s1065
                                 - 1.80899458358671 z_k1_s1066
                                 - 1.33499811821013 z_k1_s1067
                                 - 0.310999669630674 z_k1_s1070
                                 - 0.228000006221057 z_k1_s1071
                                 - 0.371998190014361 z_k1_s1072
                                 - 0.286999727065266 z_k1_s1073
                                 - 0.810001280705657 z_k1_s1074
                                 - 0.945999115744038 z_k1_s1075
                                 - 0.416999691886067 z_k1_s1076
                                 - 0.307999814772418 z_k1_s1077
                                 - 0.161001050827424 z_k1_s1100
                                 - 0.346001818525655 z_k1_s1101
                                 - 0.163001083967025 z_k1_s1102
                                 - 0.187001651843339 z_k1_s1103
                                 - 0.0970006711645947 z_k1_s1104
                                 - 0.443002830092452 z_k1_s1105
                                 - 0.137000823353312 z_k1_s1106
                                 - 0.168001081715477 z_k1_s1107
                                 - 0.234000142603673 z_k1_s1110
                                 - 0.186000280582811 z_k1_s1111
                                 - 0.0690002092809744 z_k1_s1112
                                 - 0.103000238362725 z_k1_s1113
                                 - 0.106000256914094 z_k1_s1114
                                 - 0.144000673065213 z_k1_s1115
                                 - 0.145000715482424 z_k1_s1116
                                 - 0.100000219811355 z_k1_s1117
                                 - 0.184000567172593 z_k1_s1120
                                 - 0.215000937555636 z_k1_s1121
                                 - 0.181000402190032 z_k1_s1122
                                 - 0.125000412790147 z_k1_s1123
                                 - 0.28200160212509 z_k1_s1124
                                 - 0.242000104692746 z_k1_s1125
                                 - 0.375000606268823 z_k1_s1126
                                 - 0.50300118404153 z_k1_s1127
                                 - 0.163999700004425 z_k1_s1130
                                 - 0.132000188199326 z_k1_s1131
                                 - 0.2170008181845 z_k1_s1132
                                 - 0.0520000874846027 z_k1_s1133
                                 - 0.151999883077513 z_k1_s1134
                                 - 0.21400058870236 z_k1_s1135
                                 - 0.117000141790274 z_k1_s1136
                                 - 0.0530001272785943 z_k1_s1137
                                 - 0.264001901857274 z_k1_s1140
                                 - 0.369001379658739 z_k1_s1141
                                 - 0.164000613181662 z_k1_s1142
                                 - 0.201001425573057 z_k1_s1143
                                 - 0.19600086763719 z_k1_s1144
                                 - 0.290001565750096 z_k1_s1145
                                 - 0.284000580385721 z_k1_s1146
                                 - 0.206001588707254 z_k1_s1147
                                 - 0.212000135483668 z_k1_s1150
                                 - 0.240000275642151 z_k1_s1151
                                 - 0.155999855166702 z_k1_s1152
                                 - 0.170999791394162 z_k1_s1153
                                 - 0.178000798435871 z_k1_s1154
                                 - 0.121000263169851 z_k1_s1155
                                 - 0.0680000996986558 z_k1_s1156
                                 - 0.150000658277388 z_k1_s1157
                                 - 0.365001133876626 z_k1_s1160
                                 - 0.690002323087833 z_k1_s1161
                                 - 0.353000986596532 z_k1_s1162
                                 - 0.606001292127177 z_k1_s1163
                                 - 0.236000574868544 z_k1_s1164
                                 - 0.717001630215118 z_k1_s1165
                                 - 0.266000943068778 z_k1_s1166
                                 - 0.86400241014334 z_k1_s1167
                                 - 0.350999433193655 z_k1_s1170
                                 - 0.559000735604902 z_k1_s1171
                                 - 0.253000035492345 z_k1_s1172
                                 - 0.202999997624289 z_k1_s1173
                                 - 0.274001040157044 z_k1_s1174
                                 - 0.290999908152049 z_k1_s1175
                                 - 0.168999919834005 z_k1_s1176
                                 - 0.502001316915392 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.12218 w_k1_ADM_s000
                                 + 1.19566 w_k1_AXP_s000
                                 + 1.21802 w_k1_DIS_s000
                                 + 1.18153 w_k1_FDX_s000
                                 + 1.11635 w_k1_IBM_s000
                                 + 1.13401 w_k1_INTC_s000
                                 + 1.03422 w_k1_KO_s000 + 1.37104 w_k1_MAT_s000
                                 + 1.1271 w_k1_MSFT_s000 + 1.12024 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.933877 w_k1_ADM_s000
                                 + 0.968824 w_k1_AXP_s000
                                 + 0.956723 w_k1_DIS_s000 + 1.032 w_k1_FDX_s000
                                 + 0.78514 w_k1_IBM_s000
                                 + 0.993991 w_k1_INTC_s000
                                 + 0.850861 w_k1_KO_s000
                                 + 0.82311 w_k1_MAT_s000
                                 + 0.940651 w_k1_MSFT_s000
                                 + 0.965869 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.847586 w_k1_ADM_s000
                                 + 0.996073 w_k1_AXP_s000
                                 + 1.09482 w_k1_DIS_s000
                                 + 0.950901 w_k1_FDX_s000
                                 + 1.49749 w_k1_IBM_s000
                                 + 1.00714 w_k1_INTC_s000
                                 + 1.06462 w_k1_KO_s000 + 1.62581 w_k1_MAT_s000
                                 + 0.892128 w_k1_MSFT_s000
                                 + 1.08964 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.590914 w_k1_ADM_s000
                                 + 0.636249 w_k1_AXP_s000
                                 + 0.68437 w_k1_DIS_s000
                                 + 0.791131 w_k1_FDX_s000
                                 + 0.757689 w_k1_IBM_s000
                                 + 0.948788 w_k1_INTC_s000
                                 + 0.747581 w_k1_KO_s000
                                 + 0.718346 w_k1_MAT_s000
                                 + 0.878829 w_k1_MSFT_s000
                                 + 0.883994 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.07363 w_k1_ADM_s000
                                 + 0.904835 w_k1_AXP_s000
                                 + 0.960939 w_k1_DIS_s000
                                 + 0.951127 w_k1_FDX_s000
                                 + 1.0915 w_k1_IBM_s000 + 1.2199 w_k1_INTC_s000
                                 + 0.987444 w_k1_KO_s000
                                 + 0.905062 w_k1_MAT_s000
                                 + 1.15293 w_k1_MSFT_s000 + 1.0892 w_k1_PG_s000
                                 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.01654 w_k1_ADM_s000
                                 + 1.30829 w_k1_AXP_s000
                                 + 1.53208 w_k1_DIS_s000
                                 + 1.31531 w_k1_FDX_s000
                                 + 1.09737 w_k1_IBM_s000
                                 + 1.1713 w_k1_INTC_s000 + 1.17897 w_k1_KO_s000
                                 + 1.52138 w_k1_MAT_s000
                                 + 1.19426 w_k1_MSFT_s000 + 1.2726 w_k1_PG_s000
                                 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.786166 w_k1_ADM_s000
                                 + 0.659027 w_k1_AXP_s000
                                 + 0.54441 w_k1_DIS_s000
                                 + 0.798459 w_k1_FDX_s000
                                 + 0.584272 w_k1_IBM_s000
                                 + 0.782887 w_k1_INTC_s000
                                 + 0.657944 w_k1_KO_s000
                                 + 0.747236 w_k1_MAT_s000
                                 + 0.752962 w_k1_MSFT_s000
                                 + 0.875263 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.775374 w_k1_ADM_s000
                                 + 0.794047 w_k1_AXP_s000
                                 + 0.839086 w_k1_DIS_s000
                                 + 0.814545 w_k1_FDX_s000
                                 + 0.943104 w_k1_IBM_s000
                                 + 1.04204 w_k1_INTC_s000
                                 + 0.851969 w_k1_KO_s000
                                 + 0.847986 w_k1_MAT_s000
                                 + 0.957374 w_k1_MSFT_s000
                                 + 0.87329 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.853874 w_k1_ADM_s001
                                 + 1.13292 w_k1_AXP_s001
                                 + 1.40174 w_k1_DIS_s001
                                 + 1.24252 w_k1_FDX_s001 + 1.1809 w_k1_IBM_s001
                                 + 0.878488 w_k1_INTC_s001
                                 + 1.03602 w_k1_KO_s001
                                 + 0.942187 w_k1_MAT_s001
                                 + 1.16009 w_k1_MSFT_s001
                                 + 1.09002 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.886268 w_k1_ADM_s001
                                 + 1.36244 w_k1_AXP_s001
                                 + 1.54988 w_k1_DIS_s001
                                 + 1.21864 w_k1_FDX_s001
                                 + 1.68729 w_k1_IBM_s001
                                 + 1.07976 w_k1_INTC_s001
                                 + 1.44992 w_k1_KO_s001 + 1.41783 w_k1_MAT_s001
                                 + 1.28172 w_k1_MSFT_s001
                                 + 1.14442 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.862589 w_k1_ADM_s001
                                 + 0.89395 w_k1_AXP_s001
                                 + 0.826911 w_k1_DIS_s001
                                 + 0.781188 w_k1_FDX_s001
                                 + 1.49854 w_k1_IBM_s001
                                 + 0.782619 w_k1_INTC_s001
                                 + 0.939522 w_k1_KO_s001
                                 + 1.08195 w_k1_MAT_s001
                                 + 0.792322 w_k1_MSFT_s001
                                 + 1.07361 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.848562 w_k1_ADM_s001
                                 + 0.964134 w_k1_AXP_s001
                                 + 1.04017 w_k1_DIS_s001
                                 + 0.932432 w_k1_FDX_s001
                                 + 1.09857 w_k1_IBM_s001
                                 + 0.896385 w_k1_INTC_s001
                                 + 0.950484 w_k1_KO_s001
                                 + 0.879091 w_k1_MAT_s001
                                 + 0.972061 w_k1_MSFT_s001
                                 + 0.914116 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.06477 w_k1_ADM_s001
                                 + 1.35117 w_k1_AXP_s001
                                 + 1.63371 w_k1_DIS_s001
                                 + 1.41071 w_k1_FDX_s001
                                 + 0.977637 w_k1_IBM_s001
                                 + 1.11152 w_k1_INTC_s001
                                 + 1.00944 w_k1_KO_s001 + 1.10011 w_k1_MAT_s001
                                 + 1.1894 w_k1_MSFT_s001 + 1.25147 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.755877 w_k1_ADM_s001
                                 + 0.664179 w_k1_AXP_s001
                                 + 0.633747 w_k1_DIS_s001
                                 + 0.777182 w_k1_FDX_s001
                                 + 0.728466 w_k1_IBM_s001
                                 + 0.733772 w_k1_INTC_s001
                                 + 0.644735 w_k1_KO_s001
                                 + 0.532098 w_k1_MAT_s001
                                 + 0.779465 w_k1_MSFT_s001
                                 + 0.831596 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.659695 w_k1_ADM_s001
                                 + 0.661328 w_k1_AXP_s001
                                 + 0.749873 w_k1_DIS_s001
                                 + 0.825919 w_k1_FDX_s001
                                 + 0.960866 w_k1_IBM_s001
                                 + 0.809765 w_k1_INTC_s001
                                 + 0.752012 w_k1_KO_s001
                                 + 0.919734 w_k1_MAT_s001
                                 + 0.880382 w_k1_MSFT_s001
                                 + 0.857284 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.791678 w_k1_ADM_s001
                                 + 0.851451 w_k1_AXP_s001
                                 + 0.888523 w_k1_DIS_s001
                                 + 1.07463 w_k1_FDX_s001
                                 + 0.798985 w_k1_IBM_s001
                                 + 0.983868 w_k1_INTC_s001
                                 + 0.882649 w_k1_KO_s001
                                 + 0.839652 w_k1_MAT_s001
                                 + 1.06977 w_k1_MSFT_s001
                                 + 0.960661 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.20052 w_k1_ADM_s002
                                 + 1.1613 w_k1_AXP_s002 + 1.50122 w_k1_DIS_s002
                                 + 1.38511 w_k1_FDX_s002
                                 + 0.814578 w_k1_IBM_s002
                                 + 1.36221 w_k1_INTC_s002 + 1.0516 w_k1_KO_s002
                                 + 0.761161 w_k1_MAT_s002
                                 + 1.36152 w_k1_MSFT_s002
                                 + 1.08116 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.07873 w_k1_ADM_s002
                                 + 0.800215 w_k1_AXP_s002
                                 + 1.31866 w_k1_DIS_s002
                                 + 1.04647 w_k1_FDX_s002
                                 + 0.798958 w_k1_IBM_s002
                                 + 0.882404 w_k1_INTC_s002
                                 + 0.850069 w_k1_KO_s002
                                 + 0.580526 w_k1_MAT_s002
                                 + 1.12357 w_k1_MSFT_s002
                                 + 0.924902 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.912438 w_k1_ADM_s002
                                 + 0.61628 w_k1_AXP_s002
                                 + 0.754727 w_k1_DIS_s002
                                 + 0.754318 w_k1_FDX_s002
                                 + 0.783563 w_k1_IBM_s002
                                 + 0.822985 w_k1_INTC_s002
                                 + 0.787817 w_k1_KO_s002
                                 + 0.58953 w_k1_MAT_s002
                                 + 0.823314 w_k1_MSFT_s002
                                 + 0.878783 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.913592 w_k1_ADM_s002
                                 + 0.903562 w_k1_AXP_s002
                                 + 0.98825 w_k1_DIS_s002
                                 + 0.84717 w_k1_FDX_s002
                                 + 1.32647 w_k1_IBM_s002
                                 + 0.888434 w_k1_INTC_s002
                                 + 1.10583 w_k1_KO_s002 + 1.45661 w_k1_MAT_s002
                                 + 0.960035 w_k1_MSFT_s002
                                 + 0.9889 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.17901 w_k1_ADM_s002
                                 + 1.16568 w_k1_AXP_s002
                                 + 1.55111 w_k1_DIS_s002
                                 + 1.17202 w_k1_FDX_s002
                                 + 1.24272 w_k1_IBM_s002
                                 + 1.08491 w_k1_INTC_s002
                                 + 1.28646 w_k1_KO_s002 + 1.03549 w_k1_MAT_s002
                                 + 1.20088 w_k1_MSFT_s002
                                 + 1.02836 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.830291 w_k1_ADM_s002
                                 + 0.873967 w_k1_AXP_s002
                                 + 0.907882 w_k1_DIS_s002
                                 + 0.9045 w_k1_FDX_s002
                                 + 0.933946 w_k1_IBM_s002
                                 + 0.966687 w_k1_INTC_s002
                                 + 0.861568 w_k1_KO_s002
                                 + 0.694185 w_k1_MAT_s002
                                 + 0.820696 w_k1_MSFT_s002
                                 + 0.95928 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.04715 w_k1_ADM_s002
                                 + 0.73711 w_k1_AXP_s002
                                 + 0.853414 w_k1_DIS_s002
                                 + 0.913086 w_k1_FDX_s002
                                 + 0.578955 w_k1_IBM_s002
                                 + 0.744293 w_k1_INTC_s002
                                 + 0.603339 w_k1_KO_s002
                                 + 0.516996 w_k1_MAT_s002
                                 + 0.844529 w_k1_MSFT_s002
                                 + 0.832464 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.21257 w_k1_ADM_s002
                                 + 0.976267 w_k1_AXP_s002
                                 + 1.01095 w_k1_DIS_s002
                                 + 1.05758 w_k1_FDX_s002
                                 + 0.773039 w_k1_IBM_s002
                                 + 0.91012 w_k1_INTC_s002
                                 + 0.836324 w_k1_KO_s002
                                 + 1.13864 w_k1_MAT_s002
                                 + 0.942546 w_k1_MSFT_s002
                                 + 0.881383 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.12423 w_k1_ADM_s003
                                 + 0.875843 w_k1_AXP_s003
                                 + 1.10434 w_k1_DIS_s003
                                 + 0.994049 w_k1_FDX_s003
                                 + 1.24594 w_k1_IBM_s003
                                 + 1.07436 w_k1_INTC_s003 + 1.1169 w_k1_KO_s003
                                 + 0.971212 w_k1_MAT_s003
                                 + 1.0056 w_k1_MSFT_s003 + 1.14295 w_k1_PG_s003
                                 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.07819 w_k1_ADM_s003
                                 + 1.03365 w_k1_AXP_s003
                                 + 1.35471 w_k1_DIS_s003 + 1.1796 w_k1_FDX_s003
                                 + 0.923407 w_k1_IBM_s003
                                 + 1.22285 w_k1_INTC_s003
                                 + 1.13783 w_k1_KO_s003 + 1.18967 w_k1_MAT_s003
                                 + 1.10696 w_k1_MSFT_s003
                                 + 1.13529 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.18531 w_k1_ADM_s003
                                 + 1.38004 w_k1_AXP_s003
                                 + 1.26247 w_k1_DIS_s003
                                 + 1.21199 w_k1_FDX_s003
                                 + 1.25308 w_k1_IBM_s003
                                 + 1.17909 w_k1_INTC_s003 + 1.1647 w_k1_KO_s003
                                 + 1.34983 w_k1_MAT_s003
                                 + 1.01059 w_k1_MSFT_s003
                                 + 1.20235 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.857136 w_k1_ADM_s003
                                 + 0.693419 w_k1_AXP_s003
                                 + 0.690187 w_k1_DIS_s003
                                 + 0.825862 w_k1_FDX_s003
                                 + 0.697799 w_k1_IBM_s003
                                 + 0.849013 w_k1_INTC_s003
                                 + 0.73077 w_k1_KO_s003
                                 + 0.754142 w_k1_MAT_s003
                                 + 0.734924 w_k1_MSFT_s003
                                 + 0.813564 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.06926 w_k1_ADM_s003
                                 + 0.974237 w_k1_AXP_s003
                                 + 0.894607 w_k1_DIS_s003
                                 + 0.942558 w_k1_FDX_s003
                                 + 0.902574 w_k1_IBM_s003
                                 + 0.971996 w_k1_INTC_s003
                                 + 0.896187 w_k1_KO_s003
                                 + 0.863861 w_k1_MAT_s003
                                 + 0.810752 w_k1_MSFT_s003
                                 + 0.881731 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.03906 w_k1_ADM_s003
                                 + 0.886598 w_k1_AXP_s003
                                 + 1.08288 w_k1_DIS_s003
                                 + 1.08034 w_k1_FDX_s003
                                 + 0.690876 w_k1_IBM_s003
                                 + 0.975556 w_k1_INTC_s003
                                 + 1.02558 w_k1_KO_s003
                                 + 0.910611 w_k1_MAT_s003
                                 + 1.00551 w_k1_MSFT_s003 + 1.0591 w_k1_PG_s003
                                 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.0517 w_k1_ADM_s003
                                 + 0.773973 w_k1_AXP_s003
                                 + 0.716364 w_k1_DIS_s003
                                 + 0.807605 w_k1_FDX_s003
                                 + 1.00032 w_k1_IBM_s003
                                 + 1.02968 w_k1_INTC_s003
                                 + 0.98594 w_k1_KO_s003 + 1.29282 w_k1_MAT_s003
                                 + 0.899933 w_k1_MSFT_s003
                                 + 0.941295 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.57419 w_k1_ADM_s003
                                 + 1.4346 w_k1_AXP_s003 + 1.86889 w_k1_DIS_s003
                                 + 1.44537 w_k1_FDX_s003
                                 + 1.06511 w_k1_IBM_s003
                                 + 1.50845 w_k1_INTC_s003
                                 + 1.44184 w_k1_KO_s003 + 1.06764 w_k1_MAT_s003
                                 + 1.39597 w_k1_MSFT_s003
                                 + 1.14677 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.653551 w_k1_ADM_s004
                                 + 0.826745 w_k1_AXP_s004
                                 + 0.78878 w_k1_DIS_s004
                                 + 0.727713 w_k1_FDX_s004
                                 + 1.36465 w_k1_IBM_s004
                                 + 0.808461 w_k1_INTC_s004
                                 + 0.999955 w_k1_KO_s004
                                 + 1.08107 w_k1_MAT_s004
                                 + 0.781767 w_k1_MSFT_s004
                                 + 0.803526 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.751651 w_k1_ADM_s004
                                 + 0.669516 w_k1_AXP_s004
                                 + 0.568629 w_k1_DIS_s004
                                 + 0.709621 w_k1_FDX_s004
                                 + 0.795426 w_k1_IBM_s004
                                 + 0.767402 w_k1_INTC_s004
                                 + 0.780634 w_k1_KO_s004
                                 + 0.628143 w_k1_MAT_s004
                                 + 0.65881 w_k1_MSFT_s004
                                 + 0.826376 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.795264 w_k1_ADM_s004
                                 + 0.988885 w_k1_AXP_s004
                                 + 0.854755 w_k1_DIS_s004
                                 + 0.887332 w_k1_FDX_s004
                                 + 1.09353 w_k1_IBM_s004
                                 + 0.769586 w_k1_INTC_s004
                                 + 0.940182 w_k1_KO_s004
                                 + 0.899889 w_k1_MAT_s004
                                 + 0.856918 w_k1_MSFT_s004
                                 + 0.866679 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.956766 w_k1_ADM_s004
                                 + 0.990425 w_k1_AXP_s004
                                 + 0.936902 w_k1_DIS_s004
                                 + 0.931959 w_k1_FDX_s004
                                 + 0.699718 w_k1_IBM_s004
                                 + 0.895734 w_k1_INTC_s004
                                 + 1.00831 w_k1_KO_s004
                                 + 0.658727 w_k1_MAT_s004
                                 + 0.833867 w_k1_MSFT_s004
                                 + 0.906976 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.640936 w_k1_ADM_s004
                                 + 0.826912 w_k1_AXP_s004
                                 + 1.08399 w_k1_DIS_s004
                                 + 0.997124 w_k1_FDX_s004
                                 + 0.885315 w_k1_IBM_s004
                                 + 1.13509 w_k1_INTC_s004
                                 + 1.00951 w_k1_KO_s004
                                 + 0.656864 w_k1_MAT_s004
                                 + 1.07222 w_k1_MSFT_s004
                                 + 0.830569 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.14553 w_k1_ADM_s004
                                 + 1.53566 w_k1_AXP_s004
                                 + 1.54184 w_k1_DIS_s004 + 1.2449 w_k1_FDX_s004
                                 + 1.56522 w_k1_IBM_s004
                                 + 1.39572 w_k1_INTC_s004
                                 + 1.45284 w_k1_KO_s004 + 1.39744 w_k1_MAT_s004
                                 + 1.12297 w_k1_MSFT_s004
                                 + 1.12205 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.40142 w_k1_ADM_s004
                                 + 1.67216 w_k1_AXP_s004
                                 + 2.07542 w_k1_DIS_s004
                                 + 1.60693 w_k1_FDX_s004
                                 + 1.25777 w_k1_IBM_s004
                                 + 1.27443 w_k1_INTC_s004
                                 + 1.29155 w_k1_KO_s004
                                 + 0.894528 w_k1_MAT_s004
                                 + 1.28965 w_k1_MSFT_s004
                                 + 1.08891 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.30276 w_k1_ADM_s004
                                 + 1.1125 w_k1_AXP_s004 + 1.25988 w_k1_DIS_s004
                                 + 1.14314 w_k1_FDX_s004
                                 + 1.02429 w_k1_IBM_s004
                                 + 1.11323 w_k1_INTC_s004
                                 + 1.25763 w_k1_KO_s004
                                 + 0.742747 w_k1_MAT_s004
                                 + 1.13501 w_k1_MSFT_s004
                                 + 1.15405 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.948325 w_k1_ADM_s005
                                 + 1.66658 w_k1_AXP_s005
                                 + 1.38505 w_k1_DIS_s005 + 1.2547 w_k1_FDX_s005
                                 + 1.3216 w_k1_IBM_s005
                                 + 1.20395 w_k1_INTC_s005
                                 + 1.21719 w_k1_KO_s005 + 2.11126 w_k1_MAT_s005
                                 + 1.21598 w_k1_MSFT_s005
                                 + 1.25982 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.767352 w_k1_ADM_s005
                                 + 0.906859 w_k1_AXP_s005
                                 + 0.779354 w_k1_DIS_s005
                                 + 0.838806 w_k1_FDX_s005
                                 + 1.07899 w_k1_IBM_s005
                                 + 0.882206 w_k1_INTC_s005
                                 + 0.888728 w_k1_KO_s005
                                 + 1.26128 w_k1_MAT_s005
                                 + 0.853758 w_k1_MSFT_s005
                                 + 0.856209 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.996105 w_k1_ADM_s005
                                 + 1.28052 w_k1_AXP_s005
                                 + 1.43222 w_k1_DIS_s005
                                 + 1.36265 w_k1_FDX_s005
                                 + 0.982199 w_k1_IBM_s005
                                 + 1.22627 w_k1_INTC_s005
                                 + 1.24932 w_k1_KO_s005 + 1.11153 w_k1_MAT_s005
                                 + 1.20745 w_k1_MSFT_s005
                                 + 1.10335 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.05238 w_k1_ADM_s005
                                 + 1.20754 w_k1_AXP_s005
                                 + 1.11641 w_k1_DIS_s005
                                 + 1.13051 w_k1_FDX_s005
                                 + 0.982681 w_k1_IBM_s005
                                 + 1.21751 w_k1_INTC_s005
                                 + 1.04084 w_k1_KO_s005 + 1.20143 w_k1_MAT_s005
                                 + 1.22067 w_k1_MSFT_s005
                                 + 1.12259 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.711926 w_k1_ADM_s005
                                 + 0.571319 w_k1_AXP_s005
                                 + 0.498169 w_k1_DIS_s005
                                 + 0.725542 w_k1_FDX_s005
                                 + 0.758538 w_k1_IBM_s005
                                 + 0.691575 w_k1_INTC_s005
                                 + 0.689316 w_k1_KO_s005
                                 + 0.902687 w_k1_MAT_s005
                                 + 0.818206 w_k1_MSFT_s005
                                 + 0.832854 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.03108 w_k1_ADM_s005
                                 + 0.992654 w_k1_AXP_s005
                                 + 0.871063 w_k1_DIS_s005
                                 + 1.06626 w_k1_FDX_s005
                                 + 0.82029 w_k1_IBM_s005
                                 + 0.937424 w_k1_INTC_s005
                                 + 0.83659 w_k1_KO_s005
                                 + 0.870441 w_k1_MAT_s005
                                 + 1.0004 w_k1_MSFT_s005
                                 + 0.989104 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.28384 w_k1_ADM_s005
                                 + 1.30571 w_k1_AXP_s005
                                 + 1.51628 w_k1_DIS_s005
                                 + 1.33285 w_k1_FDX_s005
                                 + 1.55686 w_k1_IBM_s005
                                 + 1.44327 w_k1_INTC_s005
                                 + 1.56372 w_k1_KO_s005 + 1.54641 w_k1_MAT_s005
                                 + 1.38203 w_k1_MSFT_s005
                                 + 1.29096 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.747491 w_k1_ADM_s005
                                 + 0.965533 w_k1_AXP_s005
                                 + 0.97662 w_k1_DIS_s005
                                 + 0.923851 w_k1_FDX_s005
                                 + 1.50986 w_k1_IBM_s005
                                 + 0.895411 w_k1_INTC_s005
                                 + 1.08905 w_k1_KO_s005 + 2.14203 w_k1_MAT_s005
                                 + 0.908728 w_k1_MSFT_s005
                                 + 1.01286 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.11509 w_k1_ADM_s006
                                 + 0.98928 w_k1_AXP_s006
                                 + 1.80434 w_k1_DIS_s006
                                 + 1.32024 w_k1_FDX_s006
                                 + 1.06164 w_k1_IBM_s006
                                 + 1.58882 w_k1_INTC_s006 + 1.4768 w_k1_KO_s006
                                 + 0.614048 w_k1_MAT_s006
                                 + 1.42418 w_k1_MSFT_s006
                                 + 1.31527 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.889911 w_k1_ADM_s006
                                 + 0.941171 w_k1_AXP_s006
                                 + 0.776961 w_k1_DIS_s006
                                 + 0.876806 w_k1_FDX_s006
                                 + 0.75833 w_k1_IBM_s006
                                 + 0.870392 w_k1_INTC_s006
                                 + 0.850204 w_k1_KO_s006
                                 + 0.766262 w_k1_MAT_s006
                                 + 0.801062 w_k1_MSFT_s006
                                 + 0.895142 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.01538 w_k1_ADM_s006
                                 + 0.701859 w_k1_AXP_s006
                                 + 0.994957 w_k1_DIS_s006
                                 + 0.978209 w_k1_FDX_s006
                                 + 0.967711 w_k1_IBM_s006
                                 + 0.954389 w_k1_INTC_s006
                                 + 1.01506 w_k1_KO_s006
                                 + 0.676003 w_k1_MAT_s006
                                 + 0.98914 w_k1_MSFT_s006
                                 + 1.03572 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.05236 w_k1_ADM_s006
                                 + 0.976319 w_k1_AXP_s006
                                 + 1.1655 w_k1_DIS_s006 + 1.09443 w_k1_FDX_s006
                                 + 0.86456 w_k1_IBM_s006
                                 + 1.20745 w_k1_INTC_s006 + 1.2875 w_k1_KO_s006
                                 + 0.445735 w_k1_MAT_s006
                                 + 1.14406 w_k1_MSFT_s006
                                 + 1.06922 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.835926 w_k1_ADM_s006
                                 + 0.915071 w_k1_AXP_s006
                                 + 1.18706 w_k1_DIS_s006
                                 + 0.962341 w_k1_FDX_s006
                                 + 1.42252 w_k1_IBM_s006
                                 + 1.03958 w_k1_INTC_s006
                                 + 1.13948 w_k1_KO_s006
                                 + 0.943357 w_k1_MAT_s006
                                 + 0.994291 w_k1_MSFT_s006
                                 + 0.997782 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.927344 w_k1_ADM_s006
                                 + 1.06399 w_k1_AXP_s006
                                 + 0.828295 w_k1_DIS_s006
                                 + 1.07366 w_k1_FDX_s006
                                 + 0.739881 w_k1_IBM_s006
                                 + 1.16973 w_k1_INTC_s006
                                 + 0.85465 w_k1_KO_s006 + 1.02983 w_k1_MAT_s006
                                 + 1.12089 w_k1_MSFT_s006
                                 + 1.05479 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.7789 w_k1_ADM_s006
                                 + 0.561812 w_k1_AXP_s006
                                 + 0.716606 w_k1_DIS_s006
                                 + 0.73529 w_k1_FDX_s006
                                 + 0.934004 w_k1_IBM_s006
                                 + 0.889514 w_k1_INTC_s006
                                 + 0.811434 w_k1_KO_s006
                                 + 0.510459 w_k1_MAT_s006
                                 + 0.803225 w_k1_MSFT_s006
                                 + 0.878794 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.0467 w_k1_ADM_s006
                                 + 1.19845 w_k1_AXP_s006
                                 + 1.18151 w_k1_DIS_s006
                                 + 1.26719 w_k1_FDX_s006
                                 + 1.10381 w_k1_IBM_s006
                                 + 1.47388 w_k1_INTC_s006
                                 + 1.23668 w_k1_KO_s006
                                 + 0.890946 w_k1_MAT_s006
                                 + 1.18742 w_k1_MSFT_s006
                                 + 1.18648 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.01002 w_k1_ADM_s007
                                 + 0.896322 w_k1_AXP_s007
                                 + 1.17191 w_k1_DIS_s007
                                 + 0.968572 w_k1_FDX_s007
                                 + 0.885394 w_k1_IBM_s007
                                 + 0.852985 w_k1_INTC_s007
                                 + 1.05695 w_k1_KO_s007
                                 + 0.715605 w_k1_MAT_s007
                                 + 0.947264 w_k1_MSFT_s007
                                 + 1.01267 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.98777 w_k1_ADM_s007
                                 + 0.776419 w_k1_AXP_s007
                                 + 0.846785 w_k1_DIS_s007
                                 + 0.806257 w_k1_FDX_s007
                                 + 1.2407 w_k1_IBM_s007
                                 + 0.934958 w_k1_INTC_s007
                                 + 1.10392 w_k1_KO_s007
                                 + 0.993738 w_k1_MAT_s007
                                 + 0.811952 w_k1_MSFT_s007
                                 + 1.01901 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.58622 w_k1_ADM_s007
                                 + 1.55633 w_k1_AXP_s007
                                 + 1.74538 w_k1_DIS_s007
                                 + 1.39824 w_k1_FDX_s007 + 1.4146 w_k1_IBM_s007
                                 + 1.34621 w_k1_INTC_s007
                                 + 1.63423 w_k1_KO_s007 + 1.22523 w_k1_MAT_s007
                                 + 1.22751 w_k1_MSFT_s007
                                 + 1.32762 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.32303 w_k1_ADM_s007
                                 + 1.17093 w_k1_AXP_s007
                                 + 1.28629 w_k1_DIS_s007
                                 + 1.23732 w_k1_FDX_s007
                                 + 0.93213 w_k1_IBM_s007
                                 + 1.26976 w_k1_INTC_s007
                                 + 1.21669 w_k1_KO_s007 + 1.24102 w_k1_MAT_s007
                                 + 1.08707 w_k1_MSFT_s007 + 1.3407 w_k1_PG_s007
                                 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.45593 w_k1_ADM_s007
                                 + 1.19997 w_k1_AXP_s007
                                 + 0.988792 w_k1_DIS_s007
                                 + 0.946062 w_k1_FDX_s007
                                 + 1.33199 w_k1_IBM_s007
                                 + 0.913855 w_k1_INTC_s007
                                 + 1.31368 w_k1_KO_s007 + 1.13526 w_k1_MAT_s007
                                 + 0.994828 w_k1_MSFT_s007
                                 + 1.04161 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.04247 w_k1_ADM_s007
                                 + 1.30359 w_k1_AXP_s007
                                 + 0.837592 w_k1_DIS_s007
                                 + 1.0086 w_k1_FDX_s007
                                 + 0.898866 w_k1_IBM_s007
                                 + 0.95867 w_k1_INTC_s007
                                 + 0.956171 w_k1_KO_s007
                                 + 1.08828 w_k1_MAT_s007
                                 + 0.931647 w_k1_MSFT_s007
                                 + 0.973455 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.864424 w_k1_ADM_s007
                                 + 0.686018 w_k1_AXP_s007
                                 + 0.674704 w_k1_DIS_s007
                                 + 0.836188 w_k1_FDX_s007
                                 + 0.779646 w_k1_IBM_s007
                                 + 0.772429 w_k1_INTC_s007
                                 + 0.840363 w_k1_KO_s007
                                 + 0.595154 w_k1_MAT_s007
                                 + 0.882624 w_k1_MSFT_s007
                                 + 0.816068 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.15942 w_k1_ADM_s007
                                 + 1.058 w_k1_AXP_s007 + 1.30463 w_k1_DIS_s007
                                 + 0.980372 w_k1_FDX_s007
                                 + 1.42233 w_k1_IBM_s007
                                 + 0.933231 w_k1_INTC_s007
                                 + 1.40483 w_k1_KO_s007 + 1.10377 w_k1_MAT_s007
                                 + 1.00721 w_k1_MSFT_s007
                                 + 1.20959 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.993119 w_k1_ADM_s010
                                 + 0.926889 w_k1_AXP_s010
                                 + 1.00772 w_k1_DIS_s010
                                 + 0.834423 w_k1_FDX_s010
                                 + 1.54704 w_k1_IBM_s010
                                 + 0.94326 w_k1_INTC_s010
                                 + 1.18103 w_k1_KO_s010 + 1.57102 w_k1_MAT_s010
                                 + 0.970053 w_k1_MSFT_s010
                                 + 1.01896 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.659375 w_k1_ADM_s010
                                 + 0.800366 w_k1_AXP_s010
                                 + 0.519928 w_k1_DIS_s010
                                 + 0.652466 w_k1_FDX_s010
                                 + 0.927862 w_k1_IBM_s010
                                 + 0.716922 w_k1_INTC_s010
                                 + 0.746154 w_k1_KO_s010
                                 + 0.983054 w_k1_MAT_s010
                                 + 0.733445 w_k1_MSFT_s010
                                 + 0.774138 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.06818 w_k1_ADM_s010
                                 + 0.741649 w_k1_AXP_s010
                                 + 0.862842 w_k1_DIS_s010
                                 + 0.819327 w_k1_FDX_s010
                                 + 0.654927 w_k1_IBM_s010
                                 + 0.906814 w_k1_INTC_s010
                                 + 0.819884 w_k1_KO_s010
                                 + 0.524491 w_k1_MAT_s010
                                 + 0.934781 w_k1_MSFT_s010
                                 + 0.7748 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.841419 w_k1_ADM_s010
                                 + 0.757408 w_k1_AXP_s010
                                 + 0.762976 w_k1_DIS_s010
                                 + 0.768658 w_k1_FDX_s010
                                 + 0.93389 w_k1_IBM_s010
                                 + 0.744633 w_k1_INTC_s010
                                 + 0.843596 w_k1_KO_s010
                                 + 0.765328 w_k1_MAT_s010
                                 + 0.878506 w_k1_MSFT_s010
                                 + 0.769718 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.21427 w_k1_ADM_s010
                                 + 0.959718 w_k1_AXP_s010
                                 + 0.804685 w_k1_DIS_s010
                                 + 1.01635 w_k1_FDX_s010
                                 + 0.796266 w_k1_IBM_s010
                                 + 0.963921 w_k1_INTC_s010
                                 + 0.827346 w_k1_KO_s010
                                 + 0.75084 w_k1_MAT_s010
                                 + 0.91614 w_k1_MSFT_s010
                                 + 0.839296 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.17962 w_k1_ADM_s010
                                 + 0.983191 w_k1_AXP_s010
                                 + 1.18196 w_k1_DIS_s010
                                 + 1.01255 w_k1_FDX_s010
                                 + 1.29967 w_k1_IBM_s010
                                 + 1.16539 w_k1_INTC_s010
                                 + 1.34356 w_k1_KO_s010 + 1.68177 w_k1_MAT_s010
                                 + 1.04971 w_k1_MSFT_s010
                                 + 1.00477 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.39528 w_k1_ADM_s010
                                 + 1.57871 w_k1_AXP_s010
                                 + 1.80183 w_k1_DIS_s010
                                 + 1.41553 w_k1_FDX_s010
                                 + 1.36049 w_k1_IBM_s010
                                 + 1.1771 w_k1_INTC_s010 + 1.5134 w_k1_KO_s010
                                 + 1.93412 w_k1_MAT_s010
                                 + 1.38915 w_k1_MSFT_s010
                                 + 1.27104 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.19069 w_k1_ADM_s010
                                 + 1.25411 w_k1_AXP_s010
                                 + 1.39762 w_k1_DIS_s010 + 1.1932 w_k1_FDX_s010
                                 + 1.04784 w_k1_IBM_s010
                                 + 1.08849 w_k1_INTC_s010
                                 + 1.15274 w_k1_KO_s010 + 1.269 w_k1_MAT_s010
                                 + 1.28531 w_k1_MSFT_s010
                                 + 1.03801 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.02739 w_k1_ADM_s011
                                 + 0.828031 w_k1_AXP_s011
                                 + 1.03471 w_k1_DIS_s011
                                 + 1.02579 w_k1_FDX_s011
                                 + 0.952386 w_k1_IBM_s011
                                 + 0.988571 w_k1_INTC_s011
                                 + 1.05467 w_k1_KO_s011 + 0.88955 w_k1_MAT_s011
                                 + 1.0463 w_k1_MSFT_s011
                                 + 0.959468 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.639855 w_k1_ADM_s011
                                 + 0.694462 w_k1_AXP_s011
                                 + 0.877179 w_k1_DIS_s011
                                 + 0.822708 w_k1_FDX_s011
                                 + 0.797931 w_k1_IBM_s011
                                 + 0.884623 w_k1_INTC_s011
                                 + 0.910179 w_k1_KO_s011
                                 + 0.789953 w_k1_MAT_s011
                                 + 0.991303 w_k1_MSFT_s011
                                 + 0.863521 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.37704 w_k1_ADM_s011
                                 + 1.46419 w_k1_AXP_s011
                                 + 2.04453 w_k1_DIS_s011
                                 + 1.69691 w_k1_FDX_s011
                                 + 1.17485 w_k1_IBM_s011
                                 + 1.47207 w_k1_INTC_s011
                                 + 1.71023 w_k1_KO_s011 + 1.50182 w_k1_MAT_s011
                                 + 1.56546 w_k1_MSFT_s011
                                 + 1.48458 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.663511 w_k1_ADM_s011
                                 + 0.658806 w_k1_AXP_s011
                                 + 0.674408 w_k1_DIS_s011
                                 + 0.836107 w_k1_FDX_s011
                                 + 0.601052 w_k1_IBM_s011
                                 + 1.00076 w_k1_INTC_s011
                                 + 0.718383 w_k1_KO_s011
                                 + 0.59878 w_k1_MAT_s011
                                 + 0.858831 w_k1_MSFT_s011
                                 + 0.834139 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.91431 w_k1_ADM_s011
                                 + 1.0572 w_k1_AXP_s011 + 1.29022 w_k1_DIS_s011
                                 + 1.10392 w_k1_FDX_s011
                                 + 1.32265 w_k1_IBM_s011
                                 + 1.08025 w_k1_INTC_s011
                                 + 1.32617 w_k1_KO_s011 + 1.40526 w_k1_MAT_s011
                                 + 1.14254 w_k1_MSFT_s011
                                 + 1.23664 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.17976 w_k1_ADM_s011
                                 + 1.38499 w_k1_AXP_s011
                                 + 1.22444 w_k1_DIS_s011
                                 + 1.22589 w_k1_FDX_s011
                                 + 0.919827 w_k1_IBM_s011
                                 + 0.975327 w_k1_INTC_s011
                                 + 1.19333 w_k1_KO_s011 + 1.30185 w_k1_MAT_s011
                                 + 0.980196 w_k1_MSFT_s011
                                 + 1.28606 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.684772 w_k1_ADM_s011
                                 + 0.608012 w_k1_AXP_s011
                                 + 0.531149 w_k1_DIS_s011
                                 + 0.72108 w_k1_FDX_s011
                                 + 0.860955 w_k1_IBM_s011
                                 + 0.719382 w_k1_INTC_s011
                                 + 0.754152 w_k1_KO_s011
                                 + 0.935815 w_k1_MAT_s011
                                 + 0.647132 w_k1_MSFT_s011
                                 + 0.803985 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.905795 w_k1_ADM_s011
                                 + 1.06776 w_k1_AXP_s011
                                 + 1.60649 w_k1_DIS_s011
                                 + 1.10758 w_k1_FDX_s011
                                 + 1.69818 w_k1_IBM_s011
                                 + 1.17447 w_k1_INTC_s011
                                 + 1.64104 w_k1_KO_s011 + 1.77032 w_k1_MAT_s011
                                 + 1.14611 w_k1_MSFT_s011 + 1.3627 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.770062 w_k1_ADM_s012
                                 + 0.689985 w_k1_AXP_s012
                                 + 0.561793 w_k1_DIS_s012
                                 + 0.787611 w_k1_FDX_s012
                                 + 0.666082 w_k1_IBM_s012
                                 + 0.84983 w_k1_INTC_s012
                                 + 0.568863 w_k1_KO_s012
                                 + 0.506348 w_k1_MAT_s012
                                 + 0.716269 w_k1_MSFT_s012
                                 + 0.66723 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.74965 w_k1_ADM_s012
                                 + 0.721412 w_k1_AXP_s012
                                 + 0.872566 w_k1_DIS_s012
                                 + 0.837084 w_k1_FDX_s012
                                 + 0.883148 w_k1_IBM_s012
                                 + 0.895797 w_k1_INTC_s012
                                 + 0.850022 w_k1_KO_s012
                                 + 0.719115 w_k1_MAT_s012
                                 + 0.80024 w_k1_MSFT_s012
                                 + 0.752617 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.960098 w_k1_ADM_s012
                                 + 0.89375 w_k1_AXP_s012
                                 + 1.09995 w_k1_DIS_s012
                                 + 0.990894 w_k1_FDX_s012
                                 + 0.896922 w_k1_IBM_s012
                                 + 1.00913 w_k1_INTC_s012
                                 + 0.954871 w_k1_KO_s012
                                 + 0.782746 w_k1_MAT_s012
                                 + 0.829832 w_k1_MSFT_s012
                                 + 0.992205 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.941744 w_k1_ADM_s012
                                 + 1.21458 w_k1_AXP_s012
                                 + 1.31315 w_k1_DIS_s012
                                 + 1.28091 w_k1_FDX_s012
                                 + 1.01215 w_k1_IBM_s012
                                 + 1.15371 w_k1_INTC_s012
                                 + 1.09375 w_k1_KO_s012
                                 + 0.944477 w_k1_MAT_s012
                                 + 0.975918 w_k1_MSFT_s012
                                 + 1.01551 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.959027 w_k1_ADM_s012
                                 + 0.99907 w_k1_AXP_s012
                                 + 1.30567 w_k1_DIS_s012
                                 + 0.990167 w_k1_FDX_s012
                                 + 1.4561 w_k1_IBM_s012
                                 + 1.05987 w_k1_INTC_s012
                                 + 1.27668 w_k1_KO_s012 + 1.61591 w_k1_MAT_s012
                                 + 0.91591 w_k1_MSFT_s012
                                 + 1.06985 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.984569 w_k1_ADM_s012
                                 + 0.777079 w_k1_AXP_s012
                                 + 0.932526 w_k1_DIS_s012
                                 + 0.9468 w_k1_FDX_s012
                                 + 0.678037 w_k1_IBM_s012
                                 + 0.834975 w_k1_INTC_s012
                                 + 0.815345 w_k1_KO_s012
                                 + 0.74345 w_k1_MAT_s012
                                 + 0.930331 w_k1_MSFT_s012
                                 + 0.911967 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.969806 w_k1_ADM_s012
                                 + 1.04314 w_k1_AXP_s012
                                 + 1.00941 w_k1_DIS_s012
                                 + 1.08132 w_k1_FDX_s012
                                 + 0.830582 w_k1_IBM_s012
                                 + 1.08134 w_k1_INTC_s012
                                 + 0.797717 w_k1_KO_s012
                                 + 0.785017 w_k1_MAT_s012
                                 + 0.851143 w_k1_MSFT_s012
                                 + 0.9332 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.810331 w_k1_ADM_s012
                                 + 1.0036 w_k1_AXP_s012
                                 + 0.807477 w_k1_DIS_s012
                                 + 0.962912 w_k1_FDX_s012
                                 + 1.01433 w_k1_IBM_s012
                                 + 0.971516 w_k1_INTC_s012
                                 + 0.854875 w_k1_KO_s012
                                 + 0.845453 w_k1_MAT_s012
                                 + 0.875103 w_k1_MSFT_s012
                                 + 0.809274 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.06644 w_k1_ADM_s013
                                 + 1.01476 w_k1_AXP_s013
                                 + 0.935372 w_k1_DIS_s013
                                 + 0.965459 w_k1_FDX_s013
                                 + 0.829559 w_k1_IBM_s013
                                 + 1.05051 w_k1_INTC_s013
                                 + 0.811096 w_k1_KO_s013
                                 + 0.713671 w_k1_MAT_s013
                                 + 0.912419 w_k1_MSFT_s013
                                 + 0.859998 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.23283 w_k1_ADM_s013
                                 + 0.820443 w_k1_AXP_s013
                                 + 0.910744 w_k1_DIS_s013
                                 + 0.884313 w_k1_FDX_s013
                                 + 0.749152 w_k1_IBM_s013
                                 + 0.973121 w_k1_INTC_s013
                                 + 0.759534 w_k1_KO_s013
                                 + 0.498705 w_k1_MAT_s013
                                 + 0.865529 w_k1_MSFT_s013
                                 + 0.852576 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.712024 w_k1_ADM_s013
                                 + 0.674669 w_k1_AXP_s013
                                 + 0.582627 w_k1_DIS_s013
                                 + 0.667355 w_k1_FDX_s013
                                 + 0.688414 w_k1_IBM_s013
                                 + 1.0276 w_k1_INTC_s013
                                 + 0.608027 w_k1_KO_s013
                                 + 0.542554 w_k1_MAT_s013
                                 + 0.796653 w_k1_MSFT_s013
                                 + 0.794353 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.03621 w_k1_ADM_s013
                                 + 1.49447 w_k1_AXP_s013
                                 + 1.30135 w_k1_DIS_s013
                                 + 1.33952 w_k1_FDX_s013 + 1.1959 w_k1_IBM_s013
                                 + 1.62165 w_k1_INTC_s013
                                 + 1.11543 w_k1_KO_s013 + 1.28412 w_k1_MAT_s013
                                 + 1.25123 w_k1_MSFT_s013
                                 + 1.07526 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.882284 w_k1_ADM_s013
                                 + 0.911294 w_k1_AXP_s013
                                 + 0.982779 w_k1_DIS_s013
                                 + 0.896036 w_k1_FDX_s013
                                 + 1.27806 w_k1_IBM_s013
                                 + 1.18756 w_k1_INTC_s013
                                 + 1.01573 w_k1_KO_s013 + 1.0644 w_k1_MAT_s013
                                 + 1.00102 w_k1_MSFT_s013 + 1.0204 w_k1_PG_s013
                                 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.30192 w_k1_ADM_s013
                                 + 1.08574 w_k1_AXP_s013
                                 + 1.07801 w_k1_DIS_s013
                                 + 1.13781 w_k1_FDX_s013
                                 + 1.07595 w_k1_IBM_s013
                                 + 1.1729 w_k1_INTC_s013 + 1.02328 w_k1_KO_s013
                                 + 0.932053 w_k1_MAT_s013
                                 + 1.0602 w_k1_MSFT_s013 + 1.03192 w_k1_PG_s013
                                 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.05517 w_k1_ADM_s013
                                 + 1.06751 w_k1_AXP_s013
                                 + 1.84123 w_k1_DIS_s013 + 1.1098 w_k1_FDX_s013
                                 + 1.32597 w_k1_IBM_s013
                                 + 1.39918 w_k1_INTC_s013
                                 + 1.33579 w_k1_KO_s013 + 1.01147 w_k1_MAT_s013
                                 + 1.36655 w_k1_MSFT_s013
                                 + 1.23784 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.943187 w_k1_ADM_s013
                                 + 1.01982 w_k1_AXP_s013
                                 + 0.889676 w_k1_DIS_s013
                                 + 0.897502 w_k1_FDX_s013
                                 + 0.984101 w_k1_IBM_s013
                                 + 1.0357 w_k1_INTC_s013
                                 + 0.846235 w_k1_KO_s013
                                 + 1.01717 w_k1_MAT_s013
                                 + 0.902225 w_k1_MSFT_s013
                                 + 0.878897 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.20251 w_k1_ADM_s014
                                 + 1.2343 w_k1_AXP_s014 + 1.10503 w_k1_DIS_s014
                                 + 1.0402 w_k1_FDX_s014 + 1.39378 w_k1_IBM_s014
                                 + 1.13249 w_k1_INTC_s014
                                 + 1.40506 w_k1_KO_s014 + 1.18641 w_k1_MAT_s014
                                 + 1.03482 w_k1_MSFT_s014
                                 + 1.09626 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.12529 w_k1_ADM_s014
                                 + 1.63073 w_k1_AXP_s014
                                 + 0.865539 w_k1_DIS_s014
                                 + 1.03302 w_k1_FDX_s014
                                 + 0.999035 w_k1_IBM_s014
                                 + 1.21703 w_k1_INTC_s014
                                 + 1.02209 w_k1_KO_s014 + 1.47492 w_k1_MAT_s014
                                 + 1.11527 w_k1_MSFT_s014
                                 + 1.00521 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.905634 w_k1_ADM_s014
                                 + 0.936514 w_k1_AXP_s014
                                 + 0.64107 w_k1_DIS_s014
                                 + 0.883943 w_k1_FDX_s014
                                 + 0.658501 w_k1_IBM_s014
                                 + 1.14672 w_k1_INTC_s014
                                 + 0.809228 w_k1_KO_s014
                                 + 0.597649 w_k1_MAT_s014
                                 + 1.00307 w_k1_MSFT_s014
                                 + 0.932514 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.33509 w_k1_ADM_s014
                                 + 0.996684 w_k1_AXP_s014
                                 + 1.21606 w_k1_DIS_s014
                                 + 1.01232 w_k1_FDX_s014
                                 + 1.05417 w_k1_IBM_s014
                                 + 1.14358 w_k1_INTC_s014
                                 + 1.40756 w_k1_KO_s014
                                 + 0.865381 w_k1_MAT_s014
                                 + 1.15386 w_k1_MSFT_s014
                                 + 1.03901 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.32341 w_k1_ADM_s014
                                 + 1.20897 w_k1_AXP_s014
                                 + 1.32349 w_k1_DIS_s014
                                 + 1.29364 w_k1_FDX_s014 + 0.8634 w_k1_IBM_s014
                                 + 1.22234 w_k1_INTC_s014
                                 + 1.09258 w_k1_KO_s014
                                 + 0.808552 w_k1_MAT_s014
                                 + 1.26746 w_k1_MSFT_s014
                                 + 1.04083 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.21832 w_k1_ADM_s014
                                 + 0.841193 w_k1_AXP_s014
                                 + 0.876453 w_k1_DIS_s014
                                 + 0.96963 w_k1_FDX_s014
                                 + 0.853801 w_k1_IBM_s014
                                 + 0.965553 w_k1_INTC_s014
                                 + 0.988711 w_k1_KO_s014
                                 + 0.61638 w_k1_MAT_s014
                                 + 0.928786 w_k1_MSFT_s014
                                 + 0.836889 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.884035 w_k1_ADM_s014
                                 + 0.82717 w_k1_AXP_s014
                                 + 0.636258 w_k1_DIS_s014
                                 + 0.763161 w_k1_FDX_s014
                                 + 0.797448 w_k1_IBM_s014
                                 + 1.05735 w_k1_INTC_s014
                                 + 0.828481 w_k1_KO_s014
                                 + 1.07919 w_k1_MAT_s014
                                 + 0.991203 w_k1_MSFT_s014
                                 + 0.83972 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.20831 w_k1_ADM_s014
                                 + 1.39652 w_k1_AXP_s014
                                 + 1.85746 w_k1_DIS_s014
                                 + 1.32302 w_k1_FDX_s014
                                 + 1.24983 w_k1_IBM_s014
                                 + 1.60876 w_k1_INTC_s014
                                 + 1.77174 w_k1_KO_s014 + 1.12386 w_k1_MAT_s014
                                 + 1.47155 w_k1_MSFT_s014
                                 + 1.40579 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.756406 w_k1_ADM_s015
                                 + 0.857602 w_k1_AXP_s015
                                 + 0.591209 w_k1_DIS_s015
                                 + 0.711975 w_k1_FDX_s015
                                 + 0.801378 w_k1_IBM_s015
                                 + 0.943328 w_k1_INTC_s015
                                 + 0.717278 w_k1_KO_s015
                                 + 0.739918 w_k1_MAT_s015
                                 + 0.783004 w_k1_MSFT_s015
                                 + 0.727515 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.747949 w_k1_ADM_s015
                                 + 0.894398 w_k1_AXP_s015
                                 + 0.628061 w_k1_DIS_s015
                                 + 0.788103 w_k1_FDX_s015
                                 + 1.07274 w_k1_IBM_s015
                                 + 0.875035 w_k1_INTC_s015
                                 + 0.70295 w_k1_KO_s015
                                 + 0.910117 w_k1_MAT_s015
                                 + 0.697773 w_k1_MSFT_s015
                                 + 0.726414 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.7858 w_k1_ADM_s015
                                 + 0.852162 w_k1_AXP_s015
                                 + 0.788813 w_k1_DIS_s015
                                 + 0.943731 w_k1_FDX_s015
                                 + 0.687006 w_k1_IBM_s015
                                 + 1.11207 w_k1_INTC_s015
                                 + 0.718091 w_k1_KO_s015
                                 + 0.675685 w_k1_MAT_s015
                                 + 0.93373 w_k1_MSFT_s015
                                 + 0.798547 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.58266 w_k1_ADM_s015
                                 + 1.40946 w_k1_AXP_s015
                                 + 1.04661 w_k1_DIS_s015 + 1.1629 w_k1_FDX_s015
                                 + 1.19308 w_k1_IBM_s015
                                 + 1.1475 w_k1_INTC_s015 + 1.13959 w_k1_KO_s015
                                 + 1.41746 w_k1_MAT_s015
                                 + 1.06479 w_k1_MSFT_s015
                                 + 1.01117 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.09032 w_k1_ADM_s015
                                 + 1.19872 w_k1_AXP_s015
                                 + 0.924329 w_k1_DIS_s015
                                 + 1.03256 w_k1_FDX_s015
                                 + 1.03426 w_k1_IBM_s015
                                 + 1.09259 w_k1_INTC_s015 + 1.0112 w_k1_KO_s015
                                 + 1.35246 w_k1_MAT_s015
                                 + 0.923627 w_k1_MSFT_s015 + 1.047 w_k1_PG_s015
                                 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.24643 w_k1_ADM_s015
                                 + 1.10468 w_k1_AXP_s015
                                 + 0.719761 w_k1_DIS_s015
                                 + 0.917337 w_k1_FDX_s015
                                 + 0.76741 w_k1_IBM_s015
                                 + 1.05733 w_k1_INTC_s015
                                 + 0.775265 w_k1_KO_s015
                                 + 0.890548 w_k1_MAT_s015
                                 + 0.884429 w_k1_MSFT_s015
                                 + 0.930513 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.54624 w_k1_ADM_s015
                                 + 1.27115 w_k1_AXP_s015 + 1.3875 w_k1_DIS_s015
                                 + 1.18567 w_k1_FDX_s015
                                 + 0.929569 w_k1_IBM_s015
                                 + 1.13723 w_k1_INTC_s015
                                 + 1.10246 w_k1_KO_s015 + 1.04435 w_k1_MAT_s015
                                 + 1.17615 w_k1_MSFT_s015 + 1.1117 w_k1_PG_s015
                                 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.43389 w_k1_ADM_s015
                                 + 1.34814 w_k1_AXP_s015 + 1.3686 w_k1_DIS_s015
                                 + 1.12519 w_k1_FDX_s015
                                 + 1.46943 w_k1_IBM_s015
                                 + 1.25796 w_k1_INTC_s015
                                 + 1.29749 w_k1_KO_s015 + 1.22476 w_k1_MAT_s015
                                 + 1.11879 w_k1_MSFT_s015
                                 + 1.02105 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.994866 w_k1_ADM_s016
                                 + 0.925941 w_k1_AXP_s016
                                 + 1.0262 w_k1_DIS_s016 + 0.87399 w_k1_FDX_s016
                                 + 1.28455 w_k1_IBM_s016
                                 + 1.27212 w_k1_INTC_s016
                                 + 1.24137 w_k1_KO_s016 + 1.21006 w_k1_MAT_s016
                                 + 1.08775 w_k1_MSFT_s016
                                 + 0.925524 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.832434 w_k1_ADM_s016
                                 + 0.81492 w_k1_AXP_s016
                                 + 0.74064 w_k1_DIS_s016
                                 + 0.842129 w_k1_FDX_s016
                                 + 0.932736 w_k1_IBM_s016
                                 + 0.851843 w_k1_INTC_s016
                                 + 0.739263 w_k1_KO_s016
                                 + 0.967459 w_k1_MAT_s016
                                 + 0.877715 w_k1_MSFT_s016
                                 + 0.872796 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.23276 w_k1_ADM_s016
                                 + 1.41837 w_k1_AXP_s016
                                 + 1.68322 w_k1_DIS_s016
                                 + 1.35345 w_k1_FDX_s016
                                 + 1.07333 w_k1_IBM_s016
                                 + 1.43466 w_k1_INTC_s016
                                 + 1.30015 w_k1_KO_s016 + 1.01889 w_k1_MAT_s016
                                 + 1.26576 w_k1_MSFT_s016 + 1.0137 w_k1_PG_s016
                                 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.06428 w_k1_ADM_s016
                                 + 0.770979 w_k1_AXP_s016
                                 + 0.835268 w_k1_DIS_s016
                                 + 0.872921 w_k1_FDX_s016
                                 + 0.706695 w_k1_IBM_s016
                                 + 0.970732 w_k1_INTC_s016
                                 + 0.810291 w_k1_KO_s016
                                 + 0.620934 w_k1_MAT_s016
                                 + 0.937181 w_k1_MSFT_s016
                                 + 0.846332 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.3763 w_k1_ADM_s016
                                 + 1.43271 w_k1_AXP_s016
                                 + 1.33662 w_k1_DIS_s016
                                 + 1.18743 w_k1_FDX_s016
                                 + 1.28746 w_k1_IBM_s016
                                 + 1.35828 w_k1_INTC_s016
                                 + 1.26184 w_k1_KO_s016 + 1.22629 w_k1_MAT_s016
                                 + 1.10113 w_k1_MSFT_s016
                                 + 1.22819 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.26812 w_k1_ADM_s016
                                 + 1.6236 w_k1_AXP_s016 + 2.19209 w_k1_DIS_s016
                                 + 1.37275 w_k1_FDX_s016
                                 + 1.42091 w_k1_IBM_s016
                                 + 1.77862 w_k1_INTC_s016
                                 + 1.44413 w_k1_KO_s016
                                 + 0.947733 w_k1_MAT_s016
                                 + 1.37562 w_k1_MSFT_s016
                                 + 1.05464 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.08595 w_k1_ADM_s016
                                 + 1.00042 w_k1_AXP_s016
                                 + 1.26198 w_k1_DIS_s016 + 1.0964 w_k1_FDX_s016
                                 + 0.868342 w_k1_IBM_s016
                                 + 1.28201 w_k1_INTC_s016
                                 + 0.954551 w_k1_KO_s016
                                 + 0.643412 w_k1_MAT_s016
                                 + 1.13098 w_k1_MSFT_s016
                                 + 0.912723 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.74985 w_k1_ADM_s016
                                 + 0.827147 w_k1_AXP_s016
                                 + 0.690973 w_k1_DIS_s016
                                 + 0.863594 w_k1_FDX_s016
                                 + 0.665406 w_k1_IBM_s016
                                 + 1.10265 w_k1_INTC_s016
                                 + 0.634008 w_k1_KO_s016
                                 + 0.458222 w_k1_MAT_s016
                                 + 0.853028 w_k1_MSFT_s016
                                 + 0.74492 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.4821 w_k1_ADM_s017
                                 + 1.17358 w_k1_AXP_s017
                                 + 1.21903 w_k1_DIS_s017
                                 + 0.989897 w_k1_FDX_s017
                                 + 1.4307 w_k1_IBM_s017
                                 + 1.24657 w_k1_INTC_s017
                                 + 1.31589 w_k1_KO_s017
                                 + 0.922544 w_k1_MAT_s017
                                 + 1.0975 w_k1_MSFT_s017 + 1.10838 w_k1_PG_s017
                                 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.750543 w_k1_ADM_s017
                                 + 0.665981 w_k1_AXP_s017
                                 + 0.754412 w_k1_DIS_s017
                                 + 0.756746 w_k1_FDX_s017
                                 + 1.02225 w_k1_IBM_s017
                                 + 0.951129 w_k1_INTC_s017
                                 + 1.02048 w_k1_KO_s017
                                 + 0.927726 w_k1_MAT_s017
                                 + 0.678986 w_k1_MSFT_s017
                                 + 0.912585 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.982587 w_k1_ADM_s017
                                 + 0.683 w_k1_AXP_s017 + 0.934476 w_k1_DIS_s017
                                 + 0.894119 w_k1_FDX_s017
                                 + 0.747947 w_k1_IBM_s017
                                 + 0.896937 w_k1_INTC_s017
                                 + 0.965976 w_k1_KO_s017
                                 + 0.653347 w_k1_MAT_s017
                                 + 0.902923 w_k1_MSFT_s017
                                 + 0.92528 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.974651 w_k1_ADM_s017
                                 + 0.80121 w_k1_AXP_s017
                                 + 0.419314 w_k1_DIS_s017
                                 + 0.76147 w_k1_FDX_s017
                                 + 0.664316 w_k1_IBM_s017
                                 + 0.692863 w_k1_INTC_s017
                                 + 0.610468 w_k1_KO_s017
                                 + 0.698359 w_k1_MAT_s017
                                 + 0.57561 w_k1_MSFT_s017
                                 + 0.828104 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.37993 w_k1_ADM_s017
                                 + 1.57373 w_k1_AXP_s017
                                 + 1.26034 w_k1_DIS_s017
                                 + 1.29665 w_k1_FDX_s017
                                 + 1.11018 w_k1_IBM_s017
                                 + 1.24762 w_k1_INTC_s017
                                 + 1.27653 w_k1_KO_s017 + 1.52803 w_k1_MAT_s017
                                 + 1.0661 w_k1_MSFT_s017 + 1.14767 w_k1_PG_s017
                                 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.20203 w_k1_ADM_s017
                                 + 1.17311 w_k1_AXP_s017
                                 + 1.04165 w_k1_DIS_s017
                                 + 1.24426 w_k1_FDX_s017
                                 + 0.886055 w_k1_IBM_s017
                                 + 1.33755 w_k1_INTC_s017
                                 + 0.99829 w_k1_KO_s017
                                 + 0.801501 w_k1_MAT_s017
                                 + 1.00686 w_k1_MSFT_s017
                                 + 1.08259 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.998729 w_k1_ADM_s017
                                 + 0.989192 w_k1_AXP_s017
                                 + 0.81975 w_k1_DIS_s017
                                 + 0.892996 w_k1_FDX_s017
                                 + 0.932018 w_k1_IBM_s017
                                 + 1.14741 w_k1_INTC_s017
                                 + 0.964982 w_k1_KO_s017
                                 + 0.735965 w_k1_MAT_s017
                                 + 0.81378 w_k1_MSFT_s017
                                 + 0.875623 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.918125 w_k1_ADM_s017
                                 + 0.899416 w_k1_AXP_s017
                                 + 1.00881 w_k1_DIS_s017
                                 + 0.829343 w_k1_FDX_s017
                                 + 1.18884 w_k1_IBM_s017
                                 + 1.04631 w_k1_INTC_s017
                                 + 1.04597 w_k1_KO_s017 + 1.16569 w_k1_MAT_s017
                                 + 0.875903 w_k1_MSFT_s017
                                 + 1.07346 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.816932 w_k1_ADM_s100
                                 + 0.862784 w_k1_AXP_s100
                                 + 0.718621 w_k1_DIS_s100
                                 + 0.804707 w_k1_FDX_s100
                                 + 1.31806 w_k1_IBM_s100
                                 + 0.803149 w_k1_INTC_s100
                                 + 0.796375 w_k1_KO_s100
                                 + 1.03387 w_k1_MAT_s100
                                 + 0.715371 w_k1_MSFT_s100
                                 + 0.917353 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.21009 w_k1_ADM_s100
                                 + 0.857778 w_k1_AXP_s100
                                 + 1.24503 w_k1_DIS_s100
                                 + 1.06544 w_k1_FDX_s100
                                 + 0.805516 w_k1_IBM_s100
                                 + 0.970852 w_k1_INTC_s100
                                 + 0.971839 w_k1_KO_s100
                                 + 0.636898 w_k1_MAT_s100
                                 + 1.09801 w_k1_MSFT_s100
                                 + 0.997535 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.98222 w_k1_ADM_s100
                                 + 0.983583 w_k1_AXP_s100
                                 + 0.961107 w_k1_DIS_s100
                                 + 0.930436 w_k1_FDX_s100
                                 + 1.1692 w_k1_IBM_s100
                                 + 0.799597 w_k1_INTC_s100
                                 + 1.02741 w_k1_KO_s100 + 1.12974 w_k1_MAT_s100
                                 + 0.876642 w_k1_MSFT_s100
                                 + 1.03466 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.27749 w_k1_ADM_s100
                                 + 1.1927 w_k1_AXP_s100 + 1.09328 w_k1_DIS_s100
                                 + 1.28738 w_k1_FDX_s100
                                 + 0.839095 w_k1_IBM_s100
                                 + 1.02404 w_k1_INTC_s100
                                 + 0.843888 w_k1_KO_s100
                                 + 0.688434 w_k1_MAT_s100
                                 + 0.990573 w_k1_MSFT_s100
                                 + 1.0998 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.986784 w_k1_ADM_s100
                                 + 0.831318 w_k1_AXP_s100
                                 + 0.629427 w_k1_DIS_s100
                                 + 0.832721 w_k1_FDX_s100
                                 + 0.747353 w_k1_IBM_s100
                                 + 0.689255 w_k1_INTC_s100
                                 + 0.731564 w_k1_KO_s100
                                 + 0.510109 w_k1_MAT_s100
                                 + 0.782683 w_k1_MSFT_s100
                                 + 0.83925 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.983371 w_k1_ADM_s100
                                 + 1.07123 w_k1_AXP_s100
                                 + 0.828759 w_k1_DIS_s100
                                 + 0.979046 w_k1_FDX_s100
                                 + 0.890253 w_k1_IBM_s100
                                 + 0.972041 w_k1_INTC_s100
                                 + 0.851441 w_k1_KO_s100
                                 + 0.871709 w_k1_MAT_s100
                                 + 0.829551 w_k1_MSFT_s100
                                 + 0.988695 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.01703 w_k1_ADM_s100
                                 + 1.34412 w_k1_AXP_s100
                                 + 1.32585 w_k1_DIS_s100
                                 + 1.19619 w_k1_FDX_s100
                                 + 1.26239 w_k1_IBM_s100
                                 + 1.18873 w_k1_INTC_s100 + 1.1467 w_k1_KO_s100
                                 + 1.22871 w_k1_MAT_s100
                                 + 1.16446 w_k1_MSFT_s100
                                 + 1.28176 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.34307 w_k1_ADM_s100
                                 + 1.83921 w_k1_AXP_s100
                                 + 1.49662 w_k1_DIS_s100
                                 + 1.43635 w_k1_FDX_s100
                                 + 1.13793 w_k1_IBM_s100
                                 + 1.59677 w_k1_INTC_s100 + 1.2633 w_k1_KO_s100
                                 + 1.24655 w_k1_MAT_s100
                                 + 1.32996 w_k1_MSFT_s100
                                 + 1.28373 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.0324 w_k1_ADM_s101
                                 + 1.24989 w_k1_AXP_s101
                                 + 1.34121 w_k1_DIS_s101
                                 + 1.07233 w_k1_FDX_s101
                                 + 1.82395 w_k1_IBM_s101
                                 + 1.02778 w_k1_INTC_s101
                                 + 1.41947 w_k1_KO_s101 + 1.7474 w_k1_MAT_s101
                                 + 1.08135 w_k1_MSFT_s101
                                 + 1.15401 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.09168 w_k1_ADM_s101
                                 + 1.23535 w_k1_AXP_s101
                                 + 1.50519 w_k1_DIS_s101
                                 + 1.30237 w_k1_FDX_s101
                                 + 1.30252 w_k1_IBM_s101
                                 + 1.29503 w_k1_INTC_s101
                                 + 1.26947 w_k1_KO_s101 + 1.36935 w_k1_MAT_s101
                                 + 1.24371 w_k1_MSFT_s101
                                 + 1.12339 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.938818 w_k1_ADM_s101
                                 + 0.743004 w_k1_AXP_s101
                                 + 0.814665 w_k1_DIS_s101
                                 + 0.949781 w_k1_FDX_s101
                                 + 0.559958 w_k1_IBM_s101
                                 + 0.833228 w_k1_INTC_s101
                                 + 0.68481 w_k1_KO_s101
                                 + 0.572983 w_k1_MAT_s101
                                 + 0.876566 w_k1_MSFT_s101
                                 + 0.799043 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.18938 w_k1_ADM_s101
                                 + 1.12315 w_k1_AXP_s101
                                 + 1.05742 w_k1_DIS_s101
                                 + 1.04451 w_k1_FDX_s101 + 1.1547 w_k1_IBM_s101
                                 + 0.988691 w_k1_INTC_s101
                                 + 1.07348 w_k1_KO_s101 + 1.42856 w_k1_MAT_s101
                                 + 0.982963 w_k1_MSFT_s101
                                 + 0.958017 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.859525 w_k1_ADM_s101
                                 + 0.7692 w_k1_AXP_s101
                                 + 0.769948 w_k1_DIS_s101
                                 + 0.823148 w_k1_FDX_s101
                                 + 1.29174 w_k1_IBM_s101
                                 + 0.969733 w_k1_INTC_s101
                                 + 1.03673 w_k1_KO_s101 + 1.01572 w_k1_MAT_s101
                                 + 0.890484 w_k1_MSFT_s101
                                 + 0.915241 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.765191 w_k1_ADM_s101
                                 + 0.701474 w_k1_AXP_s101
                                 + 0.651768 w_k1_DIS_s101
                                 + 0.739087 w_k1_FDX_s101
                                 + 0.773485 w_k1_IBM_s101
                                 + 0.855223 w_k1_INTC_s101
                                 + 0.785687 w_k1_KO_s101
                                 + 0.808163 w_k1_MAT_s101
                                 + 0.787737 w_k1_MSFT_s101
                                 + 0.786608 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.02239 w_k1_ADM_s101
                                 + 1.0128 w_k1_AXP_s101 + 0.77243 w_k1_DIS_s101
                                 + 0.925969 w_k1_FDX_s101
                                 + 0.858991 w_k1_IBM_s101
                                 + 1.02085 w_k1_INTC_s101
                                 + 0.78969 w_k1_KO_s101
                                 + 0.688672 w_k1_MAT_s101
                                 + 0.898576 w_k1_MSFT_s101
                                 + 0.900031 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.17491 w_k1_ADM_s101
                                 + 1.83289 w_k1_AXP_s101
                                 + 1.49634 w_k1_DIS_s101
                                 + 1.42632 w_k1_FDX_s101
                                 + 1.20432 w_k1_IBM_s101
                                 + 1.26071 w_k1_INTC_s101
                                 + 1.11629 w_k1_KO_s101 + 1.60179 w_k1_MAT_s101
                                 + 1.1841 w_k1_MSFT_s101 + 1.1658 w_k1_PG_s101
                                 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.917136 w_k1_ADM_s102
                                 + 0.658451 w_k1_AXP_s102
                                 + 0.624145 w_k1_DIS_s102
                                 + 0.679931 w_k1_FDX_s102
                                 + 0.973312 w_k1_IBM_s102
                                 + 0.601438 w_k1_INTC_s102
                                 + 0.754596 w_k1_KO_s102
                                 + 0.807102 w_k1_MAT_s102
                                 + 0.692832 w_k1_MSFT_s102
                                 + 0.836772 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.30776 w_k1_ADM_s102
                                 + 0.755402 w_k1_AXP_s102
                                 + 0.701966 w_k1_DIS_s102
                                 + 0.761588 w_k1_FDX_s102
                                 + 1.01179 w_k1_IBM_s102
                                 + 0.546831 w_k1_INTC_s102
                                 + 0.81391 w_k1_KO_s102 + 0.69776 w_k1_MAT_s102
                                 + 0.654054 w_k1_MSFT_s102
                                 + 0.880638 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.35072 w_k1_ADM_s102
                                 + 1.0211 w_k1_AXP_s102 + 0.99202 w_k1_DIS_s102
                                 + 0.979455 w_k1_FDX_s102
                                 + 0.742562 w_k1_IBM_s102
                                 + 0.893464 w_k1_INTC_s102
                                 + 1.02825 w_k1_KO_s102 + 0.77971 w_k1_MAT_s102
                                 + 1.0573 w_k1_MSFT_s102
                                 + 0.913802 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.05948 w_k1_ADM_s102
                                 + 1.12421 w_k1_AXP_s102 + 1.0309 w_k1_DIS_s102
                                 + 1.04814 w_k1_FDX_s102
                                 + 1.32935 w_k1_IBM_s102
                                 + 0.892021 w_k1_INTC_s102
                                 + 1.00779 w_k1_KO_s102 + 1.15604 w_k1_MAT_s102
                                 + 1.00574 w_k1_MSFT_s102
                                 + 1.05486 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.58946 w_k1_ADM_s102
                                 + 1.33008 w_k1_AXP_s102
                                 + 1.29091 w_k1_DIS_s102
                                 + 1.23039 w_k1_FDX_s102
                                 + 1.04638 w_k1_IBM_s102
                                 + 1.08487 w_k1_INTC_s102 + 1.1561 w_k1_KO_s102
                                 + 0.913829 w_k1_MAT_s102
                                 + 1.26232 w_k1_MSFT_s102
                                 + 1.18701 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 2.01218 w_k1_ADM_s102
                                 + 1.62004 w_k1_AXP_s102
                                 + 1.82346 w_k1_DIS_s102 + 1.3574 w_k1_FDX_s102
                                 + 1.40637 w_k1_IBM_s102
                                 + 1.12161 w_k1_INTC_s102
                                 + 1.71375 w_k1_KO_s102 + 1.4505 w_k1_MAT_s102
                                 + 1.50074 w_k1_MSFT_s102
                                 + 1.30578 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.16056 w_k1_ADM_s102
                                 + 1.21155 w_k1_AXP_s102 + 1.1976 w_k1_DIS_s102
                                 + 0.982948 w_k1_FDX_s102
                                 + 1.70771 w_k1_IBM_s102
                                 + 1.07726 w_k1_INTC_s102 + 1.4746 w_k1_KO_s102
                                 + 1.44279 w_k1_MAT_s102
                                 + 1.17686 w_k1_MSFT_s102
                                 + 1.31307 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.57867 w_k1_ADM_s102
                                 + 1.08052 w_k1_AXP_s102
                                 + 1.02096 w_k1_DIS_s102
                                 + 0.960963 w_k1_FDX_s102
                                 + 1.02431 w_k1_IBM_s102
                                 + 0.702121 w_k1_INTC_s102
                                 + 1.09096 w_k1_KO_s102
                                 + 0.871401 w_k1_MAT_s102
                                 + 0.882189 w_k1_MSFT_s102
                                 + 0.957034 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.849815 w_k1_ADM_s103
                                 + 0.719969 w_k1_AXP_s103
                                 + 0.867489 w_k1_DIS_s103
                                 + 0.907379 w_k1_FDX_s103
                                 + 0.700258 w_k1_IBM_s103
                                 + 0.986979 w_k1_INTC_s103
                                 + 0.753791 w_k1_KO_s103
                                 + 0.753486 w_k1_MAT_s103
                                 + 0.997248 w_k1_MSFT_s103
                                 + 0.873108 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.975442 w_k1_ADM_s103
                                 + 1.03576 w_k1_AXP_s103
                                 + 0.765532 w_k1_DIS_s103
                                 + 0.915506 w_k1_FDX_s103
                                 + 0.687084 w_k1_IBM_s103
                                 + 0.92999 w_k1_INTC_s103
                                 + 0.819732 w_k1_KO_s103
                                 + 0.984858 w_k1_MAT_s103
                                 + 0.875851 w_k1_MSFT_s103
                                 + 0.929125 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.758725 w_k1_ADM_s103
                                 + 0.619628 w_k1_AXP_s103
                                 + 0.571804 w_k1_DIS_s103
                                 + 0.712787 w_k1_FDX_s103
                                 + 0.686593 w_k1_IBM_s103
                                 + 0.824083 w_k1_INTC_s103
                                 + 0.71341 w_k1_KO_s103
                                 + 0.661263 w_k1_MAT_s103
                                 + 0.666938 w_k1_MSFT_s103
                                 + 0.723513 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.18421 w_k1_ADM_s103
                                 + 1.49243 w_k1_AXP_s103
                                 + 1.15245 w_k1_DIS_s103
                                 + 1.26215 w_k1_FDX_s103
                                 + 0.865241 w_k1_IBM_s103
                                 + 1.12855 w_k1_INTC_s103
                                 + 0.947064 w_k1_KO_s103 + 1.3287 w_k1_MAT_s103
                                 + 1.12436 w_k1_MSFT_s103
                                 + 1.07732 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.02212 w_k1_ADM_s103
                                 + 0.999784 w_k1_AXP_s103
                                 + 1.09387 w_k1_DIS_s103
                                 + 1.05446 w_k1_FDX_s103
                                 + 0.94929 w_k1_IBM_s103
                                 + 1.10189 w_k1_INTC_s103
                                 + 1.16454 w_k1_KO_s103 + 1.13834 w_k1_MAT_s103
                                 + 1.13444 w_k1_MSFT_s103
                                 + 0.970749 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.901255 w_k1_ADM_s103
                                 + 0.940798 w_k1_AXP_s103
                                 + 0.767658 w_k1_DIS_s103
                                 + 0.818518 w_k1_FDX_s103
                                 + 1.07404 w_k1_IBM_s103
                                 + 0.793332 w_k1_INTC_s103
                                 + 0.772786 w_k1_KO_s103
                                 + 1.16039 w_k1_MAT_s103
                                 + 0.795809 w_k1_MSFT_s103
                                 + 0.870423 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.822443 w_k1_ADM_s103
                                 + 0.679895 w_k1_AXP_s103
                                 + 0.770672 w_k1_DIS_s103
                                 + 0.844806 w_k1_FDX_s103
                                 + 0.852882 w_k1_IBM_s103
                                 + 0.863031 w_k1_INTC_s103
                                 + 0.948645 w_k1_KO_s103
                                 + 1.25558 w_k1_MAT_s103
                                 + 0.885263 w_k1_MSFT_s103
                                 + 0.884193 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.2677 w_k1_ADM_s103
                                 + 1.22671 w_k1_AXP_s103
                                 + 1.36397 w_k1_DIS_s103
                                 + 1.10812 w_k1_FDX_s103
                                 + 1.46027 w_k1_IBM_s103
                                 + 1.13294 w_k1_INTC_s103
                                 + 1.34454 w_k1_KO_s103 + 1.69931 w_k1_MAT_s103
                                 + 1.15197 w_k1_MSFT_s103
                                 + 1.10202 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.922892 w_k1_ADM_s104
                                 + 1.19414 w_k1_AXP_s104
                                 + 0.995825 w_k1_DIS_s104
                                 + 1.0945 w_k1_FDX_s104 + 1.1418 w_k1_IBM_s104
                                 + 1.05505 w_k1_INTC_s104 + 1.0069 w_k1_KO_s104
                                 + 0.960294 w_k1_MAT_s104
                                 + 0.908416 w_k1_MSFT_s104
                                 + 0.95356 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.43877 w_k1_ADM_s104
                                 + 1.38169 w_k1_AXP_s104
                                 + 1.77249 w_k1_DIS_s104
                                 + 1.40173 w_k1_FDX_s104
                                 + 1.08552 w_k1_IBM_s104
                                 + 1.22171 w_k1_INTC_s104
                                 + 1.36891 w_k1_KO_s104 + 1.30219 w_k1_MAT_s104
                                 + 1.38989 w_k1_MSFT_s104
                                 + 1.16799 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.00324 w_k1_ADM_s104
                                 + 0.998584 w_k1_AXP_s104
                                 + 1.6416 w_k1_DIS_s104 + 1.30779 w_k1_FDX_s104
                                 + 0.819969 w_k1_IBM_s104
                                 + 1.14412 w_k1_INTC_s104
                                 + 1.13577 w_k1_KO_s104
                                 + 0.863799 w_k1_MAT_s104
                                 + 1.25838 w_k1_MSFT_s104 + 1.0487 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.982341 w_k1_ADM_s104
                                 + 0.967033 w_k1_AXP_s104
                                 + 0.946599 w_k1_DIS_s104
                                 + 1.0786 w_k1_FDX_s104
                                 + 0.634532 w_k1_IBM_s104
                                 + 0.991121 w_k1_INTC_s104
                                 + 0.762464 w_k1_KO_s104
                                 + 0.761841 w_k1_MAT_s104
                                 + 0.961542 w_k1_MSFT_s104
                                 + 0.940768 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.822021 w_k1_ADM_s104
                                 + 0.815079 w_k1_AXP_s104
                                 + 0.816423 w_k1_DIS_s104
                                 + 0.806453 w_k1_FDX_s104
                                 + 0.992049 w_k1_IBM_s104
                                 + 0.865422 w_k1_INTC_s104
                                 + 0.870572 w_k1_KO_s104
                                 + 0.552391 w_k1_MAT_s104
                                 + 0.828874 w_k1_MSFT_s104
                                 + 0.894146 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.863852 w_k1_ADM_s104
                                 + 1.00525 w_k1_AXP_s104
                                 + 0.976901 w_k1_DIS_s104
                                 + 0.941423 w_k1_FDX_s104
                                 + 1.77112 w_k1_IBM_s104
                                 + 1.11496 w_k1_INTC_s104
                                 + 1.31184 w_k1_KO_s104 + 1.33271 w_k1_MAT_s104
                                 + 1.04536 w_k1_MSFT_s104
                                 + 1.03168 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.1214 w_k1_ADM_s104
                                 + 1.39658 w_k1_AXP_s104
                                 + 1.49528 w_k1_DIS_s104 + 1.2288 w_k1_FDX_s104
                                 + 1.30125 w_k1_IBM_s104
                                 + 1.09675 w_k1_INTC_s104
                                 + 1.31681 w_k1_KO_s104 + 1.41445 w_k1_MAT_s104
                                 + 1.16475 w_k1_MSFT_s104
                                 + 1.05618 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.681534 w_k1_ADM_s104
                                 + 0.767413 w_k1_AXP_s104
                                 + 0.793947 w_k1_DIS_s104
                                 + 0.699599 w_k1_FDX_s104
                                 + 1.31563 w_k1_IBM_s104
                                 + 0.945161 w_k1_INTC_s104
                                 + 1.1624 w_k1_KO_s104 + 0.954406 w_k1_MAT_s104
                                 + 0.862216 w_k1_MSFT_s104
                                 + 0.863668 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.722957 w_k1_ADM_s105
                                 + 0.726048 w_k1_AXP_s105
                                 + 0.592389 w_k1_DIS_s105
                                 + 0.720128 w_k1_FDX_s105
                                 + 0.644974 w_k1_IBM_s105
                                 + 1.02587 w_k1_INTC_s105
                                 + 0.71986 w_k1_KO_s105
                                 + 0.556334 w_k1_MAT_s105
                                 + 1.00397 w_k1_MSFT_s105
                                 + 0.885145 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.21167 w_k1_ADM_s105
                                 + 1.039 w_k1_AXP_s105 + 0.853295 w_k1_DIS_s105
                                 + 1.00286 w_k1_FDX_s105
                                 + 0.85124 w_k1_IBM_s105
                                 + 1.21716 w_k1_INTC_s105
                                 + 0.87631 w_k1_KO_s105
                                 + 0.902227 w_k1_MAT_s105
                                 + 1.07904 w_k1_MSFT_s105
                                 + 1.08471 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.881132 w_k1_ADM_s105
                                 + 1.11976 w_k1_AXP_s105
                                 + 0.979647 w_k1_DIS_s105
                                 + 0.935133 w_k1_FDX_s105
                                 + 1.09681 w_k1_IBM_s105
                                 + 0.976231 w_k1_INTC_s105
                                 + 1.12994 w_k1_KO_s105 + 1.08966 w_k1_MAT_s105
                                 + 0.976198 w_k1_MSFT_s105
                                 + 1.02252 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.02772 w_k1_ADM_s105
                                 + 1.1467 w_k1_AXP_s105 + 1.23228 w_k1_DIS_s105
                                 + 1.11082 w_k1_FDX_s105 + 1.0277 w_k1_IBM_s105
                                 + 1.31416 w_k1_INTC_s105
                                 + 1.20215 w_k1_KO_s105 + 1.37241 w_k1_MAT_s105
                                 + 1.28796 w_k1_MSFT_s105
                                 + 1.15266 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.22664 w_k1_ADM_s105
                                 + 1.35233 w_k1_AXP_s105 + 1.526 w_k1_DIS_s105
                                 + 1.37241 w_k1_FDX_s105
                                 + 1.26869 w_k1_IBM_s105
                                 + 1.30925 w_k1_INTC_s105
                                 + 1.43999 w_k1_KO_s105 + 1.62395 w_k1_MAT_s105
                                 + 1.40288 w_k1_MSFT_s105
                                 + 1.28146 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.765483 w_k1_ADM_s105
                                 + 0.667841 w_k1_AXP_s105
                                 + 0.647887 w_k1_DIS_s105
                                 + 0.759939 w_k1_FDX_s105
                                 + 0.990568 w_k1_IBM_s105
                                 + 0.972663 w_k1_INTC_s105
                                 + 0.87156 w_k1_KO_s105
                                 + 0.794295 w_k1_MAT_s105
                                 + 0.882792 w_k1_MSFT_s105
                                 + 0.911488 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.975402 w_k1_ADM_s105
                                 + 1.1571 w_k1_AXP_s105
                                 + 0.876705 w_k1_DIS_s105
                                 + 0.97917 w_k1_FDX_s105
                                 + 1.28519 w_k1_IBM_s105
                                 + 1.05755 w_k1_INTC_s105
                                 + 1.30792 w_k1_KO_s105 + 1.69937 w_k1_MAT_s105
                                 + 1.07545 w_k1_MSFT_s105
                                 + 1.19288 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.921432 w_k1_ADM_s105
                                 + 1.01248 w_k1_AXP_s105
                                 + 0.736348 w_k1_DIS_s105
                                 + 1.05212 w_k1_FDX_s105
                                 + 0.712718 w_k1_IBM_s105
                                 + 1.08096 w_k1_INTC_s105
                                 + 0.846827 w_k1_KO_s105
                                 + 0.71771 w_k1_MAT_s105
                                 + 0.948359 w_k1_MSFT_s105
                                 + 0.972985 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.848186 w_k1_ADM_s106
                                 + 1.01671 w_k1_AXP_s106 + 1.1071 w_k1_DIS_s106
                                 + 0.951875 w_k1_FDX_s106
                                 + 1.09891 w_k1_IBM_s106
                                 + 0.975523 w_k1_INTC_s106
                                 + 1.13106 w_k1_KO_s106 + 1.27405 w_k1_MAT_s106
                                 + 0.96017 w_k1_MSFT_s106
                                 + 0.979869 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.731801 w_k1_ADM_s106
                                 + 0.907975 w_k1_AXP_s106
                                 + 0.861623 w_k1_DIS_s106
                                 + 0.779066 w_k1_FDX_s106
                                 + 1.3175 w_k1_IBM_s106
                                 + 0.995469 w_k1_INTC_s106
                                 + 1.01593 w_k1_KO_s106 + 1.1069 w_k1_MAT_s106
                                 + 0.851721 w_k1_MSFT_s106
                                 + 0.955733 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.823843 w_k1_ADM_s106
                                 + 0.629345 w_k1_AXP_s106
                                 + 0.953926 w_k1_DIS_s106
                                 + 0.834345 w_k1_FDX_s106
                                 + 0.780222 w_k1_IBM_s106
                                 + 0.977502 w_k1_INTC_s106
                                 + 0.857823 w_k1_KO_s106
                                 + 0.580812 w_k1_MAT_s106
                                 + 0.983231 w_k1_MSFT_s106
                                 + 0.860471 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.99836 w_k1_ADM_s106
                                 + 1.03452 w_k1_AXP_s106
                                 + 0.679702 w_k1_DIS_s106
                                 + 0.861838 w_k1_FDX_s106
                                 + 0.996992 w_k1_IBM_s106
                                 + 0.974486 w_k1_INTC_s106
                                 + 0.839159 w_k1_KO_s106
                                 + 1.30788 w_k1_MAT_s106
                                 + 0.89142 w_k1_MSFT_s106
                                 + 0.947498 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.16363 w_k1_ADM_s106
                                 + 1.17361 w_k1_AXP_s106
                                 + 1.45829 w_k1_DIS_s106
                                 + 1.19893 w_k1_FDX_s106
                                 + 0.998244 w_k1_IBM_s106
                                 + 1.17247 w_k1_INTC_s106
                                 + 1.13313 w_k1_KO_s106 + 1.02284 w_k1_MAT_s106
                                 + 1.1971 w_k1_MSFT_s106 + 1.0603 w_k1_PG_s106
                                 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.32659 w_k1_ADM_s106
                                 + 1.69117 w_k1_AXP_s106
                                 + 1.59385 w_k1_DIS_s106
                                 + 1.29917 w_k1_FDX_s106
                                 + 1.26809 w_k1_IBM_s106
                                 + 1.30842 w_k1_INTC_s106
                                 + 1.31883 w_k1_KO_s106 + 1.47404 w_k1_MAT_s106
                                 + 1.2956 w_k1_MSFT_s106 + 1.33538 w_k1_PG_s106
                                 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.797901 w_k1_ADM_s106
                                 + 1.30325 w_k1_AXP_s106
                                 + 0.867201 w_k1_DIS_s106
                                 + 1.03731 w_k1_FDX_s106
                                 + 0.893244 w_k1_IBM_s106
                                 + 1.16399 w_k1_INTC_s106
                                 + 0.908625 w_k1_KO_s106
                                 + 1.32347 w_k1_MAT_s106
                                 + 0.935911 w_k1_MSFT_s106
                                 + 1.00902 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.750584 w_k1_ADM_s106
                                 + 0.820188 w_k1_AXP_s106
                                 + 0.814799 w_k1_DIS_s106
                                 + 0.922646 w_k1_FDX_s106
                                 + 0.607482 w_k1_IBM_s106
                                 + 1.00237 w_k1_INTC_s106
                                 + 0.679944 w_k1_KO_s106
                                 + 0.514831 w_k1_MAT_s106
                                 + 0.937935 w_k1_MSFT_s106
                                 + 0.831862 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.828455 w_k1_ADM_s107
                                 + 0.869708 w_k1_AXP_s107
                                 + 1.35494 w_k1_DIS_s107
                                 + 1.07353 w_k1_FDX_s107
                                 + 0.994977 w_k1_IBM_s107
                                 + 1.10164 w_k1_INTC_s107
                                 + 1.23508 w_k1_KO_s107 + 0.90566 w_k1_MAT_s107
                                 + 1.18708 w_k1_MSFT_s107
                                 + 1.08344 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.740744 w_k1_ADM_s107
                                 + 1.07521 w_k1_AXP_s107
                                 + 0.906247 w_k1_DIS_s107
                                 + 0.905344 w_k1_FDX_s107
                                 + 1.1564 w_k1_IBM_s107
                                 + 1.02946 w_k1_INTC_s107 + 0.9294 w_k1_KO_s107
                                 + 1.43956 w_k1_MAT_s107
                                 + 1.04996 w_k1_MSFT_s107
                                 + 1.05745 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.13146 w_k1_ADM_s107
                                 + 1.11008 w_k1_AXP_s107
                                 + 1.43946 w_k1_DIS_s107
                                 + 1.14193 w_k1_FDX_s107
                                 + 1.32441 w_k1_IBM_s107
                                 + 1.13192 w_k1_INTC_s107
                                 + 1.29628 w_k1_KO_s107 + 1.38761 w_k1_MAT_s107
                                 + 1.05941 w_k1_MSFT_s107
                                 + 1.09193 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.829458 w_k1_ADM_s107
                                 + 1.02551 w_k1_AXP_s107
                                 + 0.790034 w_k1_DIS_s107
                                 + 0.850457 w_k1_FDX_s107
                                 + 1.73135 w_k1_IBM_s107
                                 + 0.886148 w_k1_INTC_s107
                                 + 1.11945 w_k1_KO_s107 + 1.53188 w_k1_MAT_s107
                                 + 0.851384 w_k1_MSFT_s107
                                 + 0.972472 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.866858 w_k1_ADM_s107
                                 + 0.847062 w_k1_AXP_s107
                                 + 0.890952 w_k1_DIS_s107
                                 + 0.91821 w_k1_FDX_s107
                                 + 0.981123 w_k1_IBM_s107
                                 + 0.750013 w_k1_INTC_s107
                                 + 1.03566 w_k1_KO_s107 + 1.06716 w_k1_MAT_s107
                                 + 0.893895 w_k1_MSFT_s107
                                 + 0.885306 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.874304 w_k1_ADM_s107
                                 + 0.787104 w_k1_AXP_s107
                                 + 0.898118 w_k1_DIS_s107
                                 + 1.00028 w_k1_FDX_s107
                                 + 0.643476 w_k1_IBM_s107
                                 + 0.829966 w_k1_INTC_s107
                                 + 0.823778 w_k1_KO_s107
                                 + 0.537633 w_k1_MAT_s107
                                 + 0.920864 w_k1_MSFT_s107
                                 + 0.838001 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.17197 w_k1_ADM_s107
                                 + 1.0965 w_k1_AXP_s107 + 1.65294 w_k1_DIS_s107
                                 + 1.41709 w_k1_FDX_s107 + 1.0154 w_k1_IBM_s107
                                 + 1.1642 w_k1_INTC_s107 + 1.22074 w_k1_KO_s107
                                 + 1.01335 w_k1_MAT_s107
                                 + 1.29447 w_k1_MSFT_s107
                                 + 1.10039 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.887719 w_k1_ADM_s107
                                 + 0.704959 w_k1_AXP_s107
                                 + 1.09025 w_k1_DIS_s107
                                 + 0.829576 w_k1_FDX_s107
                                 + 1.47824 w_k1_IBM_s107
                                 + 0.822591 w_k1_INTC_s107
                                 + 1.35616 w_k1_KO_s107 + 1.04266 w_k1_MAT_s107
                                 + 0.96059 w_k1_MSFT_s107
                                 + 0.998927 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.13804 w_k1_ADM_s110
                                 + 1.44931 w_k1_AXP_s110
                                 + 1.11767 w_k1_DIS_s110
                                 + 1.23292 w_k1_FDX_s110
                                 + 1.11428 w_k1_IBM_s110
                                 + 1.21782 w_k1_INTC_s110
                                 + 1.00923 w_k1_KO_s110 + 1.3124 w_k1_MAT_s110
                                 + 1.07848 w_k1_MSFT_s110
                                 + 1.00877 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.853614 w_k1_ADM_s110
                                 + 0.904608 w_k1_AXP_s110
                                 + 0.877459 w_k1_DIS_s110
                                 + 0.850874 w_k1_FDX_s110
                                 + 1.21864 w_k1_IBM_s110
                                 + 0.958326 w_k1_INTC_s110
                                 + 1.02185 w_k1_KO_s110 + 1.8038 w_k1_MAT_s110
                                 + 1.05185 w_k1_MSFT_s110
                                 + 0.940841 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.978425 w_k1_ADM_s110
                                 + 0.961322 w_k1_AXP_s110
                                 + 1.16669 w_k1_DIS_s110
                                 + 1.11769 w_k1_FDX_s110
                                 + 1.01083 w_k1_IBM_s110
                                 + 1.18994 w_k1_INTC_s110
                                 + 1.14435 w_k1_KO_s110 + 1.30764 w_k1_MAT_s110
                                 + 1.36064 w_k1_MSFT_s110
                                 + 1.06189 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.06861 w_k1_ADM_s110
                                 + 1.04322 w_k1_AXP_s110
                                 + 1.19806 w_k1_DIS_s110
                                 + 1.26411 w_k1_FDX_s110
                                 + 0.716437 w_k1_IBM_s110
                                 + 1.17121 w_k1_INTC_s110
                                 + 0.913283 w_k1_KO_s110
                                 + 0.569246 w_k1_MAT_s110
                                 + 1.14088 w_k1_MSFT_s110
                                 + 0.939726 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.863326 w_k1_ADM_s110
                                 + 1.07285 w_k1_AXP_s110
                                 + 1.17861 w_k1_DIS_s110
                                 + 1.05352 w_k1_FDX_s110
                                 + 1.32924 w_k1_IBM_s110
                                 + 1.03908 w_k1_INTC_s110
                                 + 1.12448 w_k1_KO_s110 + 1.38013 w_k1_MAT_s110
                                 + 1.0239 w_k1_MSFT_s110
                                 + 0.883372 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.7665 w_k1_ADM_s110
                                 + 0.723797 w_k1_AXP_s110
                                 + 0.738608 w_k1_DIS_s110
                                 + 0.865358 w_k1_FDX_s110
                                 + 0.831418 w_k1_IBM_s110
                                 + 1.10734 w_k1_INTC_s110
                                 + 0.765268 w_k1_KO_s110
                                 + 0.686258 w_k1_MAT_s110
                                 + 0.900778 w_k1_MSFT_s110
                                 + 0.868718 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.99386 w_k1_ADM_s110
                                 + 1.66108 w_k1_AXP_s110
                                 + 1.76341 w_k1_DIS_s110
                                 + 1.29899 w_k1_FDX_s110
                                 + 1.56441 w_k1_IBM_s110
                                 + 1.22767 w_k1_INTC_s110
                                 + 1.34234 w_k1_KO_s110 + 1.96333 w_k1_MAT_s110
                                 + 1.32251 w_k1_MSFT_s110
                                 + 1.08238 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.04369 w_k1_ADM_s110
                                 + 0.998801 w_k1_AXP_s110
                                 + 0.968876 w_k1_DIS_s110
                                 + 1.07978 w_k1_FDX_s110
                                 + 1.05732 w_k1_IBM_s110
                                 + 1.08548 w_k1_INTC_s110
                                 + 1.04219 w_k1_KO_s110 + 1.00122 w_k1_MAT_s110
                                 + 1.01703 w_k1_MSFT_s110
                                 + 0.988047 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.887472 w_k1_ADM_s111
                                 + 0.748275 w_k1_AXP_s111
                                 + 0.881876 w_k1_DIS_s111
                                 + 0.874938 w_k1_FDX_s111
                                 + 1.11172 w_k1_IBM_s111
                                 + 0.845274 w_k1_INTC_s111
                                 + 0.971085 w_k1_KO_s111
                                 + 0.96602 w_k1_MAT_s111
                                 + 0.978592 w_k1_MSFT_s111
                                 + 0.953974 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.602198 w_k1_ADM_s111
                                 + 0.684934 w_k1_AXP_s111
                                 + 0.650519 w_k1_DIS_s111
                                 + 0.717336 w_k1_FDX_s111
                                 + 0.733747 w_k1_IBM_s111
                                 + 0.828819 w_k1_INTC_s111
                                 + 0.637269 w_k1_KO_s111
                                 + 0.573095 w_k1_MAT_s111
                                 + 0.738096 w_k1_MSFT_s111
                                 + 0.752658 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.6908 w_k1_ADM_s111
                                 + 1.38293 w_k1_AXP_s111
                                 + 1.98954 w_k1_DIS_s111
                                 + 1.39151 w_k1_FDX_s111
                                 + 1.27439 w_k1_IBM_s111
                                 + 1.50849 w_k1_INTC_s111 + 1.5274 w_k1_KO_s111
                                 + 1.20675 w_k1_MAT_s111
                                 + 1.78382 w_k1_MSFT_s111
                                 + 1.36588 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.09245 w_k1_ADM_s111
                                 + 1.60365 w_k1_AXP_s111
                                 + 1.57428 w_k1_DIS_s111 + 1.1783 w_k1_FDX_s111
                                 + 1.41102 w_k1_IBM_s111 + 1.427 w_k1_INTC_s111
                                 + 1.22399 w_k1_KO_s111 + 1.32255 w_k1_MAT_s111
                                 + 1.1979 w_k1_MSFT_s111 + 1.10537 w_k1_PG_s111
                                 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.896491 w_k1_ADM_s111
                                 + 1.23154 w_k1_AXP_s111
                                 + 1.08572 w_k1_DIS_s111
                                 + 1.01671 w_k1_FDX_s111
                                 + 0.866765 w_k1_IBM_s111
                                 + 0.956539 w_k1_INTC_s111
                                 + 0.891121 w_k1_KO_s111
                                 + 1.02975 w_k1_MAT_s111
                                 + 1.07768 w_k1_MSFT_s111
                                 + 1.03196 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.944082 w_k1_ADM_s111
                                 + 0.667716 w_k1_AXP_s111
                                 + 0.933801 w_k1_DIS_s111
                                 + 0.894148 w_k1_FDX_s111
                                 + 0.864399 w_k1_IBM_s111
                                 + 0.921123 w_k1_INTC_s111
                                 + 0.956181 w_k1_KO_s111
                                 + 0.722454 w_k1_MAT_s111
                                 + 1.03626 w_k1_MSFT_s111 + 1.0372 w_k1_PG_s111
                                 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.734879 w_k1_ADM_s111
                                 + 1.13732 w_k1_AXP_s111
                                 + 1.13086 w_k1_DIS_s111
                                 + 0.933911 w_k1_FDX_s111
                                 + 1.43014 w_k1_IBM_s111
                                 + 1.00072 w_k1_INTC_s111
                                 + 1.08981 w_k1_KO_s111 + 1.05289 w_k1_MAT_s111
                                 + 1.08951 w_k1_MSFT_s111
                                 + 1.11405 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.0575 w_k1_ADM_s111
                                 + 0.8901 w_k1_AXP_s111 + 1.27835 w_k1_DIS_s111
                                 + 1.08898 w_k1_FDX_s111
                                 + 0.876204 w_k1_IBM_s111
                                 + 1.05551 w_k1_INTC_s111
                                 + 1.12996 w_k1_KO_s111
                                 + 0.621294 w_k1_MAT_s111
                                 + 1.13983 w_k1_MSFT_s111
                                 + 1.03373 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.919382 w_k1_ADM_s112
                                 + 1.15901 w_k1_AXP_s112
                                 + 1.29761 w_k1_DIS_s112 + 1.2656 w_k1_FDX_s112
                                 + 0.840305 w_k1_IBM_s112
                                 + 1.33611 w_k1_INTC_s112 + 1.1274 w_k1_KO_s112
                                 + 0.623921 w_k1_MAT_s112
                                 + 1.16007 w_k1_MSFT_s112 + 1.2247 w_k1_PG_s112
                                 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.711147 w_k1_ADM_s112
                                 + 0.573968 w_k1_AXP_s112
                                 + 0.836006 w_k1_DIS_s112
                                 + 0.816946 w_k1_FDX_s112
                                 + 0.795568 w_k1_IBM_s112
                                 + 0.825906 w_k1_INTC_s112
                                 + 0.785189 w_k1_KO_s112
                                 + 0.526731 w_k1_MAT_s112
                                 + 0.841621 w_k1_MSFT_s112
                                 + 0.900894 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.03041 w_k1_ADM_s112
                                 + 0.846582 w_k1_AXP_s112
                                 + 0.879105 w_k1_DIS_s112
                                 + 1.06018 w_k1_FDX_s112
                                 + 0.71947 w_k1_IBM_s112
                                 + 1.1349 w_k1_INTC_s112
                                 + 0.742109 w_k1_KO_s112
                                 + 0.421409 w_k1_MAT_s112
                                 + 0.88363 w_k1_MSFT_s112
                                 + 0.970344 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.877114 w_k1_ADM_s112
                                 + 0.754294 w_k1_AXP_s112
                                 + 0.527795 w_k1_DIS_s112
                                 + 0.700499 w_k1_FDX_s112
                                 + 0.833733 w_k1_IBM_s112
                                 + 0.630766 w_k1_INTC_s112
                                 + 0.687619 w_k1_KO_s112
                                 + 0.51906 w_k1_MAT_s112
                                 + 0.563357 w_k1_MSFT_s112
                                 + 0.780016 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.20703 w_k1_ADM_s112
                                 + 1.47079 w_k1_AXP_s112
                                 + 1.52337 w_k1_DIS_s112
                                 + 1.29683 w_k1_FDX_s112
                                 + 1.43704 w_k1_IBM_s112
                                 + 1.46113 w_k1_INTC_s112
                                 + 1.43611 w_k1_KO_s112 + 1.21715 w_k1_MAT_s112
                                 + 1.19712 w_k1_MSFT_s112
                                 + 1.18241 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.712937 w_k1_ADM_s112
                                 + 0.822934 w_k1_AXP_s112
                                 + 0.774118 w_k1_DIS_s112
                                 + 0.910475 w_k1_FDX_s112
                                 + 0.793488 w_k1_IBM_s112
                                 + 0.999723 w_k1_INTC_s112
                                 + 0.73283 w_k1_KO_s112
                                 + 0.640931 w_k1_MAT_s112
                                 + 0.843264 w_k1_MSFT_s112
                                 + 0.898703 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.856533 w_k1_ADM_s112
                                 + 1.10566 w_k1_AXP_s112
                                 + 1.11331 w_k1_DIS_s112
                                 + 1.08552 w_k1_FDX_s112
                                 + 1.25936 w_k1_IBM_s112
                                 + 1.1351 w_k1_INTC_s112 + 1.10058 w_k1_KO_s112
                                 + 0.796265 w_k1_MAT_s112
                                 + 0.945374 w_k1_MSFT_s112
                                 + 1.0656 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.978036 w_k1_ADM_s112
                                 + 0.976294 w_k1_AXP_s112
                                 + 0.950701 w_k1_DIS_s112
                                 + 0.942337 w_k1_FDX_s112
                                 + 1.04583 w_k1_IBM_s112
                                 + 0.992298 w_k1_INTC_s112
                                 + 1.01692 w_k1_KO_s112 + 0.85134 w_k1_MAT_s112
                                 + 0.980503 w_k1_MSFT_s112
                                 + 1.09285 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.19331 w_k1_ADM_s113
                                 + 1.32008 w_k1_AXP_s113
                                 + 1.41552 w_k1_DIS_s113 + 1.2644 w_k1_FDX_s113
                                 + 1.27212 w_k1_IBM_s113
                                 + 1.10796 w_k1_INTC_s113
                                 + 1.26549 w_k1_KO_s113 + 1.10362 w_k1_MAT_s113
                                 + 1.11128 w_k1_MSFT_s113
                                 + 1.22985 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.06368 w_k1_ADM_s113
                                 + 1.4978 w_k1_AXP_s113 + 1.18576 w_k1_DIS_s113
                                 + 1.13246 w_k1_FDX_s113
                                 + 1.47458 w_k1_IBM_s113
                                 + 0.95331 w_k1_INTC_s113
                                 + 1.43234 w_k1_KO_s113 + 1.61625 w_k1_MAT_s113
                                 + 0.986304 w_k1_MSFT_s113
                                 + 1.24825 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.09177 w_k1_ADM_s113
                                 + 1.00377 w_k1_AXP_s113
                                 + 1.15256 w_k1_DIS_s113
                                 + 1.13756 w_k1_FDX_s113
                                 + 0.90225 w_k1_IBM_s113
                                 + 1.12211 w_k1_INTC_s113 + 1.0753 w_k1_KO_s113
                                 + 0.783917 w_k1_MAT_s113
                                 + 1.04292 w_k1_MSFT_s113
                                 + 1.14484 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.854858 w_k1_ADM_s113
                                 + 0.586919 w_k1_AXP_s113
                                 + 0.682835 w_k1_DIS_s113
                                 + 0.716962 w_k1_FDX_s113
                                 + 0.791635 w_k1_IBM_s113
                                 + 0.830887 w_k1_INTC_s113
                                 + 0.847812 w_k1_KO_s113
                                 + 0.375345 w_k1_MAT_s113
                                 + 0.822588 w_k1_MSFT_s113
                                 + 0.853393 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.15078 w_k1_ADM_s113
                                 + 1.00231 w_k1_AXP_s113
                                 + 0.819738 w_k1_DIS_s113
                                 + 0.933235 w_k1_FDX_s113
                                 + 1.19569 w_k1_IBM_s113
                                 + 0.771509 w_k1_INTC_s113
                                 + 1.01766 w_k1_KO_s113 + 1.01446 w_k1_MAT_s113
                                 + 0.818578 w_k1_MSFT_s113
                                 + 1.02987 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.948921 w_k1_ADM_s113
                                 + 1.13857 w_k1_AXP_s113
                                 + 0.770481 w_k1_DIS_s113
                                 + 0.907276 w_k1_FDX_s113
                                 + 0.874464 w_k1_IBM_s113
                                 + 0.944772 w_k1_INTC_s113
                                 + 0.793058 w_k1_KO_s113
                                 + 0.727142 w_k1_MAT_s113
                                 + 0.890021 w_k1_MSFT_s113
                                 + 0.951701 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.845639 w_k1_ADM_s113
                                 + 0.98042 w_k1_AXP_s113
                                 + 0.983207 w_k1_DIS_s113
                                 + 0.897976 w_k1_FDX_s113
                                 + 1.0746 w_k1_IBM_s113
                                 + 1.02064 w_k1_INTC_s113
                                 + 1.03816 w_k1_KO_s113 + 0.83988 w_k1_MAT_s113
                                 + 1.00207 w_k1_MSFT_s113
                                 + 0.902059 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.96413 w_k1_ADM_s113
                                 + 0.76981 w_k1_AXP_s113
                                 + 0.689834 w_k1_DIS_s113
                                 + 0.815011 w_k1_FDX_s113
                                 + 0.661528 w_k1_IBM_s113
                                 + 0.802222 w_k1_INTC_s113
                                 + 0.783468 w_k1_KO_s113
                                 + 0.434422 w_k1_MAT_s113
                                 + 0.85614 w_k1_MSFT_s113
                                 + 0.946366 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.29482 w_k1_ADM_s114
                                 + 1.75476 w_k1_AXP_s114
                                 + 1.62886 w_k1_DIS_s114
                                 + 1.26794 w_k1_FDX_s114
                                 + 1.56782 w_k1_IBM_s114
                                 + 1.03223 w_k1_INTC_s114
                                 + 1.54529 w_k1_KO_s114 + 1.95382 w_k1_MAT_s114
                                 + 1.40049 w_k1_MSFT_s114
                                 + 1.32147 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.871702 w_k1_ADM_s114
                                 + 0.736308 w_k1_AXP_s114
                                 + 0.666487 w_k1_DIS_s114
                                 + 0.704841 w_k1_FDX_s114
                                 + 0.945881 w_k1_IBM_s114
                                 + 0.692021 w_k1_INTC_s114
                                 + 0.963481 w_k1_KO_s114
                                 + 0.726406 w_k1_MAT_s114
                                 + 0.856664 w_k1_MSFT_s114
                                 + 0.932413 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.21454 w_k1_ADM_s114
                                 + 1.17074 w_k1_AXP_s114
                                 + 0.844949 w_k1_DIS_s114
                                 + 1.03455 w_k1_FDX_s114
                                 + 0.693751 w_k1_IBM_s114
                                 + 0.891915 w_k1_INTC_s114
                                 + 0.718391 w_k1_KO_s114
                                 + 0.566359 w_k1_MAT_s114
                                 + 0.913323 w_k1_MSFT_s114
                                 + 0.814252 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.895922 w_k1_ADM_s114
                                 + 1.04832 w_k1_AXP_s114
                                 + 1.15428 w_k1_DIS_s114
                                 + 0.943335 w_k1_FDX_s114
                                 + 1.19462 w_k1_IBM_s114
                                 + 0.945364 w_k1_INTC_s114
                                 + 1.20943 w_k1_KO_s114
                                 + 0.993112 w_k1_MAT_s114
                                 + 1.0505 w_k1_MSFT_s114 + 1.01492 w_k1_PG_s114
                                 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.2124 w_k1_ADM_s114
                                 + 1.20465 w_k1_AXP_s114
                                 + 1.32421 w_k1_DIS_s114
                                 + 1.26125 w_k1_FDX_s114
                                 + 0.990197 w_k1_IBM_s114
                                 + 1.3163 w_k1_INTC_s114 + 1.10336 w_k1_KO_s114
                                 + 0.756884 w_k1_MAT_s114
                                 + 1.18563 w_k1_MSFT_s114
                                 + 1.20742 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.940682 w_k1_ADM_s114
                                 + 0.896165 w_k1_AXP_s114
                                 + 1.11629 w_k1_DIS_s114
                                 + 1.03066 w_k1_FDX_s114
                                 + 1.02947 w_k1_IBM_s114
                                 + 0.947181 w_k1_INTC_s114
                                 + 1.11107 w_k1_KO_s114 + 1.17475 w_k1_MAT_s114
                                 + 1.17523 w_k1_MSFT_s114
                                 + 1.17331 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.99228 w_k1_ADM_s114
                                 + 1.10309 w_k1_AXP_s114
                                 + 0.855041 w_k1_DIS_s114
                                 + 0.90586 w_k1_FDX_s114
                                 + 1.26604 w_k1_IBM_s114
                                 + 0.854523 w_k1_INTC_s114
                                 + 1.00623 w_k1_KO_s114 + 1.04114 w_k1_MAT_s114
                                 + 0.925259 w_k1_MSFT_s114
                                 + 1.14972 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.86378 w_k1_ADM_s114
                                 + 0.871634 w_k1_AXP_s114
                                 + 0.817873 w_k1_DIS_s114
                                 + 0.911843 w_k1_FDX_s114
                                 + 0.779063 w_k1_IBM_s114
                                 + 0.89163 w_k1_INTC_s114
                                 + 0.827603 w_k1_KO_s114
                                 + 0.971813 w_k1_MAT_s114
                                 + 1.04903 w_k1_MSFT_s114
                                 + 1.00955 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.26848 w_k1_ADM_s115
                                 + 0.966777 w_k1_AXP_s115
                                 + 1.10818 w_k1_DIS_s115
                                 + 1.08882 w_k1_FDX_s115
                                 + 0.89919 w_k1_IBM_s115
                                 + 0.983112 w_k1_INTC_s115
                                 + 1.02829 w_k1_KO_s115
                                 + 0.806692 w_k1_MAT_s115
                                 + 1.00525 w_k1_MSFT_s115
                                 + 0.956381 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.821734 w_k1_ADM_s115
                                 + 0.773476 w_k1_AXP_s115
                                 + 0.834758 w_k1_DIS_s115
                                 + 0.92184 w_k1_FDX_s115
                                 + 0.645773 w_k1_IBM_s115
                                 + 0.864159 w_k1_INTC_s115
                                 + 0.796453 w_k1_KO_s115
                                 + 0.908313 w_k1_MAT_s115
                                 + 0.83244 w_k1_MSFT_s115
                                 + 0.890604 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.12717 w_k1_ADM_s115
                                 + 0.985231 w_k1_AXP_s115
                                 + 0.987384 w_k1_DIS_s115
                                 + 0.977828 w_k1_FDX_s115
                                 + 1.35897 w_k1_IBM_s115
                                 + 0.925271 w_k1_INTC_s115
                                 + 1.08447 w_k1_KO_s115 + 1.18415 w_k1_MAT_s115
                                 + 0.984509 w_k1_MSFT_s115
                                 + 1.01471 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.38183 w_k1_ADM_s115
                                 + 1.02459 w_k1_AXP_s115
                                 + 1.56762 w_k1_DIS_s115
                                 + 1.28967 w_k1_FDX_s115
                                 + 1.04599 w_k1_IBM_s115 + 1.072 w_k1_INTC_s115
                                 + 1.17542 w_k1_KO_s115
                                 + 0.874155 w_k1_MAT_s115
                                 + 1.24973 w_k1_MSFT_s115
                                 + 1.19682 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.690764 w_k1_ADM_s115
                                 + 0.598349 w_k1_AXP_s115
                                 + 0.670463 w_k1_DIS_s115
                                 + 0.735926 w_k1_FDX_s115
                                 + 0.962086 w_k1_IBM_s115
                                 + 0.659868 w_k1_INTC_s115
                                 + 0.835749 w_k1_KO_s115
                                 + 0.725453 w_k1_MAT_s115
                                 + 0.727536 w_k1_MSFT_s115
                                 + 0.777864 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.29969 w_k1_ADM_s115
                                 + 1.61957 w_k1_AXP_s115
                                 + 1.39453 w_k1_DIS_s115
                                 + 1.29352 w_k1_FDX_s115
                                 + 1.23406 w_k1_IBM_s115
                                 + 1.12057 w_k1_INTC_s115
                                 + 1.14589 w_k1_KO_s115 + 1.36026 w_k1_MAT_s115
                                 + 0.976695 w_k1_MSFT_s115
                                 + 1.19344 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.19076 w_k1_ADM_s115
                                 + 0.92077 w_k1_AXP_s115 + 1.2052 w_k1_DIS_s115
                                 + 0.963623 w_k1_FDX_s115
                                 + 1.24006 w_k1_IBM_s115
                                 + 0.85153 w_k1_INTC_s115
                                 + 0.979915 w_k1_KO_s115
                                 + 0.848179 w_k1_MAT_s115
                                 + 0.805813 w_k1_MSFT_s115
                                 + 1.02325 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.815416 w_k1_ADM_s115
                                 + 0.82307 w_k1_AXP_s115
                                 + 0.923315 w_k1_DIS_s115
                                 + 0.85882 w_k1_FDX_s115
                                 + 0.990643 w_k1_IBM_s115
                                 + 0.675654 w_k1_INTC_s115
                                 + 0.830638 w_k1_KO_s115
                                 + 1.03471 w_k1_MAT_s115
                                 + 0.825105 w_k1_MSFT_s115
                                 + 0.906805 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.701191 w_k1_ADM_s116
                                 + 0.902558 w_k1_AXP_s116
                                 + 1.04545 w_k1_DIS_s116
                                 + 0.884891 w_k1_FDX_s116
                                 + 1.37451 w_k1_IBM_s116
                                 + 0.978322 w_k1_INTC_s116
                                 + 1.08159 w_k1_KO_s116 + 1.13918 w_k1_MAT_s116
                                 + 1.01698 w_k1_MSFT_s116 + 1.0386 w_k1_PG_s116
                                 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.06316 w_k1_ADM_s116
                                 + 1.5835 w_k1_AXP_s116 + 1.44969 w_k1_DIS_s116
                                 + 1.43313 w_k1_FDX_s116
                                 + 1.08396 w_k1_IBM_s116
                                 + 1.37504 w_k1_INTC_s116
                                 + 1.02174 w_k1_KO_s116 + 1.33088 w_k1_MAT_s116
                                 + 1.22402 w_k1_MSFT_s116
                                 + 1.15538 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.645875 w_k1_ADM_s116
                                 + 0.845103 w_k1_AXP_s116
                                 + 0.586656 w_k1_DIS_s116
                                 + 0.814728 w_k1_FDX_s116
                                 + 1.02302 w_k1_IBM_s116
                                 + 0.835949 w_k1_INTC_s116
                                 + 0.699564 w_k1_KO_s116
                                 + 0.953146 w_k1_MAT_s116
                                 + 0.76755 w_k1_MSFT_s116
                                 + 0.883058 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.867218 w_k1_ADM_s116
                                 + 0.949071 w_k1_AXP_s116
                                 + 0.90971 w_k1_DIS_s116
                                 + 1.03268 w_k1_FDX_s116
                                 + 0.821785 w_k1_IBM_s116
                                 + 0.96805 w_k1_INTC_s116
                                 + 0.827704 w_k1_KO_s116
                                 + 0.929747 w_k1_MAT_s116
                                 + 1.20091 w_k1_MSFT_s116
                                 + 1.05671 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.966567 w_k1_ADM_s116
                                 + 0.941691 w_k1_AXP_s116
                                 + 0.861035 w_k1_DIS_s116
                                 + 0.9265 w_k1_FDX_s116
                                 + 0.860838 w_k1_IBM_s116
                                 + 0.813621 w_k1_INTC_s116
                                 + 0.777647 w_k1_KO_s116
                                 + 0.835676 w_k1_MAT_s116
                                 + 0.899328 w_k1_MSFT_s116
                                 + 0.9917 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.746268 w_k1_ADM_s116
                                 + 0.555819 w_k1_AXP_s116
                                 + 0.730494 w_k1_DIS_s116
                                 + 0.823641 w_k1_FDX_s116
                                 + 0.695515 w_k1_IBM_s116
                                 + 0.764376 w_k1_INTC_s116
                                 + 0.826748 w_k1_KO_s116
                                 + 0.556388 w_k1_MAT_s116
                                 + 0.955891 w_k1_MSFT_s116
                                 + 0.824555 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.10367 w_k1_ADM_s116
                                 + 0.941149 w_k1_AXP_s116
                                 + 1.10277 w_k1_DIS_s116 + 1.2095 w_k1_FDX_s116
                                 + 0.657785 w_k1_IBM_s116
                                 + 1.02444 w_k1_INTC_s116
                                 + 0.811964 w_k1_KO_s116
                                 + 0.776303 w_k1_MAT_s116
                                 + 1.34571 w_k1_MSFT_s116
                                 + 1.17284 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.05609 w_k1_ADM_s116
                                 + 1.25519 w_k1_AXP_s116
                                 + 1.23196 w_k1_DIS_s116
                                 + 1.09045 w_k1_FDX_s116 + 1.4068 w_k1_IBM_s116
                                 + 0.904805 w_k1_INTC_s116
                                 + 1.22394 w_k1_KO_s116 + 1.67114 w_k1_MAT_s116
                                 + 0.97721 w_k1_MSFT_s116
                                 + 1.11815 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.883988 w_k1_ADM_s117
                                 + 0.800378 w_k1_AXP_s117
                                 + 0.95418 w_k1_DIS_s117
                                 + 0.864916 w_k1_FDX_s117
                                 + 1.22038 w_k1_IBM_s117
                                 + 1.02928 w_k1_INTC_s117
                                 + 1.27033 w_k1_KO_s117
                                 + 0.902187 w_k1_MAT_s117
                                 + 0.898023 w_k1_MSFT_s117
                                 + 1.02773 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.15182 w_k1_ADM_s117
                                 + 0.959844 w_k1_AXP_s117
                                 + 0.968867 w_k1_DIS_s117
                                 + 0.980971 w_k1_FDX_s117
                                 + 1.06486 w_k1_IBM_s117
                                 + 0.969958 w_k1_INTC_s117
                                 + 1.03024 w_k1_KO_s117
                                 + 0.801854 w_k1_MAT_s117
                                 + 0.828855 w_k1_MSFT_s117
                                 + 0.997371 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.29858 w_k1_ADM_s117
                                 + 1.34673 w_k1_AXP_s117
                                 + 1.32712 w_k1_DIS_s117
                                 + 1.12536 w_k1_FDX_s117
                                 + 1.66345 w_k1_IBM_s117
                                 + 1.1394 w_k1_INTC_s117 + 1.68874 w_k1_KO_s117
                                 + 1.53236 w_k1_MAT_s117
                                 + 1.0573 w_k1_MSFT_s117 + 1.11503 w_k1_PG_s117
                                 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.38204 w_k1_ADM_s117
                                 + 1.08637 w_k1_AXP_s117
                                 + 1.38985 w_k1_DIS_s117
                                 + 1.33181 w_k1_FDX_s117
                                 + 0.922927 w_k1_IBM_s117
                                 + 1.29907 w_k1_INTC_s117
                                 + 1.33744 w_k1_KO_s117 + 1.14812 w_k1_MAT_s117
                                 + 1.33188 w_k1_MSFT_s117
                                 + 1.19861 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.746028 w_k1_ADM_s117
                                 + 0.750852 w_k1_AXP_s117
                                 + 0.708796 w_k1_DIS_s117
                                 + 0.731381 w_k1_FDX_s117
                                 + 0.992357 w_k1_IBM_s117
                                 + 0.747246 w_k1_INTC_s117
                                 + 0.97274 w_k1_KO_s117
                                 + 0.893267 w_k1_MAT_s117
                                 + 0.755717 w_k1_MSFT_s117
                                 + 0.933456 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.11597 w_k1_ADM_s117
                                 + 1.02678 w_k1_AXP_s117
                                 + 1.54585 w_k1_DIS_s117
                                 + 1.19584 w_k1_FDX_s117
                                 + 1.21551 w_k1_IBM_s117
                                 + 1.29808 w_k1_INTC_s117
                                 + 1.49531 w_k1_KO_s117 + 1.32718 w_k1_MAT_s117
                                 + 1.3543 w_k1_MSFT_s117 + 1.15692 w_k1_PG_s117
                                 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.06968 w_k1_ADM_s117
                                 + 0.847905 w_k1_AXP_s117
                                 + 1.24473 w_k1_DIS_s117
                                 + 0.97389 w_k1_FDX_s117
                                 + 1.26065 w_k1_IBM_s117 + 1.185 w_k1_INTC_s117
                                 + 1.48789 w_k1_KO_s117 + 1.37783 w_k1_MAT_s117
                                 + 1.13744 w_k1_MSFT_s117
                                 + 1.09931 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.934769 w_k1_ADM_s117
                                 + 0.844861 w_k1_AXP_s117
                                 + 0.871943 w_k1_DIS_s117
                                 + 0.944695 w_k1_FDX_s117
                                 + 0.673296 w_k1_IBM_s117
                                 + 1.03124 w_k1_INTC_s117
                                 + 0.85232 w_k1_KO_s117
                                 + 0.716263 w_k1_MAT_s117
                                 + 0.932521 w_k1_MSFT_s117
                                 + 0.848235 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0267841876651408 w_k1_ADM_s000
                                 + 0.0099558893926642 w_k1_ADM_s001
                                 + 0.0171774429390353 w_k1_ADM_s002
                                 + 0.00916919583786952 w_k1_ADM_s003
                                 + 0.0257133206026034 w_k1_ADM_s004
                                 + 0.0243646071964733 w_k1_ADM_s005
                                 + 0.00996911125800399 w_k1_ADM_s006
                                 + 0.0112672690145002 w_k1_ADM_s007
                                 + 0.0133849744359038 w_k1_ADM_s010
                                 + 0.0194416696073586 w_k1_ADM_s011
                                 + 0.00861220009320676 w_k1_ADM_s012
                                 + 0.00617240830805329 w_k1_ADM_s013
                                 + 0.00762642342160097 w_k1_ADM_s014
                                 + 0.0109398220802983 w_k1_ADM_s015
                                 + 0.010227640010618 w_k1_ADM_s016
                                 + 0.00670866064937726 w_k1_ADM_s017
                                 + 0.116094092762507 w_k1_ADM_s100
                                 + 0.151258062652357 w_k1_ADM_s101
                                 + 0.0524853560039307 w_k1_ADM_s102
                                 + 0.0802880980387839 w_k1_ADM_s103
                                 + 0.0378796804353986 w_k1_ADM_s104
                                 + 0.0290495807573068 w_k1_ADM_s105
                                 + 0.131173426454283 w_k1_ADM_s106
                                 + 0.0337686947714894 w_k1_ADM_s107
                                 + 0.0157269936093261 w_k1_ADM_s110
                                 + 0.00992150668768248 w_k1_ADM_s111
                                 + 0.0194427720044968 w_k1_ADM_s112
                                 + 0.0114550308994881 w_k1_ADM_s113
                                 + 0.02012939940145 w_k1_ADM_s114
                                 + 0.0136176942791036 w_k1_ADM_s115
                                 + 0.0371227542704347 w_k1_ADM_s116
                                 + 0.027424331357872 w_k1_ADM_s117
                                 + 0.028358375283156 w_k1_AXP_s000
                                 + 0.0113173185296688 w_k1_AXP_s001
                                 + 0.0154241181958757 w_k1_AXP_s002
                                 + 0.00814802693822658 w_k1_AXP_s003
                                 + 0.0275273007059464 w_k1_AXP_s004
                                 + 0.0274763353409309 w_k1_AXP_s005
                                 + 0.00970521762186816 w_k1_AXP_s006
                                 + 0.0101568849576545 w_k1_AXP_s007
                                 + 0.0124612997938322 w_k1_AXP_s010
                                 + 0.0192452062747752 w_k1_AXP_s011
                                 + 0.00914775698365069 w_k1_AXP_s012
                                 + 0.00592564330163398 w_k1_AXP_s013
                                 + 0.00720691180157471 w_k1_AXP_s014
                                 + 0.0105517361896078 w_k1_AXP_s015
                                 + 0.0102757327776643 w_k1_AXP_s016
                                 + 0.00632541086662712 w_k1_AXP_s017
                                 + 0.124623537258466 w_k1_AXP_s100
                                 + 0.153121974945791 w_k1_AXP_s101
                                 + 0.0413776371317892 w_k1_AXP_s102
                                 + 0.080008264927419 w_k1_AXP_s103
                                 + 0.0397422176014516 w_k1_AXP_s104
                                 + 0.0312384076589322 w_k1_AXP_s105
                                 + 0.154621607705873 w_k1_AXP_s106
                                 + 0.0332798402464886 w_k1_AXP_s107
                                 + 0.0171819710285049 w_k1_AXP_s110
                                 + 0.0104371068376175 w_k1_AXP_s111
                                 + 0.0210379635818222 w_k1_AXP_s112
                                 + 0.0121405631656285 w_k1_AXP_s113
                                 + 0.021265124248484 w_k1_AXP_s114
                                 + 0.0120804036732858 w_k1_AXP_s115
                                 + 0.0425111027030425 w_k1_AXP_s116
                                 + 0.02450685897958 w_k1_AXP_s117
                                 + 0.0301457052280966 w_k1_DIS_s000
                                 + 0.0123812245681712 w_k1_DIS_s001
                                 + 0.0186612440586999 w_k1_DIS_s002
                                 + 0.00890093288892255 w_k1_DIS_s003
                                 + 0.0287771266892457 w_k1_DIS_s004
                                 + 0.0255367614236186 w_k1_DIS_s005
                                 + 0.010967998710955 w_k1_DIS_s006
                                 + 0.0106831093928672 w_k1_DIS_s007
                                 + 0.013164193689405 w_k1_DIS_s010
                                 + 0.0230964772380691 w_k1_DIS_s011
                                 + 0.0100834731031531 w_k1_DIS_s012
                                 + 0.00615415623953645 w_k1_DIS_s013
                                 + 0.00660108930532025 w_k1_DIS_s014
                                 + 0.00909556664219435 w_k1_DIS_s015
                                 + 0.0112167130158616 w_k1_DIS_s016
                                 + 0.00590047915240457 w_k1_DIS_s017
                                 + 0.113810285139574 w_k1_DIS_s100
                                 + 0.147359590601335 w_k1_DIS_s101
                                 + 0.0404938328321574 w_k1_DIS_s102
                                 + 0.0791789296905155 w_k1_DIS_s103
                                 + 0.0426926681080348 w_k1_DIS_s104
                                 + 0.0284276749155132 w_k1_DIS_s105
                                 + 0.152071359931282 w_k1_DIS_s106
                                 + 0.0398658963462041 w_k1_DIS_s107
                                 + 0.0174365162114877 w_k1_DIS_s110
                                 + 0.0116815107945779 w_k1_DIS_s111
                                 + 0.0215622107251526 w_k1_DIS_s112
                                 + 0.0111535945720224 w_k1_DIS_s113
                                 + 0.0204112827438093 w_k1_DIS_s114
                                 + 0.0136590393848819 w_k1_DIS_s115
                                 + 0.0422497989845476 w_k1_DIS_s116
                                 + 0.0278654381243489 w_k1_DIS_s117
                                 + 0.0293513822647982 w_k1_FDX_s000
                                 + 0.0121410490482025 w_k1_FDX_s001
                                 + 0.016576623831268 w_k1_FDX_s002
                                 + 0.00864249681556948 w_k1_FDX_s003
                                 + 0.0267268857947297 w_k1_FDX_s004
                                 + 0.0272104121025611 w_k1_FDX_s005
                                 + 0.010777009722901 w_k1_FDX_s006
                                 + 0.010108158231291 w_k1_FDX_s007
                                 + 0.012181338285679 w_k1_FDX_s010
                                 + 0.0218401656522966 w_k1_FDX_s011
                                 + 0.0096720111397689 w_k1_FDX_s012
                                 + 0.00577417812238971 w_k1_FDX_s013
                                 + 0.00658655530456965 w_k1_FDX_s014
                                 + 0.00939590838119447 w_k1_FDX_s015
                                 + 0.0101181312754306 w_k1_FDX_s016
                                 + 0.00602966910393994 w_k1_FDX_s017
                                 + 0.11663410030942 w_k1_FDX_s100
                                 + 0.148233193449813 w_k1_FDX_s101
                                 + 0.0385366950154493 w_k1_FDX_s102
                                 + 0.077730883061644 w_k1_FDX_s103
                                 + 0.0406120628622095 w_k1_FDX_s104
                                 + 0.0297939932241166 w_k1_FDX_s105
                                 + 0.140085550719865 w_k1_FDX_s106
                                 + 0.0374561335214232 w_k1_FDX_s107
                                 + 0.0175639580424406 w_k1_FDX_s110
                                 + 0.0103638541371501 w_k1_FDX_s111
                                 + 0.0215508141218596 w_k1_FDX_s112
                                 + 0.0112524790632798 w_k1_FDX_s113
                                 + 0.019453093675042 w_k1_FDX_s114
                                 + 0.0130700787037943 w_k1_FDX_s115
                                 + 0.0429834049091601 w_k1_FDX_s116
                                 + 0.0259423603407104 w_k1_FDX_s117
                                 + 0.0293441038252804 w_k1_IBM_s000
                                 + 0.0123021618368404 w_k1_IBM_s001
                                 + 0.0160144870591665 w_k1_IBM_s002
                                 + 0.00776531333778699 w_k1_IBM_s003
                                 + 0.0276963973119988 w_k1_IBM_s004
                                 + 0.0270394221005079 w_k1_IBM_s005
                                 + 0.0101994327983981 w_k1_IBM_s006
                                 + 0.0100808212727501 w_k1_IBM_s007
                                 + 0.0125265779487881 w_k1_IBM_s010
                                 + 0.0218123252278141 w_k1_IBM_s011
                                 + 0.0096493506431082 w_k1_IBM_s012
                                 + 0.00600635214188689 w_k1_IBM_s013
                                 + 0.00645442258367974 w_k1_IBM_s014
                                 + 0.0091486066651856 w_k1_IBM_s015
                                 + 0.00974808152410196 w_k1_IBM_s016
                                 + 0.00603162260041404 w_k1_IBM_s017
                                 + 0.122425358000276 w_k1_IBM_s100
                                 + 0.156290470998153 w_k1_IBM_s101
                                 + 0.0409426992339826 w_k1_IBM_s102
                                 + 0.0784365110974819 w_k1_IBM_s103
                                 + 0.0398770921354262 w_k1_IBM_s104
                                 + 0.0301915103545079 w_k1_IBM_s105
                                 + 0.141330900867175 w_k1_IBM_s106
                                 + 0.038448295726381 w_k1_IBM_s107
                                 + 0.0178900725610236 w_k1_IBM_s110
                                 + 0.0114123162577297 w_k1_IBM_s111
                                 + 0.021556920442148 w_k1_IBM_s112
                                 + 0.0116989618490958 w_k1_IBM_s113
                                 + 0.0212950646169541 w_k1_IBM_s114
                                 + 0.0128997226148565 w_k1_IBM_s115
                                 + 0.0420105614045167 w_k1_IBM_s116
                                 + 0.0280848267562187 w_k1_IBM_s117
                                 + 0.0311145585398702 w_k1_INTC_s000
                                 + 0.0108172365591749 w_k1_INTC_s001
                                 + 0.0158871067165702 w_k1_INTC_s002
                                 + 0.00893320627798081 w_k1_INTC_s003
                                 + 0.0261677883554998 w_k1_INTC_s004
                                 + 0.0262977759765906 w_k1_INTC_s005
                                 + 0.0118226301630293 w_k1_INTC_s006
                                 + 0.0097911856697541 w_k1_INTC_s007
                                 + 0.0119861888849205 w_k1_INTC_s010
                                 + 0.0215109242445569 w_k1_INTC_s011
                                 + 0.00960931390380295 w_k1_INTC_s012
                                 + 0.00686035972496474 w_k1_INTC_s013
                                 + 0.00740163750646925 w_k1_INTC_s014
                                 + 0.0100460379377579 w_k1_INTC_s015
                                 + 0.0118942228385178 w_k1_INTC_s016
                                 + 0.00673595001945185 w_k1_INTC_s017
                                 + 0.110346927929322 w_k1_INTC_s100
                                 + 0.146757359798121 w_k1_INTC_s101
                                 + 0.0337713868546935 w_k1_INTC_s102
                                 + 0.0789458199353555 w_k1_INTC_s103
                                 + 0.0397865298347748 w_k1_INTC_s104
                                 + 0.0332603766820183 w_k1_INTC_s105
                                 + 0.149960080970738 w_k1_INTC_s106
                                 + 0.0338421049740748 w_k1_INTC_s107
                                 + 0.0188253237922519 w_k1_INTC_s110
                                 + 0.0108771432866826 w_k1_INTC_s111
                                 + 0.0228644325763182 w_k1_INTC_s112
                                 + 0.010756311266173 w_k1_INTC_s113
                                 + 0.018232036451099 w_k1_INTC_s114
                                 + 0.0115577115918543 w_k1_INTC_s115
                                 + 0.039819293748728 w_k1_INTC_s116
                                 + 0.0275589946454352 w_k1_INTC_s117
                                 + 0.0276966343348567 w_k1_KO_s000
                                 + 0.0109064983953573 w_k1_KO_s001
                                 + 0.016107615155213 w_k1_KO_s002
                                 + 0.0085368445839422 w_k1_KO_s003
                                 + 0.0285589831256677 w_k1_KO_s004
                                 + 0.0257512637834552 w_k1_KO_s005
                                 + 0.0111841441773135 w_k1_KO_s006
                                 + 0.0112072567796508 w_k1_KO_s007
                                 + 0.0127540746226668 w_k1_KO_s010
                                 + 0.0237357961993589 w_k1_KO_s011
                                 + 0.0092725283242063 w_k1_KO_s012
                                 + 0.00549790990596806 w_k1_KO_s013
                                 + 0.00749297016776077 w_k1_KO_s014
                                 + 0.00894384212324282 w_k1_KO_s015
                                 + 0.00972780969692244 w_k1_KO_s016
                                 + 0.006266644379831 w_k1_KO_s017
                                 + 0.108121878170392 w_k1_KO_s100
                                 + 0.145236859670885 w_k1_KO_s101
                                 + 0.0420223597024249 w_k1_KO_s102
                                 + 0.0799706459038978 w_k1_KO_s103
                                 + 0.0397950543752551 w_k1_KO_s104
                                 + 0.0321625139343888 w_k1_KO_s105
                                 + 0.142166621519621 w_k1_KO_s106
                                 + 0.0394443275458411 w_k1_KO_s107
                                 + 0.016804467058886 w_k1_KO_s110
                                 + 0.0108039766332802 w_k1_KO_s111
                                 + 0.0210309368577265 w_k1_KO_s112
                                 + 0.0116142425869579 w_k1_KO_s113
                                 + 0.0211912802962556 w_k1_KO_s114
                                 + 0.0125796784192778 w_k1_KO_s115
                                 + 0.0389809612809826 w_k1_KO_s116
                                 + 0.0310152752219481 w_k1_KO_s117
                                 + 0.0316589753017904 w_k1_MAT_s000
                                 + 0.0108655721132231 w_k1_MAT_s001
                                 + 0.0154604571434212 w_k1_MAT_s002
                                 + 0.00844537710422887 w_k1_MAT_s003
                                 + 0.0219073105290317 w_k1_MAT_s004
                                 + 0.0328945022283805 w_k1_MAT_s005
                                 + 0.00776789043582485 w_k1_MAT_s006
                                 + 0.00943324444956947 w_k1_MAT_s007
                                 + 0.0138337585013733 w_k1_MAT_s010
                                 + 0.022995165857497 w_k1_MAT_s011
                                 + 0.00928326684017061 w_k1_MAT_s012
                                 + 0.00519004755706864 w_k1_MAT_s013
                                 + 0.00625406283829963 w_k1_MAT_s014
                                 + 0.00999183343352223 w_k1_MAT_s015
                                 + 0.00844590561075308 w_k1_MAT_s016
                                 + 0.00569022552152359 w_k1_MAT_s017
                                 + 0.108885175404844 w_k1_MAT_s100
                                 + 0.166559286022759 w_k1_MAT_s101
                                 + 0.036882021149201 w_k1_MAT_s102
                                 + 0.0956193006194516 w_k1_MAT_s103
                                 + 0.0354678155951787 w_k1_MAT_s104
                                 + 0.0345477171737453 w_k1_MAT_s105
                                 + 0.155581324608243 w_k1_MAT_s106
                                 + 0.0368242423405189 w_k1_MAT_s107
                                 + 0.0203008418112375 w_k1_MAT_s110
                                 + 0.00980120583062754 w_k1_MAT_s111
                                 + 0.015943758943598 w_k1_MAT_s112
                                 + 0.0101506437415815 w_k1_MAT_s113
                                 + 0.0206126580936861 w_k1_MAT_s114
                                 + 0.0122982820370038 w_k1_MAT_s115
                                 + 0.0448051908239265 w_k1_MAT_s116
                                 + 0.0260904306718372 w_k1_MAT_s117
                                 + 0.0296650149203758 w_k1_MSFT_s000
                                 + 0.0118913027409672 w_k1_MSFT_s001
                                 + 0.0166574924333721 w_k1_MSFT_s002
                                 + 0.00804726909663512 w_k1_MSFT_s003
                                 + 0.0255731559881263 w_k1_MSFT_s004
                                 + 0.0270475957375101 w_k1_MSFT_s005
                                 + 0.0108326322355166 w_k1_MSFT_s006
                                 + 0.0096940709582247 w_k1_MSFT_s007
                                 + 0.0128664900694791 w_k1_MSFT_s010
                                 + 0.0219323541066361 w_k1_MSFT_s011
                                 + 0.0084373223471336 w_k1_MSFT_s012
                                 + 0.0059354012445954 w_k1_MSFT_s013
                                 + 0.00704185842604036 w_k1_MSFT_s014
                                 + 0.00897845277716163 w_k1_MSFT_s015
                                 + 0.0103650336216947 w_k1_MSFT_s016
                                 + 0.00549373906578262 w_k1_MSFT_s017
                                 + 0.10801909773539 w_k1_MSFT_s100
                                 + 0.142080989753556 w_k1_MSFT_s101
                                 + 0.0399731910361831 w_k1_MSFT_s102
                                 + 0.078523175240719 w_k1_MSFT_s103
                                 + 0.0395078703165619 w_k1_MSFT_s104
                                 + 0.032378205991929 w_k1_MSFT_s105
                                 + 0.141866589813563 w_k1_MSFT_s106
                                 + 0.0367786571023279 w_k1_MSFT_s107
                                 + 0.0182312395507702 w_k1_MSFT_s110
                                 + 0.0114816414391673 w_k1_MSFT_s111
                                 + 0.0201412981228849 w_k1_MSFT_s112
                                 + 0.0105903768080679 w_k1_MSFT_s113
                                 + 0.0209885873945533 w_k1_MSFT_s114
                                 + 0.0120642979477381 w_k1_MSFT_s115
                                 + 0.0431436367024303 w_k1_MSFT_s116
                                 + 0.0257792715160849 w_k1_MSFT_s117
                                 + 0.0302133400202082 w_k1_PG_s000
                                 + 0.0118412587525084 w_k1_PG_s001
                                 + 0.0155798992395275 w_k1_PG_s002
                                 + 0.00842558110971479 w_k1_PG_s003
                                 + 0.0254076958936515 w_k1_PG_s004
                                 + 0.026693570936109 w_k1_PG_s005
                                 + 0.0108834205244959 w_k1_PG_s006
                                 + 0.0106238158203722 w_k1_PG_s007
                                 + 0.011409473189092 w_k1_PG_s010
                                 + 0.0224367899992144 w_k1_PG_s011
                                 + 0.00883158955343842 w_k1_PG_s012
                                 + 0.00569799183509755 w_k1_PG_s013
                                 + 0.00645160219045609 w_k1_PG_s014
                                 + 0.00874339977080781 w_k1_PG_s015
                                 + 0.00918638733652933 w_k1_PG_s016
                                 + 0.0060797673124101 w_k1_PG_s017
                                 + 0.119907038611296 w_k1_PG_s100
                                 + 0.138813025443309 w_k1_PG_s101
                                 + 0.0398157741762142 w_k1_PG_s102
                                 + 0.0750737950686631 w_k1_PG_s103
                                 + 0.0381544348096959 w_k1_PG_s104
                                 + 0.0319894392655715 w_k1_PG_s105
                                 + 0.141147594139119 w_k1_PG_s106
                                 + 0.0353972209157881 w_k1_PG_s107
                                 + 0.0162157738772771 w_k1_PG_s110
                                 + 0.0109498082809189 w_k1_PG_s111
                                 + 0.0219240674997638 w_k1_PG_s112
                                 + 0.0117517523652833 w_k1_PG_s113
                                 + 0.021418776266593 w_k1_PG_s114
                                 + 0.012679181065324 w_k1_PG_s115
                                 + 0.0423170292258888 w_k1_PG_s116
                                 + 0.0264771414448114 w_k1_PG_s117 >= 1.14
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
