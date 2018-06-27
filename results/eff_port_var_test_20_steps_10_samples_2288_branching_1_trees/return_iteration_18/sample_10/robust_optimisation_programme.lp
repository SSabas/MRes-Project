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
 w_k1_ADM_s0_return_constraint:  0.95966 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.10017 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.20684 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.83656 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.800769 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.09928 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.18559 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.82905 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.93032 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.14882 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.790505 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.01234 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.929277 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.766853 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.12564 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.896417 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.986686 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.39787 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.23014 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.09405 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.18535 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.84302 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.35583 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.3705 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.15422 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.21357 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.09949 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.829126 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.14042 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.985764 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.854373 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.12794 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.0229 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.685107 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.799465 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.24019 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.01076 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.13345 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.04088 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.13985 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.15103 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.701911 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.807814 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.27563 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.31611 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.657353 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.843551 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.41965 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.799928 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.18707 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.00004 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.597616 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.06054 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.742892 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.902641 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.72791 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.2505 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.48173 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.22287 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.721675 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.17271 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.36519 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.918195 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.05477 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.906976 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.688507 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.885482 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.880285 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.721735 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.03186 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.84505 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.588738 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.675683 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.45316 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.932832 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.991452 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  0.951734 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.34545 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.2835 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.673324 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.774864 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.21217 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.57765 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.661046 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.963115 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.19103 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.617168 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.970533 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.992998 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.517872 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.23006 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.759076 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.19185 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.74234 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.48337 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.32145 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.04343 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.766019 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.3148 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.67225 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.20617 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.06875 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.851739 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.587837 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.977978 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.897674 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.635891 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.14355 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.811454 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.534958 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.709268 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.59533 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.870992 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.904166 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  0.968807 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  1.11873 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.1623 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.797988 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.847918 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.10462 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.2993 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.863913 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.09525 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.06549 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.696896 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.09481 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.968333 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.698381 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.0467 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.799115 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.03144 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.44675 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.28657 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.05263 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.05137 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.854266 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.24278 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.35063 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.1553 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.22331 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.966898 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.724706 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.869706 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.944528 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.856656 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.13802 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.870354 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.615926 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.813871 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.3824 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.11275 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.828438 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.951693 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.31136 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.10681 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.878051 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.759581 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.16443 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.33751 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.497785 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.606813 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.73267 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.15234 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.842088 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.07669 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.709183 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.42537 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.10855 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.931458 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.19128 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.828856 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.81385 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.981085 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.604624 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.0024 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.36634 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.792583 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.622087 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.892289 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.855016 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.28386 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.734873 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.621525 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.870859 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.17824 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.954401 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.828323 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.22574 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.57245 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.65063 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 0.973191 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.18234 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.991857 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.80341 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.888011 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.17422 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.39495 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.85288 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.20325 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.1477 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.8288 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.06832 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.13589 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.781532 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.0367 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.909963 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.11375 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.26017 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.26722 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.21401 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.07918 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.828561 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.05865 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.11492 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.915427 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.995202 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.898586 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.708896 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.780872 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.991302 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.769895 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.807755 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.718108 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.672543 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.770319 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.01817 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.04819 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.828183 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.967001 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.272 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.1735 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.745849 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.794079 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.1359 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.29866 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.61604 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.823159 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.35873 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.787471 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.875913 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.04425 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.694934 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.35785 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.990246 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.0746 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.59776 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.33627 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.43029 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.10255 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.781724 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.15511 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.39989 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.981452 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.823263 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.975285 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.753018 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.04526 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.895508 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.74516 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.999162 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.976312 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.770482 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.760473 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.36066 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.772784 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.23916 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.996344 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.31353 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.16889 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.850672 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.708924 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.26658 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.34934 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.366191 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.621078 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 2.08752 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.01412 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.794395 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.892195 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.613996 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.38494 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.02724 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.756239 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.19002 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.8225 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 2.04648 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.02105 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.534973 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.19966 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.3216 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.10354 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.752315 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.936311 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.712524 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.02734 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.721108 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.531913 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.863486 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.1081 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.86814 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.641887 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.35019 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.600125 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.37324 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 0.982083 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.10161 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.15231 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.796243 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.875786 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.16819 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.23396 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.818231 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.10245 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.02432 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.750137 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.932404 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.938873 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.727865 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.10394 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.813365 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.08402 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.47643 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.25648 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.18839 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.999323 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.856915 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.12163 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.35026 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.18434 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.20762 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.982316 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.790517 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.969715 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.981327 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.808208 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.03857 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.840144 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.65302 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.752474 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.29008 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.05609 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.861013 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   0.955758 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.12099 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.07591 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.894581 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.88154 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.06916 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.20699 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.789836 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.934944 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.20884 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.875805 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.982429 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.05129 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.842601 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.05793 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.887027 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.0433 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.23839 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.06233 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.21126 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.04648 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.813563 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.18103 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.3158 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.04406 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.963673 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.922915 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.805431 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.981733 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.866073 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.859374 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.998841 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.02692 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.850578 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.840304 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.13151 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.957536 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.05176 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.772998959860304 z_k1_s0000
                                 - 0.578999977404583 z_k1_s0001
                                 - 0.837997429074387 z_k1_s0002
                                 - 0.559000300446346 z_k1_s0003
                                 - 0.948001743747054 z_k1_s0004
                                 - 0.917999984108828 z_k1_s0005
                                 - 0.868999461100638 z_k1_s0006
                                 - 0.928002707989066 z_k1_s0007
                                 - 0.136000118428788 z_k1_s0010
                                 - 0.28499927318871 z_k1_s0011
                                 - 0.173999940420592 z_k1_s0012
                                 - 0.443000773971097 z_k1_s0013
                                 - 0.253000447511711 z_k1_s0014
                                 - 0.458000223507189 z_k1_s0015
                                 - 0.186999949952019 z_k1_s0016
                                 - 0.497001225301767 z_k1_s0017
                                 - 1.20299851090684 z_k1_s0020
                                 - 0.672997932506597 z_k1_s0021
                                 - 0.381999598004197 z_k1_s0022
                                 - 1.00700005316126 z_k1_s0023
                                 - 0.667000274073863 z_k1_s0024
                                 - 0.497999645669217 z_k1_s0025
                                 - 0.365999915988085 z_k1_s0026
                                 - 0.432999564867307 z_k1_s0027
                                 - 0.85299942173869 z_k1_s0030
                                 - 0.476000065846864 z_k1_s0031
                                 - 0.765001222468711 z_k1_s0032
                                 - 0.780999000153252 z_k1_s0033
                                 - 1.26400161116379 z_k1_s0034
                                 - 0.274000243716151 z_k1_s0035
                                 - 0.335000239017529 z_k1_s0036
                                 - 0.463000177902362 z_k1_s0037
                                 - 0.306000001438426 z_k1_s0040
                                 - 0.356999402545141 z_k1_s0041
                                 - 0.688998618077436 z_k1_s0042
                                 - 0.412999821453917 z_k1_s0043
                                 - 0.16399991960495 z_k1_s0044
                                 - 0.393999679324154 z_k1_s0045
                                 - 0.598001205348978 z_k1_s0046
                                 - 0.346999327740003 z_k1_s0047
                                 - 0.482998434715716 z_k1_s0050
                                 - 1.71999899724362 z_k1_s0051
                                 - 0.553998821221632 z_k1_s0052
                                 - 1.49499270993665 z_k1_s0053
                                 - 1.91599374819083 z_k1_s0054
                                 - 0.489998858533497 z_k1_s0055
                                 - 0.444998456869636 z_k1_s0056
                                 - 1.45499273325656 z_k1_s0057
                                 - 1.14100580143369 z_k1_s0060
                                 - 1.92400752505538 z_k1_s0061
                                 - 0.652000921729607 z_k1_s0062
                                 - 0.754001194405308 z_k1_s0063
                                 - 0.704001040209342 z_k1_s0064
                                 - 3.23501029060951 z_k1_s0065
                                 - 1.10400122667535 z_k1_s0066
                                 - 0.957001757614826 z_k1_s0067
                                 - 0.435999271066109 z_k1_s0070
                                 - 0.593999867011921 z_k1_s0071
                                 - 0.170000086063321 z_k1_s0072
                                 - 0.31800015351139 z_k1_s0073
                                 - 0.509998986990204 z_k1_s0074
                                 - 0.192999903288395 z_k1_s0075
                                 - 0.743000368669692 z_k1_s0076
                                 - 0.213999805493885 z_k1_s0077
                                 - 0.252000489922798 z_k1_s0100
                                 - 0.246000951258495 z_k1_s0101
                                 - 0.259000553698487 z_k1_s0102
                                 - 0.191000966164369 z_k1_s0103
                                 - 0.225999479040808 z_k1_s0104
                                 - 0.208000863047898 z_k1_s0105
                                 - 0.304000705684773 z_k1_s0106
                                 - 0.120000164496494 z_k1_s0107
                                 - 0.679001745523096 z_k1_s0110
                                 - 0.363000322719695 z_k1_s0111
                                 - 0.216000001915755 z_k1_s0112
                                 - 0.833000823888488 z_k1_s0113
                                 - 0.321000109347079 z_k1_s0114
                                 - 0.354000216139817 z_k1_s0115
                                 - 1.04700203278712 z_k1_s0116
                                 - 0.447002027465578 z_k1_s0117
                                 - 0.628000954886999 z_k1_s0120
                                 - 0.464000823942058 z_k1_s0121
                                 - 0.134000061645229 z_k1_s0122
                                 - 0.316000446650126 z_k1_s0123
                                 - 0.232000146370869 z_k1_s0124
                                 - 0.318000415856752 z_k1_s0125
                                 - 0.274001093310978 z_k1_s0126
                                 - 0.290000846963987 z_k1_s0127
                                 - 0.158000037699911 z_k1_s0130
                                 - 0.2499998974999 z_k1_s0131
                                 - 0.201999985499902 z_k1_s0132
                                 - 0.197999850499938 z_k1_s0133
                                 - 0.195999782999955 z_k1_s0134
                                 - 0.339999518999949 z_k1_s0135
                                 - 0.195000603249751 z_k1_s0136
                                 - 0.168999896549941 z_k1_s0137
                                 - 0.411999421695473 z_k1_s0140
                                 - 0.376001022235965 z_k1_s0141
                                 - 0.550003224400351 z_k1_s0142
                                 - 0.416000312471629 z_k1_s0143
                                 - 0.533999661295724 z_k1_s0144
                                 - 0.772002651342353 z_k1_s0145
                                 - 0.393002884529451 z_k1_s0146
                                 - 0.393999260213129 z_k1_s0147
                                 - 0.418002066419139 z_k1_s0150
                                 - 0.728001539406866 z_k1_s0151
                                 - 0.506001835138135 z_k1_s0152
                                 - 0.208000258973167 z_k1_s0153
                                 - 0.186000316793418 z_k1_s0154
                                 - 0.291000271015231 z_k1_s0155
                                 - 0.239000206271939 z_k1_s0156
                                 - 0.589000897679026 z_k1_s0157
                                 - 1.06200275293213 z_k1_s0160
                                 - 0.429000637846178 z_k1_s0161
                                 - 0.369000203363893 z_k1_s0162
                                 - 0.389000531058148 z_k1_s0163
                                 - 0.716998883157788 z_k1_s0164
                                 - 0.881002996611926 z_k1_s0165
                                 - 1.2580012463717 z_k1_s0166
                                 - 0.38100029026035 z_k1_s0167
                                 - 1.36499930883745 z_k1_s0170
                                 - 0.684996732116428 z_k1_s0171
                                 - 1.13399765832883 z_k1_s0172
                                 - 0.342999808995074 z_k1_s0173
                                 - 0.71300184708067 z_k1_s0174
                                 - 0.999000028858348 z_k1_s0175
                                 - 1.07099966564141 z_k1_s0176
                                 - 0.447999391936897 z_k1_s0177
                                 - 0.0339999421155256 z_k1_s1000
                                 - 0.110999846524173 z_k1_s1001
                                 - 0.144999575639955 z_k1_s1002
                                 - 0.0839999989911278 z_k1_s1003
                                 - 0.103999964941437 z_k1_s1004
                                 - 0.0559999046608657 z_k1_s1005
                                 - 0.114999981714064 z_k1_s1006
                                 - 0.0609999316484003 z_k1_s1007
                                 - 0.271999842235293 z_k1_s1010
                                 - 0.188999851244446 z_k1_s1011
                                 - 0.16399962312472 z_k1_s1012
                                 - 0.22899993239651 z_k1_s1013
                                 - 0.173999643412736 z_k1_s1014
                                 - 0.191999679931165 z_k1_s1015
                                 - 0.439999118675848 z_k1_s1016
                                 - 0.113999876484011 z_k1_s1017
                                 - 0.0279999494542521 z_k1_s1020
                                 - 0.0389999494362599 z_k1_s1021
                                 - 0.120999898802186 z_k1_s1022
                                 - 0.0790001008706818 z_k1_s1023
                                 - 0.0409999494329886 z_k1_s1024
                                 - 0.0990001008379686 z_k1_s1025
                                 - 0.0269999494558878 z_k1_s1026
                                 - 0.0709998988839694 z_k1_s1027
                                 - 0.0909998523712768 z_k1_s1030
                                 - 0.165999724653106 z_k1_s1031
                                 - 0.144999532998468 z_k1_s1032
                                 - 0.0529999684337549 z_k1_s1033
                                 - 0.0449999783890309 z_k1_s1034
                                 - 0.0719999104025159 z_k1_s1035
                                 - 0.202999712997413 z_k1_s1036
                                 - 0.142000029618602 z_k1_s1037
                                 - 0.155000635444444 z_k1_s1040
                                 - 0.0600002053009554 z_k1_s1041
                                 - 0.0540001556707743 z_k1_s1042
                                 - 0.0700001586842104 z_k1_s1043
                                 - 0.07700018425266 z_k1_s1044
                                 - 0.131999927860563 z_k1_s1045
                                 - 0.2140010588077 z_k1_s1046
                                 - 0.208000621176379 z_k1_s1047
                                 - 0.170000218722255 z_k1_s1050
                                 - 0.0910000769336436 z_k1_s1051
                                 - 0.249999961355952 z_k1_s1052
                                 - 0.170000218722255 z_k1_s1053
                                 - 0.198000417394293 z_k1_s1054
                                 - 0.146000173432042 z_k1_s1055
                                 - 0.310999762926645 z_k1_s1056
                                 - 0.413999600005173 z_k1_s1057
                                 - 0.196000213383845 z_k1_s1060
                                 - 0.187000170841192 z_k1_s1061
                                 - 0.677999401676685 z_k1_s1062
                                 - 0.554000717135044 z_k1_s1063
                                 - 0.263999346316237 z_k1_s1064
                                 - 0.168000081028926 z_k1_s1065
                                 - 0.175000114117655 z_k1_s1066
                                 - 0.155000019578427 z_k1_s1067
                                 - 0.0469999297268001 z_k1_s1070
                                 - 0.113999722494943 z_k1_s1071
                                 - 0.0989995856142732 z_k1_s1072
                                 - 0.0950000682453638 z_k1_s1073
                                 - 0.113999722494943 z_k1_s1074
                                 - 0.0459999605346993 z_k1_s1075
                                 - 0.0509999262950348 z_k1_s1076
                                 - 0.0329999417379786 z_k1_s1077
                                 - 0.143999991891824 z_k1_s1100
                                 - 0.146999891806173 z_k1_s1101
                                 - 0.138000192063125 z_k1_s1102
                                 - 0.135000292148776 z_k1_s1103
                                 - 0.252999952448808 z_k1_s1104
                                 - 0.169000356845499 z_k1_s1105
                                 - 0.0850000418417313 z_k1_s1106
                                 - 0.128000525681961 z_k1_s1107
                                 - 0.520000116844412 z_k1_s1110
                                 - 0.488000002699946 z_k1_s1111
                                 - 0.133000344062699 z_k1_s1112
                                 - 0.371001106112006 z_k1_s1113
                                 - 0.538002353554621 z_k1_s1114
                                 - 0.351000165770136 z_k1_s1115
                                 - 0.488000002699946 z_k1_s1116
                                 - 0.587000138583992 z_k1_s1117
                                 - 0.20699991692337 z_k1_s1120
                                 - 0.165000222927811 z_k1_s1121
                                 - 0.498000395439835 z_k1_s1122
                                 - 0.19300041012551 z_k1_s1123
                                 - 0.362001442197168 z_k1_s1124
                                 - 0.189000383382981 z_k1_s1125
                                 - 0.178000212040863 z_k1_s1126
                                 - 0.164000314042344 z_k1_s1127
                                 - 0.121999629187818 z_k1_s1130
                                 - 0.118999883002969 z_k1_s1131
                                 - 0.102999631683136 z_k1_s1132
                                 - 0.222000477686485 z_k1_s1133
                                 - 0.20800037815668 z_k1_s1134
                                 - 0.0569999925140462 z_k1_s1135
                                 - 0.0840000193786064 z_k1_s1136
                                 - 0.0479999835591928 z_k1_s1137
                                 - 0.130000037450315 z_k1_s1140
                                 - 0.50800056330427 z_k1_s1141
                                 - 0.258999892192568 z_k1_s1142
                                 - 0.150000043211902 z_k1_s1143
                                 - 0.227000039334014 z_k1_s1144
                                 - 0.494001132591106 z_k1_s1145
                                 - 0.311000011412684 z_k1_s1146
                                 - 0.527000907557746 z_k1_s1147
                                 - 0.577999117998744 z_k1_s1150
                                 - 0.301000312379355 z_k1_s1151
                                 - 0.25000019896751 z_k1_s1152
                                 - 0.41600162628163 z_k1_s1153
                                 - 0.354001743389053 z_k1_s1154
                                 - 0.2730003967279 z_k1_s1155
                                 - 0.38000001634233 z_k1_s1156
                                 - 0.699001458791659 z_k1_s1157
                                 - 0.234999650988334 z_k1_s1160
                                 - 0.114999769462375 z_k1_s1161
                                 - 0.0720000225155762 z_k1_s1162
                                 - 0.132999985691272 z_k1_s1163
                                 - 0.116000353475099 z_k1_s1164
                                 - 0.103999897322497 z_k1_s1165
                                 - 0.0779999697918735 z_k1_s1166
                                 - 0.0829999882554554 z_k1_s1167
                                 - 0.30699987681544 z_k1_s1170
                                 - 0.0910000807336001 z_k1_s1171
                                 - 0.213999725516247 z_k1_s1172
                                 - 0.116000023803248 z_k1_s1173
                                 - 0.241000678151754 z_k1_s1174
                                 - 0.173999879204828 z_k1_s1175
                                 - 0.321000370774592 z_k1_s1176
                                 - 0.100999964061433 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.47518 w_k1_ADM_s000
                                 + 1.70804 w_k1_AXP_s000
                                 + 1.77362 w_k1_DIS_s000
                                 + 1.52589 w_k1_FDX_s000
                                 + 1.17012 w_k1_IBM_s000
                                 + 1.25727 w_k1_INTC_s000
                                 + 1.48809 w_k1_KO_s000 + 1.10753 w_k1_MAT_s000
                                 + 1.49296 w_k1_MSFT_s000
                                 + 1.28548 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.638958 w_k1_ADM_s000
                                 + 0.558554 w_k1_AXP_s000
                                 + 0.5041 w_k1_DIS_s000
                                 + 0.615901 w_k1_FDX_s000
                                 + 1.06082 w_k1_IBM_s000
                                 + 0.719197 w_k1_INTC_s000
                                 + 0.767583 w_k1_KO_s000
                                 + 0.895755 w_k1_MAT_s000
                                 + 0.700432 w_k1_MSFT_s000
                                 + 0.781917 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.01684 w_k1_ADM_s000
                                 + 0.988226 w_k1_AXP_s000
                                 + 1.04908 w_k1_DIS_s000
                                 + 1.13637 w_k1_FDX_s000
                                 + 0.715306 w_k1_IBM_s000
                                 + 1.15936 w_k1_INTC_s000
                                 + 1.08786 w_k1_KO_s000
                                 + 0.649742 w_k1_MAT_s000
                                 + 1.19062 w_k1_MSFT_s000
                                 + 1.07993 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.721757 w_k1_ADM_s000
                                 + 0.711493 w_k1_AXP_s000
                                 + 0.605802 w_k1_DIS_s000
                                 + 0.799031 w_k1_FDX_s000
                                 + 0.701499 w_k1_IBM_s000
                                 + 0.923355 w_k1_INTC_s000
                                 + 0.703761 w_k1_KO_s000
                                 + 0.690289 w_k1_MAT_s000
                                 + 0.718741 w_k1_MSFT_s000
                                 + 0.833941 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.35229 w_k1_ADM_s000
                                 + 1.35843 w_k1_AXP_s000
                                 + 1.26839 w_k1_DIS_s000
                                 + 1.21445 w_k1_FDX_s000
                                 + 0.985309 w_k1_IBM_s000
                                 + 1.0225 w_k1_INTC_s000 + 1.19666 w_k1_KO_s000
                                 + 1.20045 w_k1_MAT_s000
                                 + 1.17068 w_k1_MSFT_s000
                                 + 1.14793 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.863101 w_k1_ADM_s000
                                 + 0.986026 w_k1_AXP_s000
                                 + 0.990951 w_k1_DIS_s000
                                 + 0.828301 w_k1_FDX_s000
                                 + 1.62183 w_k1_IBM_s000
                                 + 0.91806 w_k1_INTC_s000
                                 + 1.29416 w_k1_KO_s000 + 1.24243 w_k1_MAT_s000
                                 + 0.980784 w_k1_MSFT_s000
                                 + 1.05366 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.941538 w_k1_ADM_s000
                                 + 0.871184 w_k1_AXP_s000
                                 + 0.833333 w_k1_DIS_s000
                                 + 0.874622 w_k1_FDX_s000
                                 + 0.905725 w_k1_IBM_s000
                                 + 0.930258 w_k1_INTC_s000
                                 + 0.946587 w_k1_KO_s000
                                 + 0.814956 w_k1_MAT_s000
                                 + 0.923876 w_k1_MSFT_s000
                                 + 1.08005 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.995563 w_k1_ADM_s000
                                 + 0.927755 w_k1_AXP_s000
                                 + 0.914527 w_k1_DIS_s000
                                 + 0.888143 w_k1_FDX_s000
                                 + 1.15891 w_k1_IBM_s000
                                 + 0.880016 w_k1_INTC_s000
                                 + 1.00277 w_k1_KO_s000 + 1.25777 w_k1_MAT_s000
                                 + 0.926398 w_k1_MSFT_s000
                                 + 1.00543 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.797651 w_k1_ADM_s001
                                 + 0.617362 w_k1_AXP_s001
                                 + 0.609585 w_k1_DIS_s001
                                 + 0.745465 w_k1_FDX_s001
                                 + 0.611461 w_k1_IBM_s001
                                 + 0.785261 w_k1_INTC_s001
                                 + 0.705299 w_k1_KO_s001
                                 + 0.695045 w_k1_MAT_s001
                                 + 0.78487 w_k1_MSFT_s001
                                 + 0.776003 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.943401 w_k1_ADM_s001
                                 + 0.918994 w_k1_AXP_s001
                                 + 1.06899 w_k1_DIS_s001
                                 + 0.916059 w_k1_FDX_s001
                                 + 1.3185 w_k1_IBM_s001
                                 + 1.16545 w_k1_INTC_s001
                                 + 1.26603 w_k1_KO_s001 + 1.45787 w_k1_MAT_s001
                                 + 1.05666 w_k1_MSFT_s001
                                 + 1.12682 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.14776 w_k1_ADM_s001
                                 + 1.5821 w_k1_AXP_s001 + 1.73323 w_k1_DIS_s001
                                 + 1.40739 w_k1_FDX_s001
                                 + 1.31762 w_k1_IBM_s001
                                 + 1.62459 w_k1_INTC_s001
                                 + 1.60002 w_k1_KO_s001 + 1.52851 w_k1_MAT_s001
                                 + 1.43995 w_k1_MSFT_s001
                                 + 1.37498 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.07364 w_k1_ADM_s001
                                 + 1.0903 w_k1_AXP_s001
                                 + 0.810431 w_k1_DIS_s001
                                 + 0.965202 w_k1_FDX_s001
                                 + 1.15045 w_k1_IBM_s001
                                 + 0.963178 w_k1_INTC_s001
                                 + 0.818975 w_k1_KO_s001
                                 + 1.43125 w_k1_MAT_s001
                                 + 0.760245 w_k1_MSFT_s001
                                 + 1.08318 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.03462 w_k1_ADM_s001
                                 + 1.22713 w_k1_AXP_s001 + 1.266 w_k1_DIS_s001
                                 + 1.11142 w_k1_FDX_s001
                                 + 1.60624 w_k1_IBM_s001
                                 + 1.12493 w_k1_INTC_s001
                                 + 1.26256 w_k1_KO_s001 + 1.68348 w_k1_MAT_s001
                                 + 1.1771 w_k1_MSFT_s001 + 1.16824 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.81014 w_k1_ADM_s001
                                 + 0.805357 w_k1_AXP_s001
                                 + 0.807821 w_k1_DIS_s001
                                 + 0.817187 w_k1_FDX_s001
                                 + 0.926178 w_k1_IBM_s001
                                 + 1.11032 w_k1_INTC_s001
                                 + 0.887376 w_k1_KO_s001 + 1.0128 w_k1_MAT_s001
                                 + 0.890648 w_k1_MSFT_s001
                                 + 0.93344 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.20744 w_k1_ADM_s001
                                 + 0.923632 w_k1_AXP_s001
                                 + 1.38804 w_k1_DIS_s001 + 1.2937 w_k1_FDX_s001
                                 + 0.958968 w_k1_IBM_s001
                                 + 1.29724 w_k1_INTC_s001
                                 + 1.24074 w_k1_KO_s001 + 1.16354 w_k1_MAT_s001
                                 + 1.28003 w_k1_MSFT_s001
                                 + 1.14002 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.928364 w_k1_ADM_s001
                                 + 0.995366 w_k1_AXP_s001
                                 + 1.06573 w_k1_DIS_s001 + 1.0994 w_k1_FDX_s001
                                 + 0.851347 w_k1_IBM_s001
                                 + 1.1366 w_k1_INTC_s001
                                 + 0.994118 w_k1_KO_s001
                                 + 1.48379 w_k1_MAT_s001
                                 + 1.12851 w_k1_MSFT_s001
                                 + 1.03245 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.927253 w_k1_ADM_s002
                                 + 0.933616 w_k1_AXP_s002
                                 + 0.923459 w_k1_DIS_s002
                                 + 0.889933 w_k1_FDX_s002
                                 + 1.09907 w_k1_IBM_s002
                                 + 0.904292 w_k1_INTC_s002
                                 + 0.995563 w_k1_KO_s002
                                 + 1.00352 w_k1_MAT_s002
                                 + 0.921953 w_k1_MSFT_s002
                                 + 1.10032 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.36919 w_k1_ADM_s002
                                 + 1.30171 w_k1_AXP_s002
                                 + 1.08289 w_k1_DIS_s002 + 1.1922 w_k1_FDX_s002
                                 + 0.876581 w_k1_IBM_s002
                                 + 0.888538 w_k1_INTC_s002
                                 + 1.00383 w_k1_KO_s002
                                 + 0.927861 w_k1_MAT_s002
                                 + 0.971844 w_k1_MSFT_s002
                                 + 1.1606 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.00666 w_k1_ADM_s002
                                 + 0.863439 w_k1_AXP_s002
                                 + 0.745065 w_k1_DIS_s002
                                 + 0.851544 w_k1_FDX_s002
                                 + 0.741894 w_k1_IBM_s002
                                 + 0.745786 w_k1_INTC_s002
                                 + 0.768956 w_k1_KO_s002
                                 + 0.651398 w_k1_MAT_s002
                                 + 0.775308 w_k1_MSFT_s002
                                 + 0.908704 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.3153 w_k1_ADM_s002
                                 + 1.45499 w_k1_AXP_s002
                                 + 1.61648 w_k1_DIS_s002
                                 + 1.23702 w_k1_FDX_s002
                                 + 1.32331 w_k1_IBM_s002
                                 + 1.32272 w_k1_INTC_s002
                                 + 1.23817 w_k1_KO_s002 + 1.36885 w_k1_MAT_s002
                                 + 1.21077 w_k1_MSFT_s002
                                 + 1.30228 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.784009 w_k1_ADM_s002
                                 + 0.749179 w_k1_AXP_s002
                                 + 0.684209 w_k1_DIS_s002
                                 + 0.681961 w_k1_FDX_s002
                                 + 1.0115 w_k1_IBM_s002
                                 + 0.79126 w_k1_INTC_s002
                                 + 0.95794 w_k1_KO_s002 + 1.00276 w_k1_MAT_s002
                                 + 0.72292 w_k1_MSFT_s002
                                 + 0.93607 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.08041 w_k1_ADM_s002
                                 + 1.02491 w_k1_AXP_s002
                                 + 1.14668 w_k1_DIS_s002
                                 + 1.08152 w_k1_FDX_s002
                                 + 0.845434 w_k1_IBM_s002
                                 + 1.03686 w_k1_INTC_s002
                                 + 1.01732 w_k1_KO_s002 + 1.13514 w_k1_MAT_s002
                                 + 0.996805 w_k1_MSFT_s002
                                 + 1.04834 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.93013 w_k1_ADM_s002
                                 + 1.38143 w_k1_AXP_s002 + 1.0525 w_k1_DIS_s002
                                 + 0.973517 w_k1_FDX_s002
                                 + 1.22834 w_k1_IBM_s002
                                 + 1.12208 w_k1_INTC_s002
                                 + 1.06726 w_k1_KO_s002 + 1.40879 w_k1_MAT_s002
                                 + 0.993872 w_k1_MSFT_s002
                                 + 1.04675 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.715898 w_k1_ADM_s002
                                 + 0.843719 w_k1_AXP_s002
                                 + 0.606923 w_k1_DIS_s002
                                 + 0.764497 w_k1_FDX_s002
                                 + 0.729802 w_k1_IBM_s002
                                 + 0.66706 w_k1_INTC_s002
                                 + 0.713669 w_k1_KO_s002
                                 + 0.789085 w_k1_MAT_s002
                                 + 0.669568 w_k1_MSFT_s002
                                 + 0.858684 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.974127 w_k1_ADM_s003
                                 + 1.19766 w_k1_AXP_s003
                                 + 0.886222 w_k1_DIS_s003
                                 + 1.04237 w_k1_FDX_s003
                                 + 1.01117 w_k1_IBM_s003
                                 + 0.976503 w_k1_INTC_s003
                                 + 0.814306 w_k1_KO_s003
                                 + 0.931893 w_k1_MAT_s003
                                 + 0.863132 w_k1_MSFT_s003
                                 + 1.04067 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.02357 w_k1_ADM_s003
                                 + 1.73993 w_k1_AXP_s003
                                 + 1.55323 w_k1_DIS_s003
                                 + 1.48153 w_k1_FDX_s003
                                 + 0.897184 w_k1_IBM_s003
                                 + 1.57742 w_k1_INTC_s003
                                 + 1.05128 w_k1_KO_s003 + 1.00198 w_k1_MAT_s003
                                 + 1.39399 w_k1_MSFT_s003
                                 + 1.16226 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.08952 w_k1_ADM_s003
                                 + 1.03178 w_k1_AXP_s003
                                 + 1.01933 w_k1_DIS_s003
                                 + 1.02219 w_k1_FDX_s003
                                 + 1.20082 w_k1_IBM_s003
                                 + 1.02894 w_k1_INTC_s003
                                 + 0.938046 w_k1_KO_s003
                                 + 0.981138 w_k1_MAT_s003
                                 + 0.938088 w_k1_MSFT_s003
                                 + 1.0066 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.18674 w_k1_ADM_s003
                                 + 1.1675 w_k1_AXP_s003 + 1.09807 w_k1_DIS_s003
                                 + 1.25363 w_k1_FDX_s003
                                 + 0.750083 w_k1_IBM_s003
                                 + 1.25808 w_k1_INTC_s003
                                 + 0.854501 w_k1_KO_s003
                                 + 0.734699 w_k1_MAT_s003
                                 + 1.07836 w_k1_MSFT_s003
                                 + 1.05553 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.918704 w_k1_ADM_s003
                                 + 0.693985 w_k1_AXP_s003
                                 + 0.602674 w_k1_DIS_s003
                                 + 0.773772 w_k1_FDX_s003
                                 + 0.816005 w_k1_IBM_s003
                                 + 0.848784 w_k1_INTC_s003
                                 + 0.684627 w_k1_KO_s003
                                 + 0.552519 w_k1_MAT_s003
                                 + 0.742718 w_k1_MSFT_s003
                                 + 0.784104 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.34575 w_k1_ADM_s003
                                 + 1.312 w_k1_AXP_s003 + 1.27758 w_k1_DIS_s003
                                 + 1.1224 w_k1_FDX_s003 + 1.89158 w_k1_IBM_s003
                                 + 1.08927 w_k1_INTC_s003
                                 + 1.45082 w_k1_KO_s003 + 1.52287 w_k1_MAT_s003
                                 + 1.05782 w_k1_MSFT_s003
                                 + 1.07141 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.34558 w_k1_ADM_s003
                                 + 1.74813 w_k1_AXP_s003
                                 + 2.14257 w_k1_DIS_s003 + 1.5911 w_k1_FDX_s003
                                 + 1.43279 w_k1_IBM_s003
                                 + 1.42076 w_k1_INTC_s003
                                 + 1.58385 w_k1_KO_s003 + 1.56803 w_k1_MAT_s003
                                 + 1.49867 w_k1_MSFT_s003
                                 + 1.51852 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.883988 w_k1_ADM_s003
                                 + 0.736095 w_k1_AXP_s003
                                 + 0.682473 w_k1_DIS_s003
                                 + 0.763575 w_k1_FDX_s003
                                 + 1.4074 w_k1_IBM_s003
                                 + 0.831178 w_k1_INTC_s003
                                 + 0.906024 w_k1_KO_s003
                                 + 0.926523 w_k1_MAT_s003
                                 + 0.690007 w_k1_MSFT_s003
                                 + 0.959741 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.977423 w_k1_ADM_s004
                                 + 1.35675 w_k1_AXP_s004 + 1.3753 w_k1_DIS_s004
                                 + 1.1484 w_k1_FDX_s004 + 1.61233 w_k1_IBM_s004
                                 + 1.15149 w_k1_INTC_s004
                                 + 1.52855 w_k1_KO_s004 + 1.79992 w_k1_MAT_s004
                                 + 1.06417 w_k1_MSFT_s004
                                 + 1.39136 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.828566 w_k1_ADM_s004
                                 + 1.06678 w_k1_AXP_s004
                                 + 1.12911 w_k1_DIS_s004
                                 + 1.12684 w_k1_FDX_s004
                                 + 0.883306 w_k1_IBM_s004
                                 + 1.17308 w_k1_INTC_s004 + 1.0225 w_k1_KO_s004
                                 + 0.874939 w_k1_MAT_s004
                                 + 0.97096 w_k1_MSFT_s004
                                 + 1.06038 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.90127 w_k1_ADM_s004
                                 + 0.791852 w_k1_AXP_s004
                                 + 0.89828 w_k1_DIS_s004
                                 + 0.90804 w_k1_FDX_s004
                                 + 0.666545 w_k1_IBM_s004
                                 + 1.01776 w_k1_INTC_s004
                                 + 0.885548 w_k1_KO_s004
                                 + 0.719509 w_k1_MAT_s004
                                 + 0.985975 w_k1_MSFT_s004
                                 + 0.909772 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.850588 w_k1_ADM_s004
                                 + 0.753576 w_k1_AXP_s004
                                 + 0.660513 w_k1_DIS_s004
                                 + 0.781899 w_k1_FDX_s004
                                 + 1.20742 w_k1_IBM_s004
                                 + 0.863983 w_k1_INTC_s004
                                 + 0.935637 w_k1_KO_s004
                                 + 1.14468 w_k1_MAT_s004
                                 + 0.789872 w_k1_MSFT_s004
                                 + 0.935369 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.0506 w_k1_ADM_s004
                                 + 1.3229 w_k1_AXP_s004 + 1.30701 w_k1_DIS_s004
                                 + 1.41418 w_k1_FDX_s004
                                 + 0.803679 w_k1_IBM_s004
                                 + 1.54557 w_k1_INTC_s004
                                 + 1.05078 w_k1_KO_s004
                                 + 0.741177 w_k1_MAT_s004
                                 + 1.11765 w_k1_MSFT_s004
                                 + 1.21568 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.04494 w_k1_ADM_s004
                                 + 1.00441 w_k1_AXP_s004
                                 + 0.76213 w_k1_DIS_s004
                                 + 0.96162 w_k1_FDX_s004
                                 + 0.948916 w_k1_IBM_s004
                                 + 0.796609 w_k1_INTC_s004
                                 + 0.849489 w_k1_KO_s004 + 1.1175 w_k1_MAT_s004
                                 + 0.788035 w_k1_MSFT_s004
                                 + 0.912509 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.33822 w_k1_ADM_s004
                                 + 1.43185 w_k1_AXP_s004
                                 + 1.37276 w_k1_DIS_s004
                                 + 1.29153 w_k1_FDX_s004
                                 + 1.19843 w_k1_IBM_s004
                                 + 1.1468 w_k1_INTC_s004 + 1.283 w_k1_KO_s004
                                 + 1.5653 w_k1_MAT_s004
                                 + 1.11123 w_k1_MSFT_s004
                                 + 1.11071 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.940218 w_k1_ADM_s004
                                 + 0.897667 w_k1_AXP_s004
                                 + 1.13916 w_k1_DIS_s004
                                 + 1.04761 w_k1_FDX_s004
                                 + 1.11373 w_k1_IBM_s004
                                 + 1.19509 w_k1_INTC_s004
                                 + 1.00673 w_k1_KO_s004
                                 + 0.801816 w_k1_MAT_s004
                                 + 1.08908 w_k1_MSFT_s004
                                 + 1.05466 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.01989 w_k1_ADM_s005
                                 + 1.09543 w_k1_AXP_s005
                                 + 1.06971 w_k1_DIS_s005
                                 + 1.13426 w_k1_FDX_s005
                                 + 1.18938 w_k1_IBM_s005
                                 + 1.01305 w_k1_INTC_s005
                                 + 1.09536 w_k1_KO_s005 + 1.5938 w_k1_MAT_s005
                                 + 1.06954 w_k1_MSFT_s005
                                 + 1.07525 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.017 w_k1_ADM_s005
                                 + 0.817991 w_k1_AXP_s005
                                 + 0.89606 w_k1_DIS_s005
                                 + 0.952858 w_k1_FDX_s005
                                 + 0.746575 w_k1_IBM_s005
                                 + 0.927673 w_k1_INTC_s005
                                 + 0.914044 w_k1_KO_s005
                                 + 0.875103 w_k1_MAT_s005
                                 + 1.00473 w_k1_MSFT_s005
                                 + 0.930435 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.29803 w_k1_ADM_s005
                                 + 0.97046 w_k1_AXP_s005
                                 + 1.33311 w_k1_DIS_s005
                                 + 1.26072 w_k1_FDX_s005
                                 + 0.833283 w_k1_IBM_s005
                                 + 1.11233 w_k1_INTC_s005
                                 + 1.09529 w_k1_KO_s005 + 1.04736 w_k1_MAT_s005
                                 + 1.25876 w_k1_MSFT_s005
                                 + 1.15388 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.725208 w_k1_ADM_s005
                                 + 0.65718 w_k1_AXP_s005
                                 + 0.560982 w_k1_DIS_s005
                                 + 0.693869 w_k1_FDX_s005
                                 + 0.807625 w_k1_IBM_s005
                                 + 0.909181 w_k1_INTC_s005
                                 + 0.781884 w_k1_KO_s005 + 0.8426 w_k1_MAT_s005
                                 + 0.76138 w_k1_MSFT_s005
                                 + 0.797406 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.905846 w_k1_ADM_s005
                                 + 0.891504 w_k1_AXP_s005
                                 + 0.902252 w_k1_DIS_s005
                                 + 0.937915 w_k1_FDX_s005
                                 + 1.1524 w_k1_IBM_s005
                                 + 1.17201 w_k1_INTC_s005
                                 + 0.948325 w_k1_KO_s005
                                 + 1.25425 w_k1_MAT_s005
                                 + 0.991937 w_k1_MSFT_s005
                                 + 0.991522 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.938062 w_k1_ADM_s005
                                 + 1.13563 w_k1_AXP_s005
                                 + 0.977338 w_k1_DIS_s005
                                 + 0.982016 w_k1_FDX_s005
                                 + 1.49934 w_k1_IBM_s005
                                 + 1.09732 w_k1_INTC_s005
                                 + 1.14461 w_k1_KO_s005 + 1.67436 w_k1_MAT_s005
                                 + 0.997831 w_k1_MSFT_s005
                                 + 1.12434 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.15724 w_k1_ADM_s005
                                 + 1.37506 w_k1_AXP_s005
                                 + 1.14909 w_k1_DIS_s005
                                 + 1.29235 w_k1_FDX_s005
                                 + 0.863115 w_k1_IBM_s005
                                 + 1.07502 w_k1_INTC_s005
                                 + 0.908903 w_k1_KO_s005
                                 + 1.28837 w_k1_MAT_s005
                                 + 1.23477 w_k1_MSFT_s005
                                 + 1.10196 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.37674 w_k1_ADM_s005
                                 + 1.24891 w_k1_AXP_s005
                                 + 1.76274 w_k1_DIS_s005
                                 + 1.45358 w_k1_FDX_s005
                                 + 1.21708 w_k1_IBM_s005
                                 + 1.2706 w_k1_INTC_s005 + 1.34761 w_k1_KO_s005
                                 + 1.62364 w_k1_MAT_s005
                                 + 1.34301 w_k1_MSFT_s005 + 1.2336 w_k1_PG_s005
                                 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.910004 w_k1_ADM_s006
                                 + 0.954712 w_k1_AXP_s006
                                 + 1.28475 w_k1_DIS_s006
                                 + 1.12635 w_k1_FDX_s006
                                 + 1.08769 w_k1_IBM_s006
                                 + 1.15382 w_k1_INTC_s006
                                 + 1.15592 w_k1_KO_s006 + 1.21439 w_k1_MAT_s006
                                 + 1.36987 w_k1_MSFT_s006
                                 + 1.14712 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.85911 w_k1_ADM_s006
                                 + 0.757457 w_k1_AXP_s006
                                 + 0.848505 w_k1_DIS_s006
                                 + 0.885709 w_k1_FDX_s006
                                 + 1.10843 w_k1_IBM_s006
                                 + 0.746025 w_k1_INTC_s006
                                 + 0.902361 w_k1_KO_s006
                                 + 1.36728 w_k1_MAT_s006
                                 + 0.945253 w_k1_MSFT_s006
                                 + 0.98008 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.38575 w_k1_ADM_s006
                                 + 1.20138 w_k1_AXP_s006
                                 + 1.94081 w_k1_DIS_s006
                                 + 1.46504 w_k1_FDX_s006
                                 + 1.05981 w_k1_IBM_s006
                                 + 1.10252 w_k1_INTC_s006
                                 + 1.29956 w_k1_KO_s006 + 1.41896 w_k1_MAT_s006
                                 + 1.37634 w_k1_MSFT_s006
                                 + 1.22889 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.805915 w_k1_ADM_s006
                                 + 0.742885 w_k1_AXP_s006
                                 + 0.80809 w_k1_DIS_s006
                                 + 0.791272 w_k1_FDX_s006
                                 + 1.37733 w_k1_IBM_s006
                                 + 0.968172 w_k1_INTC_s006
                                 + 1.01172 w_k1_KO_s006 + 1.21694 w_k1_MAT_s006
                                 + 0.978324 w_k1_MSFT_s006
                                 + 0.971991 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.20734 w_k1_ADM_s006
                                 + 1.32621 w_k1_AXP_s006
                                 + 0.939254 w_k1_DIS_s006
                                 + 1.12959 w_k1_FDX_s006
                                 + 1.04487 w_k1_IBM_s006
                                 + 1.22541 w_k1_INTC_s006
                                 + 0.94136 w_k1_KO_s006 + 1.49549 w_k1_MAT_s006
                                 + 1.10884 w_k1_MSFT_s006
                                 + 1.11242 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.873141 w_k1_ADM_s006
                                 + 0.858797 w_k1_AXP_s006
                                 + 0.831532 w_k1_DIS_s006
                                 + 0.892678 w_k1_FDX_s006
                                 + 0.794519 w_k1_IBM_s006
                                 + 0.851645 w_k1_INTC_s006
                                 + 0.815199 w_k1_KO_s006
                                 + 0.836421 w_k1_MAT_s006
                                 + 0.874287 w_k1_MSFT_s006
                                 + 0.850963 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.44115 w_k1_ADM_s006
                                 + 1.25898 w_k1_AXP_s006
                                 + 1.21258 w_k1_DIS_s006
                                 + 1.19885 w_k1_FDX_s006
                                 + 1.52749 w_k1_IBM_s006
                                 + 1.05843 w_k1_INTC_s006
                                 + 1.19297 w_k1_KO_s006 + 1.63799 w_k1_MAT_s006
                                 + 1.17306 w_k1_MSFT_s006
                                 + 1.14715 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.2691 w_k1_ADM_s006
                                 + 1.22436 w_k1_AXP_s006
                                 + 1.38992 w_k1_DIS_s006
                                 + 1.30452 w_k1_FDX_s006
                                 + 0.85325 w_k1_IBM_s006
                                 + 1.15455 w_k1_INTC_s006
                                 + 1.03518 w_k1_KO_s006
                                 + 0.982993 w_k1_MAT_s006
                                 + 1.2071 w_k1_MSFT_s006 + 1.11472 w_k1_PG_s006
                                 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.869836 w_k1_ADM_s007
                                 + 1.44637 w_k1_AXP_s007
                                 + 1.52622 w_k1_DIS_s007
                                 + 1.20309 w_k1_FDX_s007
                                 + 1.64258 w_k1_IBM_s007
                                 + 1.44322 w_k1_INTC_s007 + 1.3385 w_k1_KO_s007
                                 + 1.88908 w_k1_MAT_s007
                                 + 1.07144 w_k1_MSFT_s007
                                 + 1.25187 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.06725 w_k1_ADM_s007
                                 + 1.00409 w_k1_AXP_s007
                                 + 1.03128 w_k1_DIS_s007
                                 + 0.986778 w_k1_FDX_s007
                                 + 0.799324 w_k1_IBM_s007
                                 + 1.17345 w_k1_INTC_s007
                                 + 0.83167 w_k1_KO_s007
                                 + 0.812062 w_k1_MAT_s007
                                 + 0.955582 w_k1_MSFT_s007
                                 + 0.996573 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.761691 w_k1_ADM_s007
                                 + 0.84814 w_k1_AXP_s007
                                 + 0.92128 w_k1_DIS_s007
                                 + 0.756177 w_k1_FDX_s007
                                 + 1.1493 w_k1_IBM_s007
                                 + 1.01157 w_k1_INTC_s007
                                 + 0.981194 w_k1_KO_s007
                                 + 1.10144 w_k1_MAT_s007
                                 + 0.826335 w_k1_MSFT_s007
                                 + 0.876314 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.773734 w_k1_ADM_s007
                                 + 0.702124 w_k1_AXP_s007
                                 + 0.705988 w_k1_DIS_s007
                                 + 0.810735 w_k1_FDX_s007
                                 + 0.710074 w_k1_IBM_s007
                                 + 0.848603 w_k1_INTC_s007
                                 + 0.710201 w_k1_KO_s007
                                 + 0.766734 w_k1_MAT_s007
                                 + 0.821181 w_k1_MSFT_s007
                                 + 0.778491 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.02885 w_k1_ADM_s007
                                 + 1.23174 w_k1_AXP_s007
                                 + 1.01519 w_k1_DIS_s007
                                 + 0.914612 w_k1_FDX_s007
                                 + 1.31373 w_k1_IBM_s007
                                 + 0.968116 w_k1_INTC_s007
                                 + 1.02374 w_k1_KO_s007 + 1.25161 w_k1_MAT_s007
                                 + 0.89084 w_k1_MSFT_s007
                                 + 0.904077 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.695715 w_k1_ADM_s007
                                 + 0.776672 w_k1_AXP_s007
                                 + 0.59956 w_k1_DIS_s007
                                 + 0.728693 w_k1_FDX_s007
                                 + 0.900318 w_k1_IBM_s007
                                 + 0.905307 w_k1_INTC_s007
                                 + 0.70643 w_k1_KO_s007
                                 + 0.815573 w_k1_MAT_s007
                                 + 0.60954 w_k1_MSFT_s007
                                 + 0.774246 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.2612 w_k1_ADM_s007
                                 + 1.11695 w_k1_AXP_s007
                                 + 1.26902 w_k1_DIS_s007
                                 + 1.25461 w_k1_FDX_s007
                                 + 1.01938 w_k1_IBM_s007
                                 + 1.33215 w_k1_INTC_s007
                                 + 1.04841 w_k1_KO_s007 + 1.16291 w_k1_MAT_s007
                                 + 1.16947 w_k1_MSFT_s007
                                 + 1.04912 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.800219 w_k1_ADM_s007
                                 + 0.932409 w_k1_AXP_s007
                                 + 0.937059 w_k1_DIS_s007
                                 + 0.86629 w_k1_FDX_s007
                                 + 0.97599 w_k1_IBM_s007
                                 + 0.913869 w_k1_INTC_s007
                                 + 0.762357 w_k1_KO_s007
                                 + 0.909819 w_k1_MAT_s007
                                 + 0.898442 w_k1_MSFT_s007
                                 + 0.860435 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.09238 w_k1_ADM_s010
                                 + 1.20112 w_k1_AXP_s010
                                 + 1.06086 w_k1_DIS_s010
                                 + 1.08811 w_k1_FDX_s010
                                 + 0.708091 w_k1_IBM_s010
                                 + 1.11899 w_k1_INTC_s010
                                 + 0.918115 w_k1_KO_s010
                                 + 0.663616 w_k1_MAT_s010
                                 + 1.00623 w_k1_MSFT_s010
                                 + 1.10261 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.733003 w_k1_ADM_s010
                                 + 0.686276 w_k1_AXP_s010
                                 + 0.638714 w_k1_DIS_s010
                                 + 0.783046 w_k1_FDX_s010
                                 + 0.74404 w_k1_IBM_s010
                                 + 0.703823 w_k1_INTC_s010
                                 + 0.654919 w_k1_KO_s010
                                 + 0.502099 w_k1_MAT_s010
                                 + 0.65488 w_k1_MSFT_s010
                                 + 0.805912 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.857774 w_k1_ADM_s010
                                 + 1.30836 w_k1_AXP_s010
                                 + 1.07073 w_k1_DIS_s010
                                 + 1.08691 w_k1_FDX_s010
                                 + 0.925155 w_k1_IBM_s010
                                 + 0.97744 w_k1_INTC_s010
                                 + 1.01188 w_k1_KO_s010 + 1.06105 w_k1_MAT_s010
                                 + 1.00016 w_k1_MSFT_s010
                                 + 1.08903 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.877167 w_k1_ADM_s010
                                 + 0.901192 w_k1_AXP_s010
                                 + 1.24145 w_k1_DIS_s010
                                 + 0.984413 w_k1_FDX_s010
                                 + 0.845659 w_k1_IBM_s010
                                 + 0.943421 w_k1_INTC_s010
                                 + 0.933553 w_k1_KO_s010
                                 + 0.779812 w_k1_MAT_s010
                                 + 0.903479 w_k1_MSFT_s010
                                 + 1.02157 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.27757 w_k1_ADM_s010
                                 + 1.47557 w_k1_AXP_s010
                                 + 1.57413 w_k1_DIS_s010
                                 + 1.23905 w_k1_FDX_s010
                                 + 1.40106 w_k1_IBM_s010
                                 + 1.16558 w_k1_INTC_s010
                                 + 1.35331 w_k1_KO_s010 + 1.48284 w_k1_MAT_s010
                                 + 1.26608 w_k1_MSFT_s010
                                 + 1.28207 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.676515 w_k1_ADM_s010
                                 + 0.926792 w_k1_AXP_s010
                                 + 1.05861 w_k1_DIS_s010
                                 + 0.892819 w_k1_FDX_s010
                                 + 1.35308 w_k1_IBM_s010
                                 + 1.15313 w_k1_INTC_s010
                                 + 1.15594 w_k1_KO_s010 + 1.24718 w_k1_MAT_s010
                                 + 1.01187 w_k1_MSFT_s010
                                 + 1.15104 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.987661 w_k1_ADM_s010
                                 + 0.944024 w_k1_AXP_s010
                                 + 1.14386 w_k1_DIS_s010
                                 + 1.01789 w_k1_FDX_s010
                                 + 1.05627 w_k1_IBM_s010
                                 + 1.06135 w_k1_INTC_s010
                                 + 1.22283 w_k1_KO_s010 + 0.67693 w_k1_MAT_s010
                                 + 1.08213 w_k1_MSFT_s010
                                 + 1.12225 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.757923 w_k1_ADM_s010
                                 + 0.740354 w_k1_AXP_s010
                                 + 0.728293 w_k1_DIS_s010
                                 + 0.709055 w_k1_FDX_s010
                                 + 0.982344 w_k1_IBM_s010
                                 + 0.986067 w_k1_INTC_s010
                                 + 1.00703 w_k1_KO_s010
                                 + 0.777056 w_k1_MAT_s010
                                 + 0.83621 w_k1_MSFT_s010
                                 + 0.998657 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.844059 w_k1_ADM_s011
                                 + 0.876863 w_k1_AXP_s011
                                 + 1.08263 w_k1_DIS_s011
                                 + 0.813379 w_k1_FDX_s011
                                 + 1.31419 w_k1_IBM_s011
                                 + 0.863166 w_k1_INTC_s011
                                 + 1.07806 w_k1_KO_s011 + 1.18132 w_k1_MAT_s011
                                 + 1.03618 w_k1_MSFT_s011
                                 + 0.977718 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.839229 w_k1_ADM_s011
                                 + 0.822401 w_k1_AXP_s011
                                 + 0.845458 w_k1_DIS_s011
                                 + 0.85457 w_k1_FDX_s011
                                 + 0.801472 w_k1_IBM_s011
                                 + 1.03923 w_k1_INTC_s011
                                 + 0.959302 w_k1_KO_s011
                                 + 0.784152 w_k1_MAT_s011
                                 + 1.04748 w_k1_MSFT_s011
                                 + 1.04579 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.123 w_k1_ADM_s011
                                 + 1.5821 w_k1_AXP_s011 + 1.79157 w_k1_DIS_s011
                                 + 1.45154 w_k1_FDX_s011
                                 + 1.04373 w_k1_IBM_s011
                                 + 1.0343 w_k1_INTC_s011 + 1.19049 w_k1_KO_s011
                                 + 0.924123 w_k1_MAT_s011
                                 + 1.28994 w_k1_MSFT_s011
                                 + 1.17854 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.14525 w_k1_ADM_s011
                                 + 1.32184 w_k1_AXP_s011 + 1.1443 w_k1_DIS_s011
                                 + 1.09139 w_k1_FDX_s011
                                 + 1.11516 w_k1_IBM_s011
                                 + 1.00835 w_k1_INTC_s011
                                 + 1.10857 w_k1_KO_s011
                                 + 0.960017 w_k1_MAT_s011
                                 + 1.05451 w_k1_MSFT_s011
                                 + 1.11769 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.815632 w_k1_ADM_s011
                                 + 0.690023 w_k1_AXP_s011
                                 + 0.614122 w_k1_DIS_s011
                                 + 0.678428 w_k1_FDX_s011
                                 + 0.797283 w_k1_IBM_s011
                                 + 0.859892 w_k1_INTC_s011
                                 + 0.841124 w_k1_KO_s011
                                 + 0.492475 w_k1_MAT_s011
                                 + 0.831556 w_k1_MSFT_s011
                                 + 0.752408 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.09211 w_k1_ADM_s011
                                 + 1.70849 w_k1_AXP_s011
                                 + 1.15997 w_k1_DIS_s011
                                 + 1.09941 w_k1_FDX_s011
                                 + 1.65332 w_k1_IBM_s011
                                 + 1.05134 w_k1_INTC_s011
                                 + 1.26244 w_k1_KO_s011 + 1.65184 w_k1_MAT_s011
                                 + 1.06677 w_k1_MSFT_s011
                                 + 1.10412 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.0136 w_k1_ADM_s011
                                 + 1.13032 w_k1_AXP_s011
                                 + 1.04374 w_k1_DIS_s011
                                 + 1.10835 w_k1_FDX_s011
                                 + 0.897955 w_k1_IBM_s011
                                 + 1.10443 w_k1_INTC_s011
                                 + 1.02757 w_k1_KO_s011
                                 + 0.888471 w_k1_MAT_s011
                                 + 1.08163 w_k1_MSFT_s011 + 1.0123 w_k1_PG_s011
                                 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.900188 w_k1_ADM_s011
                                 + 0.924147 w_k1_AXP_s011
                                 + 0.80402 w_k1_DIS_s011
                                 + 0.922741 w_k1_FDX_s011
                                 + 0.612738 w_k1_IBM_s011
                                 + 0.917252 w_k1_INTC_s011
                                 + 0.863809 w_k1_KO_s011
                                 + 0.57446 w_k1_MAT_s011
                                 + 0.986375 w_k1_MSFT_s011
                                 + 0.918512 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.864637 w_k1_ADM_s012
                                 + 0.938886 w_k1_AXP_s012
                                 + 0.882733 w_k1_DIS_s012
                                 + 0.870814 w_k1_FDX_s012
                                 + 1.00976 w_k1_IBM_s012
                                 + 0.968709 w_k1_INTC_s012
                                 + 1.0415 w_k1_KO_s012 + 0.770246 w_k1_MAT_s012
                                 + 0.873436 w_k1_MSFT_s012
                                 + 0.882276 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.05416 w_k1_ADM_s012
                                 + 1.03347 w_k1_AXP_s012 + 1.0303 w_k1_DIS_s012
                                 + 1.08995 w_k1_FDX_s012
                                 + 0.697701 w_k1_IBM_s012
                                 + 1.04803 w_k1_INTC_s012
                                 + 0.953415 w_k1_KO_s012
                                 + 0.943955 w_k1_MAT_s012
                                 + 1.12959 w_k1_MSFT_s012
                                 + 0.92252 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.773692 w_k1_ADM_s012
                                 + 0.829402 w_k1_AXP_s012
                                 + 0.710485 w_k1_DIS_s012
                                 + 0.739737 w_k1_FDX_s012
                                 + 1.25085 w_k1_IBM_s012
                                 + 0.878943 w_k1_INTC_s012
                                 + 1.04195 w_k1_KO_s012 + 1.51241 w_k1_MAT_s012
                                 + 0.778355 w_k1_MSFT_s012
                                 + 0.925755 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.29738 w_k1_ADM_s012
                                 + 1.40436 w_k1_AXP_s012
                                 + 1.47489 w_k1_DIS_s012
                                 + 1.33372 w_k1_FDX_s012
                                 + 1.23834 w_k1_IBM_s012
                                 + 1.05009 w_k1_INTC_s012
                                 + 1.31849 w_k1_KO_s012 + 1.65189 w_k1_MAT_s012
                                 + 1.3712 w_k1_MSFT_s012 + 1.20049 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.928045 w_k1_ADM_s012
                                 + 1.19669 w_k1_AXP_s012
                                 + 1.21595 w_k1_DIS_s012
                                 + 1.02884 w_k1_FDX_s012
                                 + 1.66335 w_k1_IBM_s012
                                 + 1.08229 w_k1_INTC_s012
                                 + 1.54839 w_k1_KO_s012 + 1.77882 w_k1_MAT_s012
                                 + 1.19416 w_k1_MSFT_s012
                                 + 1.18394 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.12457 w_k1_ADM_s012
                                 + 1.10007 w_k1_AXP_s012
                                 + 1.11113 w_k1_DIS_s012
                                 + 0.963323 w_k1_FDX_s012
                                 + 1.23472 w_k1_IBM_s012
                                 + 1.02322 w_k1_INTC_s012
                                 + 1.15064 w_k1_KO_s012 + 1.37722 w_k1_MAT_s012
                                 + 1.00659 w_k1_MSFT_s012
                                 + 1.03754 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.974259 w_k1_ADM_s012
                                 + 0.898915 w_k1_AXP_s012
                                 + 1.09095 w_k1_DIS_s012
                                 + 1.00657 w_k1_FDX_s012
                                 + 0.884777 w_k1_IBM_s012
                                 + 1.09031 w_k1_INTC_s012
                                 + 1.04942 w_k1_KO_s012
                                 + 0.990397 w_k1_MAT_s012
                                 + 1.11689 w_k1_MSFT_s012
                                 + 1.01594 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.917549 w_k1_ADM_s012
                                 + 0.761761 w_k1_AXP_s012
                                 + 0.671976 w_k1_DIS_s012
                                 + 0.8398 w_k1_FDX_s012
                                 + 0.582487 w_k1_IBM_s012
                                 + 0.857258 w_k1_INTC_s012
                                 + 0.739012 w_k1_KO_s012
                                 + 0.846907 w_k1_MAT_s012
                                 + 0.923393 w_k1_MSFT_s012
                                 + 0.786155 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.830435 w_k1_ADM_s013
                                 + 0.882833 w_k1_AXP_s013
                                 + 0.691991 w_k1_DIS_s013
                                 + 0.913105 w_k1_FDX_s013
                                 + 0.7185 w_k1_IBM_s013
                                 + 0.871016 w_k1_INTC_s013
                                 + 0.645011 w_k1_KO_s013
                                 + 0.764025 w_k1_MAT_s013
                                 + 0.674942 w_k1_MSFT_s013
                                 + 0.889723 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.13018 w_k1_ADM_s013
                                 + 1.08972 w_k1_AXP_s013
                                 + 0.885933 w_k1_DIS_s013
                                 + 1.06775 w_k1_FDX_s013
                                 + 0.901738 w_k1_IBM_s013
                                 + 1.19624 w_k1_INTC_s013
                                 + 0.697869 w_k1_KO_s013
                                 + 1.17682 w_k1_MAT_s013
                                 + 0.887147 w_k1_MSFT_s013
                                 + 0.995247 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.42688 w_k1_ADM_s013
                                 + 1.22102 w_k1_AXP_s013
                                 + 1.44327 w_k1_DIS_s013
                                 + 1.40606 w_k1_FDX_s013
                                 + 0.930141 w_k1_IBM_s013
                                 + 1.70353 w_k1_INTC_s013
                                 + 0.974635 w_k1_KO_s013
                                 + 1.00694 w_k1_MAT_s013
                                 + 1.22907 w_k1_MSFT_s013
                                 + 1.12718 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.891823 w_k1_ADM_s013
                                 + 1.02846 w_k1_AXP_s013
                                 + 0.987686 w_k1_DIS_s013
                                 + 0.955855 w_k1_FDX_s013
                                 + 1.40095 w_k1_IBM_s013
                                 + 0.936528 w_k1_INTC_s013
                                 + 0.899125 w_k1_KO_s013
                                 + 1.35174 w_k1_MAT_s013
                                 + 0.860734 w_k1_MSFT_s013
                                 + 0.983591 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.0647 w_k1_ADM_s013
                                 + 1.22221 w_k1_AXP_s013
                                 + 1.33076 w_k1_DIS_s013
                                 + 1.18023 w_k1_FDX_s013
                                 + 1.56204 w_k1_IBM_s013
                                 + 1.12607 w_k1_INTC_s013
                                 + 1.10152 w_k1_KO_s013 + 1.49497 w_k1_MAT_s013
                                 + 1.02138 w_k1_MSFT_s013
                                 + 1.19654 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.846985 w_k1_ADM_s013
                                 + 0.672158 w_k1_AXP_s013
                                 + 0.738159 w_k1_DIS_s013
                                 + 0.791288 w_k1_FDX_s013
                                 + 0.887068 w_k1_IBM_s013
                                 + 1.01972 w_k1_INTC_s013
                                 + 0.73317 w_k1_KO_s013
                                 + 0.884068 w_k1_MAT_s013
                                 + 0.796827 w_k1_MSFT_s013
                                 + 0.903024 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.948178 w_k1_ADM_s013
                                 + 1.119 w_k1_AXP_s013 + 0.930169 w_k1_DIS_s013
                                 + 1.10444 w_k1_FDX_s013
                                 + 0.840647 w_k1_IBM_s013
                                 + 1.08717 w_k1_INTC_s013
                                 + 0.812515 w_k1_KO_s013
                                 + 1.08492 w_k1_MAT_s013
                                 + 0.886009 w_k1_MSFT_s013
                                 + 0.986088 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.1932 w_k1_ADM_s013
                                 + 0.833159 w_k1_AXP_s013
                                 + 1.20428 w_k1_DIS_s013
                                 + 1.09368 w_k1_FDX_s013
                                 + 0.836024 w_k1_IBM_s013
                                 + 1.09659 w_k1_INTC_s013
                                 + 0.875509 w_k1_KO_s013
                                 + 0.889045 w_k1_MAT_s013
                                 + 1.03803 w_k1_MSFT_s013 + 0.8942 w_k1_PG_s013
                                 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.36013 w_k1_ADM_s014
                                 + 1.27799 w_k1_AXP_s014
                                 + 1.65121 w_k1_DIS_s014 + 1.4034 w_k1_FDX_s014
                                 + 1.19252 w_k1_IBM_s014
                                 + 1.39562 w_k1_INTC_s014
                                 + 1.37281 w_k1_KO_s014 + 1.53996 w_k1_MAT_s014
                                 + 1.38916 w_k1_MSFT_s014
                                 + 1.22735 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.985654 w_k1_ADM_s014
                                 + 1.30241 w_k1_AXP_s014
                                 + 1.48018 w_k1_DIS_s014
                                 + 1.13907 w_k1_FDX_s014
                                 + 1.81633 w_k1_IBM_s014
                                 + 1.4151 w_k1_INTC_s014 + 1.29742 w_k1_KO_s014
                                 + 2.00603 w_k1_MAT_s014
                                 + 1.32466 w_k1_MSFT_s014
                                 + 1.23014 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.974353 w_k1_ADM_s014
                                 + 0.733212 w_k1_AXP_s014
                                 + 0.705399 w_k1_DIS_s014
                                 + 0.871364 w_k1_FDX_s014
                                 + 0.705447 w_k1_IBM_s014
                                 + 0.930266 w_k1_INTC_s014
                                 + 0.678308 w_k1_KO_s014
                                 + 0.551664 w_k1_MAT_s014
                                 + 0.966934 w_k1_MSFT_s014
                                 + 0.841509 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.964644 w_k1_ADM_s014
                                 + 1.05003 w_k1_AXP_s014
                                 + 1.28456 w_k1_DIS_s014
                                 + 1.03423 w_k1_FDX_s014
                                 + 1.03965 w_k1_IBM_s014
                                 + 0.98649 w_k1_INTC_s014
                                 + 0.955625 w_k1_KO_s014
                                 + 1.30888 w_k1_MAT_s014
                                 + 1.0734 w_k1_MSFT_s014
                                 + 0.975007 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.925088 w_k1_ADM_s014
                                 + 0.940914 w_k1_AXP_s014
                                 + 0.999596 w_k1_DIS_s014
                                 + 1.02624 w_k1_FDX_s014
                                 + 0.930864 w_k1_IBM_s014
                                 + 1.04692 w_k1_INTC_s014
                                 + 0.813111 w_k1_KO_s014
                                 + 1.01687 w_k1_MAT_s014
                                 + 1.1249 w_k1_MSFT_s014
                                 + 0.972105 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.804547 w_k1_ADM_s014
                                 + 0.816984 w_k1_AXP_s014
                                 + 0.81117 w_k1_DIS_s014
                                 + 0.804568 w_k1_FDX_s014
                                 + 1.06618 w_k1_IBM_s014
                                 + 0.900813 w_k1_INTC_s014
                                 + 0.943981 w_k1_KO_s014 + 1.0702 w_k1_MAT_s014
                                 + 0.929449 w_k1_MSFT_s014
                                 + 0.884907 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.995405 w_k1_ADM_s014
                                 + 1.17899 w_k1_AXP_s014
                                 + 1.03552 w_k1_DIS_s014 + 1.227 w_k1_FDX_s014
                                 + 0.681683 w_k1_IBM_s014
                                 + 1.22206 w_k1_INTC_s014
                                 + 0.77057 w_k1_KO_s014 + 0.68264 w_k1_MAT_s014
                                 + 1.07697 w_k1_MSFT_s014 + 1.0649 w_k1_PG_s014
                                 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.878107 w_k1_ADM_s014
                                 + 1.03206 w_k1_AXP_s014
                                 + 0.940486 w_k1_DIS_s014
                                 + 1.05505 w_k1_FDX_s014
                                 + 1.33168 w_k1_IBM_s014
                                 + 1.04184 w_k1_INTC_s014
                                 + 0.983674 w_k1_KO_s014
                                 + 1.49865 w_k1_MAT_s014
                                 + 1.03182 w_k1_MSFT_s014
                                 + 1.11658 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.883384 w_k1_ADM_s015
                                 + 0.773277 w_k1_AXP_s015
                                 + 0.725157 w_k1_DIS_s015
                                 + 0.971592 w_k1_FDX_s015
                                 + 0.656417 w_k1_IBM_s015
                                 + 0.81574 w_k1_INTC_s015
                                 + 0.756465 w_k1_KO_s015
                                 + 0.632364 w_k1_MAT_s015
                                 + 0.86923 w_k1_MSFT_s015
                                 + 0.859666 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.05043 w_k1_ADM_s015
                                 + 0.880953 w_k1_AXP_s015
                                 + 0.879876 w_k1_DIS_s015
                                 + 1.0293 w_k1_FDX_s015
                                 + 0.797768 w_k1_IBM_s015
                                 + 1.01352 w_k1_INTC_s015
                                 + 0.884151 w_k1_KO_s015
                                 + 0.631833 w_k1_MAT_s015
                                 + 0.994772 w_k1_MSFT_s015
                                 + 0.870178 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.02117 w_k1_ADM_s015
                                 + 1.19603 w_k1_AXP_s015
                                 + 1.61139 w_k1_DIS_s015
                                 + 1.30194 w_k1_FDX_s015
                                 + 1.13245 w_k1_IBM_s015
                                 + 1.47426 w_k1_INTC_s015 + 1.3507 w_k1_KO_s015
                                 + 1.0742 w_k1_MAT_s015
                                 + 1.20369 w_k1_MSFT_s015
                                 + 1.07059 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.985437 w_k1_ADM_s015
                                 + 0.998292 w_k1_AXP_s015
                                 + 0.927853 w_k1_DIS_s015
                                 + 0.948235 w_k1_FDX_s015
                                 + 1.21363 w_k1_IBM_s015
                                 + 0.731369 w_k1_INTC_s015
                                 + 1.03566 w_k1_KO_s015 + 1.34867 w_k1_MAT_s015
                                 + 0.843219 w_k1_MSFT_s015
                                 + 0.967766 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.735109 w_k1_ADM_s015
                                 + 0.627037 w_k1_AXP_s015
                                 + 0.548084 w_k1_DIS_s015
                                 + 0.665692 w_k1_FDX_s015
                                 + 1.0429 w_k1_IBM_s015
                                 + 0.851853 w_k1_INTC_s015
                                 + 0.832785 w_k1_KO_s015
                                 + 0.84847 w_k1_MAT_s015
                                 + 0.71094 w_k1_MSFT_s015
                                 + 0.804231 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.32254 w_k1_ADM_s015
                                 + 1.46987 w_k1_AXP_s015
                                 + 1.80835 w_k1_DIS_s015
                                 + 1.43701 w_k1_FDX_s015
                                 + 1.59599 w_k1_IBM_s015
                                 + 1.38052 w_k1_INTC_s015
                                 + 1.59842 w_k1_KO_s015 + 1.68972 w_k1_MAT_s015
                                 + 1.24555 w_k1_MSFT_s015
                                 + 1.20485 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.865424 w_k1_ADM_s015
                                 + 0.740626 w_k1_AXP_s015
                                 + 0.919883 w_k1_DIS_s015
                                 + 0.902946 w_k1_FDX_s015
                                 + 1.43661 w_k1_IBM_s015
                                 + 0.858313 w_k1_INTC_s015
                                 + 1.23659 w_k1_KO_s015 + 1.05789 w_k1_MAT_s015
                                 + 0.815871 w_k1_MSFT_s015
                                 + 0.907045 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.03785 w_k1_ADM_s015
                                 + 0.874176 w_k1_AXP_s015
                                 + 1.08284 w_k1_DIS_s015 + 1.038 w_k1_FDX_s015
                                 + 0.99299 w_k1_IBM_s015
                                 + 1.10405 w_k1_INTC_s015 + 1.0741 w_k1_KO_s015
                                 + 0.834061 w_k1_MAT_s015
                                 + 0.976631 w_k1_MSFT_s015
                                 + 1.00446 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.00382 w_k1_ADM_s016
                                 + 0.871701 w_k1_AXP_s016
                                 + 0.960319 w_k1_DIS_s016
                                 + 0.974984 w_k1_FDX_s016
                                 + 1.04479 w_k1_IBM_s016
                                 + 0.945782 w_k1_INTC_s016
                                 + 0.966765 w_k1_KO_s016
                                 + 0.875758 w_k1_MAT_s016
                                 + 1.0234 w_k1_MSFT_s016
                                 + 0.991028 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.75089 w_k1_ADM_s016
                                 + 0.857933 w_k1_AXP_s016
                                 + 0.918988 w_k1_DIS_s016
                                 + 0.782771 w_k1_FDX_s016
                                 + 1.29039 w_k1_IBM_s016
                                 + 0.866661 w_k1_INTC_s016
                                 + 1.03324 w_k1_KO_s016 + 1.0986 w_k1_MAT_s016
                                 + 0.824947 w_k1_MSFT_s016
                                 + 0.8475 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.11826 w_k1_ADM_s016
                                 + 1.32635 w_k1_AXP_s016
                                 + 1.54713 w_k1_DIS_s016
                                 + 1.11314 w_k1_FDX_s016
                                 + 1.77167 w_k1_IBM_s016
                                 + 1.18622 w_k1_INTC_s016
                                 + 1.49311 w_k1_KO_s016 + 1.89115 w_k1_MAT_s016
                                 + 1.2047 w_k1_MSFT_s016 + 1.12838 w_k1_PG_s016
                                 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.602161 w_k1_ADM_s016
                                 + 0.597159 w_k1_AXP_s016
                                 + 0.571418 w_k1_DIS_s016
                                 + 0.719895 w_k1_FDX_s016
                                 + 0.758164 w_k1_IBM_s016
                                 + 0.737837 w_k1_INTC_s016
                                 + 0.674876 w_k1_KO_s016
                                 + 0.750688 w_k1_MAT_s016
                                 + 0.796884 w_k1_MSFT_s016
                                 + 0.785599 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.727792 w_k1_ADM_s016
                                 + 0.667369 w_k1_AXP_s016
                                 + 0.768887 w_k1_DIS_s016
                                 + 0.886536 w_k1_FDX_s016
                                 + 0.671332 w_k1_IBM_s016
                                 + 0.902464 w_k1_INTC_s016
                                 + 0.831461 w_k1_KO_s016
                                 + 0.561252 w_k1_MAT_s016
                                 + 0.908325 w_k1_MSFT_s016
                                 + 0.897506 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.958484 w_k1_ADM_s016
                                 + 1.15657 w_k1_AXP_s016
                                 + 1.11684 w_k1_DIS_s016
                                 + 1.15233 w_k1_FDX_s016
                                 + 0.744537 w_k1_IBM_s016
                                 + 1.12128 w_k1_INTC_s016
                                 + 0.851132 w_k1_KO_s016
                                 + 0.778934 w_k1_MAT_s016
                                 + 1.12361 w_k1_MSFT_s016
                                 + 0.97974 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.11087 w_k1_ADM_s016
                                 + 1.14164 w_k1_AXP_s016
                                 + 1.25547 w_k1_DIS_s016
                                 + 1.26343 w_k1_FDX_s016
                                 + 1.17626 w_k1_IBM_s016
                                 + 1.13606 w_k1_INTC_s016
                                 + 1.15439 w_k1_KO_s016
                                 + 0.920307 w_k1_MAT_s016
                                 + 1.22116 w_k1_MSFT_s016
                                 + 1.12127 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.852271 w_k1_ADM_s016
                                 + 0.906753 w_k1_AXP_s016
                                 + 0.756546 w_k1_DIS_s016
                                 + 0.829425 w_k1_FDX_s016
                                 + 0.994353 w_k1_IBM_s016
                                 + 0.817778 w_k1_INTC_s016
                                 + 0.853104 w_k1_KO_s016 + 0.8945 w_k1_MAT_s016
                                 + 0.835564 w_k1_MSFT_s016
                                 + 0.874809 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.922748 w_k1_ADM_s017
                                 + 0.943013 w_k1_AXP_s017
                                 + 0.87683 w_k1_DIS_s017
                                 + 0.945591 w_k1_FDX_s017
                                 + 0.839582 w_k1_IBM_s017
                                 + 1.14406 w_k1_INTC_s017
                                 + 0.865933 w_k1_KO_s017
                                 + 0.839978 w_k1_MAT_s017
                                 + 0.940333 w_k1_MSFT_s017
                                 + 0.975133 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.26238 w_k1_ADM_s017
                                 + 1.49218 w_k1_AXP_s017 + 1.441 w_k1_DIS_s017
                                 + 1.13853 w_k1_FDX_s017
                                 + 1.63874 w_k1_IBM_s017
                                 + 1.24779 w_k1_INTC_s017
                                 + 1.43981 w_k1_KO_s017 + 1.61604 w_k1_MAT_s017
                                 + 1.31148 w_k1_MSFT_s017
                                 + 1.18021 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.10585 w_k1_ADM_s017
                                 + 1.4207 w_k1_AXP_s017 + 1.33169 w_k1_DIS_s017
                                 + 1.25947 w_k1_FDX_s017
                                 + 1.04269 w_k1_IBM_s017
                                 + 1.24753 w_k1_INTC_s017
                                 + 1.21204 w_k1_KO_s017 + 1.14408 w_k1_MAT_s017
                                 + 1.19463 w_k1_MSFT_s017
                                 + 1.16257 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.822849 w_k1_ADM_s017
                                 + 0.84645 w_k1_AXP_s017
                                 + 0.691841 w_k1_DIS_s017
                                 + 0.685019 w_k1_FDX_s017
                                 + 1.6938 w_k1_IBM_s017
                                 + 0.950278 w_k1_INTC_s017
                                 + 1.16834 w_k1_KO_s017 + 1.52737 w_k1_MAT_s017
                                 + 0.870235 w_k1_MSFT_s017
                                 + 0.947495 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.06314 w_k1_ADM_s017
                                 + 0.935826 w_k1_AXP_s017
                                 + 0.990515 w_k1_DIS_s017
                                 + 1.07074 w_k1_FDX_s017
                                 + 0.663157 w_k1_IBM_s017
                                 + 1.16856 w_k1_INTC_s017
                                 + 0.836687 w_k1_KO_s017
                                 + 0.696573 w_k1_MAT_s017
                                 + 1.06229 w_k1_MSFT_s017
                                 + 0.925991 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.955624 w_k1_ADM_s017
                                 + 0.933218 w_k1_AXP_s017
                                 + 0.813377 w_k1_DIS_s017
                                 + 0.890052 w_k1_FDX_s017
                                 + 1.19174 w_k1_IBM_s017
                                 + 1.01937 w_k1_INTC_s017
                                 + 0.960977 w_k1_KO_s017
                                 + 1.30921 w_k1_MAT_s017
                                 + 0.980816 w_k1_MSFT_s017
                                 + 0.896103 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.77438 w_k1_ADM_s017
                                 + 0.719125 w_k1_AXP_s017
                                 + 0.638981 w_k1_DIS_s017
                                 + 0.682812 w_k1_FDX_s017
                                 + 0.91636 w_k1_IBM_s017
                                 + 0.951211 w_k1_INTC_s017
                                 + 0.76391 w_k1_KO_s017
                                 + 0.757437 w_k1_MAT_s017
                                 + 0.75061 w_k1_MSFT_s017
                                 + 0.792416 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.29349 w_k1_ADM_s017
                                 + 1.44019 w_k1_AXP_s017 + 1.8028 w_k1_DIS_s017
                                 + 1.54315 w_k1_FDX_s017
                                 + 1.06353 w_k1_IBM_s017
                                 + 1.73748 w_k1_INTC_s017
                                 + 1.36084 w_k1_KO_s017 + 1.00009 w_k1_MAT_s017
                                 + 1.5291 w_k1_MSFT_s017 + 1.32193 w_k1_PG_s017
                                 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.43449 w_k1_ADM_s100
                                 + 1.51457 w_k1_AXP_s100
                                 + 0.830853 w_k1_DIS_s100
                                 + 1.0365 w_k1_FDX_s100
                                 + 0.916085 w_k1_IBM_s100
                                 + 0.881941 w_k1_INTC_s100
                                 + 0.814768 w_k1_KO_s100
                                 + 0.969357 w_k1_MAT_s100
                                 + 0.899101 w_k1_MSFT_s100
                                 + 0.882259 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.25521 w_k1_ADM_s100
                                 + 1.32784 w_k1_AXP_s100
                                 + 1.06954 w_k1_DIS_s100
                                 + 1.09306 w_k1_FDX_s100
                                 + 0.88854 w_k1_IBM_s100
                                 + 1.22993 w_k1_INTC_s100
                                 + 1.11753 w_k1_KO_s100 + 0.78302 w_k1_MAT_s100
                                 + 1.16324 w_k1_MSFT_s100
                                 + 1.00107 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.856183 w_k1_ADM_s100
                                 + 0.934397 w_k1_AXP_s100
                                 + 0.820271 w_k1_DIS_s100
                                 + 0.928038 w_k1_FDX_s100
                                 + 0.848097 w_k1_IBM_s100
                                 + 0.911336 w_k1_INTC_s100
                                 + 0.830966 w_k1_KO_s100
                                 + 0.571686 w_k1_MAT_s100
                                 + 0.995328 w_k1_MSFT_s100
                                 + 0.902771 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.889204 w_k1_ADM_s100
                                 + 1.23984 w_k1_AXP_s100
                                 + 0.809067 w_k1_DIS_s100
                                 + 0.924657 w_k1_FDX_s100
                                 + 1.10964 w_k1_IBM_s100
                                 + 1.01363 w_k1_INTC_s100
                                 + 0.917609 w_k1_KO_s100 + 1.1255 w_k1_MAT_s100
                                 + 0.96171 w_k1_MSFT_s100
                                 + 0.879862 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.07913 w_k1_ADM_s100
                                 + 1.148 w_k1_AXP_s100 + 1.03333 w_k1_DIS_s100
                                 + 0.984774 w_k1_FDX_s100
                                 + 1.58588 w_k1_IBM_s100
                                 + 1.16427 w_k1_INTC_s100
                                 + 1.42571 w_k1_KO_s100 + 1.18555 w_k1_MAT_s100
                                 + 1.12758 w_k1_MSFT_s100
                                 + 1.04149 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.985964 w_k1_ADM_s100
                                 + 0.845008 w_k1_AXP_s100
                                 + 0.928858 w_k1_DIS_s100
                                 + 0.980322 w_k1_FDX_s100
                                 + 0.89184 w_k1_IBM_s100
                                 + 1.04066 w_k1_INTC_s100
                                 + 0.900056 w_k1_KO_s100
                                 + 0.707974 w_k1_MAT_s100
                                 + 1.15958 w_k1_MSFT_s100
                                 + 0.983126 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.20497 w_k1_ADM_s100
                                 + 1.47206 w_k1_AXP_s100
                                 + 1.30034 w_k1_DIS_s100
                                 + 1.24676 w_k1_FDX_s100
                                 + 1.27833 w_k1_IBM_s100
                                 + 1.08288 w_k1_INTC_s100
                                 + 1.17522 w_k1_KO_s100 + 1.03613 w_k1_MAT_s100
                                 + 1.2889 w_k1_MSFT_s100 + 1.13715 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.679523 w_k1_ADM_s100
                                 + 0.875299 w_k1_AXP_s100
                                 + 0.654866 w_k1_DIS_s100
                                 + 0.740116 w_k1_FDX_s100
                                 + 0.719938 w_k1_IBM_s100
                                 + 0.783863 w_k1_INTC_s100
                                 + 0.67367 w_k1_KO_s100
                                 + 0.946057 w_k1_MAT_s100
                                 + 0.791387 w_k1_MSFT_s100
                                 + 0.784624 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.731627 w_k1_ADM_s101
                                 + 0.727179 w_k1_AXP_s101
                                 + 0.616823 w_k1_DIS_s101
                                 + 0.780633 w_k1_FDX_s101
                                 + 0.712026 w_k1_IBM_s101
                                 + 0.835021 w_k1_INTC_s101
                                 + 0.763244 w_k1_KO_s101
                                 + 0.652836 w_k1_MAT_s101
                                 + 0.779725 w_k1_MSFT_s101
                                 + 0.764862 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.13287 w_k1_ADM_s101
                                 + 1.26792 w_k1_AXP_s101
                                 + 1.25226 w_k1_DIS_s101
                                 + 1.14721 w_k1_FDX_s101
                                 + 1.07708 w_k1_IBM_s101
                                 + 1.00732 w_k1_INTC_s101
                                 + 1.08917 w_k1_KO_s101 + 1.14233 w_k1_MAT_s101
                                 + 0.963213 w_k1_MSFT_s101
                                 + 1.07134 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.756914 w_k1_ADM_s101
                                 + 1.24026 w_k1_AXP_s101
                                 + 0.930764 w_k1_DIS_s101
                                 + 0.901038 w_k1_FDX_s101
                                 + 1.42935 w_k1_IBM_s101
                                 + 1.01971 w_k1_INTC_s101
                                 + 1.10379 w_k1_KO_s101 + 1.48229 w_k1_MAT_s101
                                 + 0.952404 w_k1_MSFT_s101
                                 + 0.981038 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.08127 w_k1_ADM_s101
                                 + 1.13683 w_k1_AXP_s101
                                 + 1.50664 w_k1_DIS_s101 + 1.352 w_k1_FDX_s101
                                 + 0.794561 w_k1_IBM_s101
                                 + 1.09166 w_k1_INTC_s101 + 1.1969 w_k1_KO_s101
                                 + 1.13899 w_k1_MAT_s101
                                 + 1.32381 w_k1_MSFT_s101 + 1.1381 w_k1_PG_s101
                                 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.14838 w_k1_ADM_s101
                                 + 1.40076 w_k1_AXP_s101
                                 + 1.65981 w_k1_DIS_s101
                                 + 1.29887 w_k1_FDX_s101
                                 + 1.36615 w_k1_IBM_s101
                                 + 1.39384 w_k1_INTC_s101
                                 + 1.68791 w_k1_KO_s101 + 1.69201 w_k1_MAT_s101
                                 + 1.33196 w_k1_MSFT_s101
                                 + 1.17699 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.686212 w_k1_ADM_s101
                                 + 0.819648 w_k1_AXP_s101
                                 + 0.66041 w_k1_DIS_s101
                                 + 0.742069 w_k1_FDX_s101
                                 + 1.12425 w_k1_IBM_s101
                                 + 0.967839 w_k1_INTC_s101
                                 + 0.89674 w_k1_KO_s101
                                 + 0.985694 w_k1_MAT_s101
                                 + 0.714467 w_k1_MSFT_s101
                                 + 0.859277 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.09727 w_k1_ADM_s101
                                 + 0.832183 w_k1_AXP_s101
                                 + 0.950575 w_k1_DIS_s101
                                 + 1.01206 w_k1_FDX_s101
                                 + 0.75466 w_k1_IBM_s101
                                 + 0.924934 w_k1_INTC_s101
                                 + 0.923334 w_k1_KO_s101
                                 + 0.752414 w_k1_MAT_s101
                                 + 1.01295 w_k1_MSFT_s101
                                 + 0.903049 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.832813 w_k1_ADM_s101
                                 + 0.97862 w_k1_AXP_s101
                                 + 0.784707 w_k1_DIS_s101
                                 + 0.814318 w_k1_FDX_s101
                                 + 1.78477 w_k1_IBM_s101
                                 + 1.13733 w_k1_INTC_s101
                                 + 1.10853 w_k1_KO_s101 + 1.40391 w_k1_MAT_s101
                                 + 0.96397 w_k1_MSFT_s101
                                 + 1.00991 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.765039 w_k1_ADM_s102
                                 + 0.600085 w_k1_AXP_s102
                                 + 0.595012 w_k1_DIS_s102
                                 + 0.742914 w_k1_FDX_s102
                                 + 0.802106 w_k1_IBM_s102
                                 + 0.728386 w_k1_INTC_s102
                                 + 0.757769 w_k1_KO_s102
                                 + 0.494339 w_k1_MAT_s102
                                 + 0.699517 w_k1_MSFT_s102
                                 + 0.797369 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.730592 w_k1_ADM_s102
                                 + 0.934797 w_k1_AXP_s102
                                 + 0.913374 w_k1_DIS_s102
                                 + 1.01352 w_k1_FDX_s102
                                 + 0.684346 w_k1_IBM_s102
                                 + 1.21107 w_k1_INTC_s102
                                 + 0.930675 w_k1_KO_s102
                                 + 0.573704 w_k1_MAT_s102
                                 + 1.11879 w_k1_MSFT_s102
                                 + 0.907811 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.12268 w_k1_ADM_s102
                                 + 1.38556 w_k1_AXP_s102
                                 + 1.37445 w_k1_DIS_s102
                                 + 1.26562 w_k1_FDX_s102
                                 + 1.21709 w_k1_IBM_s102
                                 + 1.61899 w_k1_INTC_s102
                                 + 1.53234 w_k1_KO_s102 + 1.04252 w_k1_MAT_s102
                                 + 1.40782 w_k1_MSFT_s102
                                 + 1.23847 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.84559 w_k1_ADM_s102
                                 + 0.851984 w_k1_AXP_s102
                                 + 0.812498 w_k1_DIS_s102
                                 + 0.827817 w_k1_FDX_s102
                                 + 0.892843 w_k1_IBM_s102
                                 + 1.11478 w_k1_INTC_s102
                                 + 0.972683 w_k1_KO_s102
                                 + 0.678913 w_k1_MAT_s102
                                 + 0.863632 w_k1_MSFT_s102
                                 + 0.945772 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.949443 w_k1_ADM_s102
                                 + 1.14285 w_k1_AXP_s102
                                 + 0.98848 w_k1_DIS_s102
                                 + 1.04359 w_k1_FDX_s102
                                 + 0.997674 w_k1_IBM_s102
                                 + 1.26255 w_k1_INTC_s102 + 1.0188 w_k1_KO_s102
                                 + 0.874089 w_k1_MAT_s102
                                 + 0.920204 w_k1_MSFT_s102
                                 + 1.10307 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.16654 w_k1_ADM_s102
                                 + 0.880767 w_k1_AXP_s102
                                 + 1.15849 w_k1_DIS_s102
                                 + 1.01824 w_k1_FDX_s102
                                 + 1.13411 w_k1_IBM_s102
                                 + 1.25856 w_k1_INTC_s102
                                 + 1.29617 w_k1_KO_s102
                                 + 0.584197 w_k1_MAT_s102
                                 + 1.26705 w_k1_MSFT_s102 + 1.0993 w_k1_PG_s102
                                 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.835121 w_k1_ADM_s102
                                 + 0.697445 w_k1_AXP_s102
                                 + 0.721901 w_k1_DIS_s102
                                 + 0.833318 w_k1_FDX_s102
                                 + 0.63758 w_k1_IBM_s102
                                 + 0.917002 w_k1_INTC_s102
                                 + 0.80296 w_k1_KO_s102
                                 + 0.496921 w_k1_MAT_s102
                                 + 0.932377 w_k1_MSFT_s102
                                 + 0.821626 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.958248 w_k1_ADM_s102
                                 + 0.797824 w_k1_AXP_s102
                                 + 0.899662 w_k1_DIS_s102
                                 + 0.88957 w_k1_FDX_s102
                                 + 1.25959 w_k1_IBM_s102
                                 + 0.9934 w_k1_INTC_s102 + 1.21423 w_k1_KO_s102
                                 + 0.843981 w_k1_MAT_s102
                                 + 1.02176 w_k1_MSFT_s102
                                 + 1.02723 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.742224 w_k1_ADM_s103
                                 + 0.663856 w_k1_AXP_s103
                                 + 0.621735 w_k1_DIS_s103
                                 + 0.711684 w_k1_FDX_s103
                                 + 0.834968 w_k1_IBM_s103
                                 + 0.711056 w_k1_INTC_s103
                                 + 0.82035 w_k1_KO_s103
                                 + 0.884195 w_k1_MAT_s103
                                 + 0.742097 w_k1_MSFT_s103
                                 + 0.879958 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.16411 w_k1_ADM_s103
                                 + 0.936269 w_k1_AXP_s103
                                 + 1.15158 w_k1_DIS_s103
                                 + 1.02096 w_k1_FDX_s103
                                 + 1.03153 w_k1_IBM_s103
                                 + 0.893974 w_k1_INTC_s103
                                 + 0.932506 w_k1_KO_s103
                                 + 0.614764 w_k1_MAT_s103
                                 + 0.984159 w_k1_MSFT_s103
                                 + 0.963433 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.87442 w_k1_ADM_s103
                                 + 0.772074 w_k1_AXP_s103
                                 + 0.828042 w_k1_DIS_s103
                                 + 0.839301 w_k1_FDX_s103
                                 + 0.696501 w_k1_IBM_s103
                                 + 1.00483 w_k1_INTC_s103
                                 + 0.890139 w_k1_KO_s103
                                 + 0.659154 w_k1_MAT_s103
                                 + 0.863646 w_k1_MSFT_s103
                                 + 0.871083 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.02856 w_k1_ADM_s103
                                 + 0.894523 w_k1_AXP_s103
                                 + 1.29149 w_k1_DIS_s103
                                 + 1.25556 w_k1_FDX_s103
                                 + 0.73041 w_k1_IBM_s103
                                 + 1.11793 w_k1_INTC_s103
                                 + 1.05243 w_k1_KO_s103 + 0.59016 w_k1_MAT_s103
                                 + 1.19621 w_k1_MSFT_s103
                                 + 1.12191 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.35296 w_k1_ADM_s103
                                 + 1.81099 w_k1_AXP_s103
                                 + 2.04064 w_k1_DIS_s103
                                 + 1.53282 w_k1_FDX_s103
                                 + 1.17197 w_k1_IBM_s103
                                 + 1.55261 w_k1_INTC_s103
                                 + 1.48344 w_k1_KO_s103
                                 + 0.759585 w_k1_MAT_s103
                                 + 1.4133 w_k1_MSFT_s103 + 1.22537 w_k1_PG_s103
                                 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.928386 w_k1_ADM_s103
                                 + 1.52395 w_k1_AXP_s103
                                 + 1.09782 w_k1_DIS_s103 + 1.0711 w_k1_FDX_s103
                                 + 1.34092 w_k1_IBM_s103
                                 + 1.08639 w_k1_INTC_s103
                                 + 1.24811 w_k1_KO_s103 + 1.41321 w_k1_MAT_s103
                                 + 0.894981 w_k1_MSFT_s103
                                 + 1.03802 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.804675 w_k1_ADM_s103
                                 + 0.90139 w_k1_AXP_s103
                                 + 0.797263 w_k1_DIS_s103
                                 + 0.875826 w_k1_FDX_s103
                                 + 1.1733 w_k1_IBM_s103
                                 + 0.934962 w_k1_INTC_s103
                                 + 1.02226 w_k1_KO_s103
                                 + 0.999789 w_k1_MAT_s103
                                 + 0.836262 w_k1_MSFT_s103
                                 + 0.85368 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.01631 w_k1_ADM_s103
                                 + 1.22583 w_k1_AXP_s103
                                 + 1.58342 w_k1_DIS_s103
                                 + 1.24337 w_k1_FDX_s103
                                 + 1.29773 w_k1_IBM_s103
                                 + 1.29786 w_k1_INTC_s103
                                 + 1.43077 w_k1_KO_s103 + 1.12302 w_k1_MAT_s103
                                 + 1.04872 w_k1_MSFT_s103
                                 + 1.29421 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.00582 w_k1_ADM_s104
                                 + 0.808498 w_k1_AXP_s104
                                 + 0.766896 w_k1_DIS_s104
                                 + 0.82301 w_k1_FDX_s104
                                 + 0.960808 w_k1_IBM_s104
                                 + 0.86121 w_k1_INTC_s104
                                 + 0.790865 w_k1_KO_s104
                                 + 0.828923 w_k1_MAT_s104
                                 + 0.926283 w_k1_MSFT_s104
                                 + 0.964228 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.796572 w_k1_ADM_s104
                                 + 0.893931 w_k1_AXP_s104
                                 + 0.66032 w_k1_DIS_s104
                                 + 0.861914 w_k1_FDX_s104
                                 + 0.684664 w_k1_IBM_s104
                                 + 0.984174 w_k1_INTC_s104
                                 + 0.711373 w_k1_KO_s104
                                 + 0.86704 w_k1_MAT_s104
                                 + 0.922311 w_k1_MSFT_s104
                                 + 0.927933 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.683328 w_k1_ADM_s104
                                 + 0.610446 w_k1_AXP_s104
                                 + 0.659373 w_k1_DIS_s104
                                 + 0.711002 w_k1_FDX_s104
                                 + 0.826688 w_k1_IBM_s104
                                 + 0.766792 w_k1_INTC_s104
                                 + 0.804536 w_k1_KO_s104
                                 + 0.662009 w_k1_MAT_s104
                                 + 0.86488 w_k1_MSFT_s104
                                 + 0.968813 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.675066 w_k1_ADM_s104
                                 + 1.06353 w_k1_AXP_s104
                                 + 0.968588 w_k1_DIS_s104
                                 + 0.908834 w_k1_FDX_s104
                                 + 1.05972 w_k1_IBM_s104
                                 + 0.912317 w_k1_INTC_s104
                                 + 0.853983 w_k1_KO_s104
                                 + 1.26638 w_k1_MAT_s104
                                 + 0.91585 w_k1_MSFT_s104
                                 + 1.07927 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.45015 w_k1_ADM_s104
                                 + 1.47116 w_k1_AXP_s104
                                 + 1.66366 w_k1_DIS_s104
                                 + 1.40482 w_k1_FDX_s104
                                 + 1.12737 w_k1_IBM_s104
                                 + 1.55162 w_k1_INTC_s104
                                 + 1.13185 w_k1_KO_s104
                                 + 0.796702 w_k1_MAT_s104
                                 + 1.27043 w_k1_MSFT_s104
                                 + 1.11569 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.905764 w_k1_ADM_s104
                                 + 0.83063 w_k1_AXP_s104
                                 + 0.93754 w_k1_DIS_s104
                                 + 0.986768 w_k1_FDX_s104
                                 + 0.75846 w_k1_IBM_s104
                                 + 0.805503 w_k1_INTC_s104
                                 + 0.878248 w_k1_KO_s104
                                 + 0.875512 w_k1_MAT_s104
                                 + 0.973124 w_k1_MSFT_s104
                                 + 1.0751 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.935042 w_k1_ADM_s104
                                 + 0.948191 w_k1_AXP_s104
                                 + 1.18881 w_k1_DIS_s104
                                 + 0.983551 w_k1_FDX_s104
                                 + 1.3081 w_k1_IBM_s104
                                 + 1.00526 w_k1_INTC_s104
                                 + 1.19962 w_k1_KO_s104 + 1.07343 w_k1_MAT_s104
                                 + 1.04472 w_k1_MSFT_s104
                                 + 1.21772 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.02011 w_k1_ADM_s104
                                 + 1.24685 w_k1_AXP_s104
                                 + 1.32782 w_k1_DIS_s104
                                 + 1.16113 w_k1_FDX_s104
                                 + 1.14525 w_k1_IBM_s104
                                 + 1.22494 w_k1_INTC_s104
                                 + 1.26464 w_k1_KO_s104 + 1.39641 w_k1_MAT_s104
                                 + 1.18645 w_k1_MSFT_s104
                                 + 1.27869 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.32038 w_k1_ADM_s105
                                 + 1.16039 w_k1_AXP_s105 + 1.0232 w_k1_DIS_s105
                                 + 1.00675 w_k1_FDX_s105
                                 + 1.20077 w_k1_IBM_s105
                                 + 0.848185 w_k1_INTC_s105
                                 + 1.01983 w_k1_KO_s105 + 1.27481 w_k1_MAT_s105
                                 + 0.911219 w_k1_MSFT_s105
                                 + 1.09691 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.918525 w_k1_ADM_s105
                                 + 1.24082 w_k1_AXP_s105
                                 + 1.17883 w_k1_DIS_s105
                                 + 0.920373 w_k1_FDX_s105
                                 + 1.68315 w_k1_IBM_s105
                                 + 1.12789 w_k1_INTC_s105
                                 + 1.32084 w_k1_KO_s105 + 2.07235 w_k1_MAT_s105
                                 + 0.987291 w_k1_MSFT_s105
                                 + 0.995133 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.13049 w_k1_ADM_s105
                                 + 1.06688 w_k1_AXP_s105
                                 + 0.785219 w_k1_DIS_s105
                                 + 1.07591 w_k1_FDX_s105
                                 + 0.639879 w_k1_IBM_s105
                                 + 0.960178 w_k1_INTC_s105
                                 + 0.745596 w_k1_KO_s105
                                 + 0.839191 w_k1_MAT_s105
                                 + 1.03939 w_k1_MSFT_s105
                                 + 0.864276 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.20349 w_k1_ADM_s105
                                 + 1.3938 w_k1_AXP_s105 + 1.19697 w_k1_DIS_s105
                                 + 1.26432 w_k1_FDX_s105
                                 + 0.99812 w_k1_IBM_s105
                                 + 1.1602 w_k1_INTC_s105 + 1.01225 w_k1_KO_s105
                                 + 1.14691 w_k1_MAT_s105
                                 + 1.03883 w_k1_MSFT_s105
                                 + 1.11815 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.939504 w_k1_ADM_s105
                                 + 0.517701 w_k1_AXP_s105
                                 + 0.755567 w_k1_DIS_s105
                                 + 0.788944 w_k1_FDX_s105
                                 + 0.630616 w_k1_IBM_s105
                                 + 0.92357 w_k1_INTC_s105
                                 + 0.739164 w_k1_KO_s105
                                 + 0.512665 w_k1_MAT_s105
                                 + 0.80699 w_k1_MSFT_s105
                                 + 0.905399 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.53311 w_k1_ADM_s105
                                 + 1.4741 w_k1_AXP_s105 + 1.78185 w_k1_DIS_s105
                                 + 1.31519 w_k1_FDX_s105 + 1.3787 w_k1_IBM_s105
                                 + 1.31744 w_k1_INTC_s105
                                 + 1.33428 w_k1_KO_s105
                                 + 0.766687 w_k1_MAT_s105
                                 + 1.18384 w_k1_MSFT_s105
                                 + 1.19028 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.810839 w_k1_ADM_s105
                                 + 0.948524 w_k1_AXP_s105
                                 + 1.04149 w_k1_DIS_s105
                                 + 0.874984 w_k1_FDX_s105
                                 + 1.19004 w_k1_IBM_s105
                                 + 1.24778 w_k1_INTC_s105
                                 + 1.11594 w_k1_KO_s105
                                 + 0.840588 w_k1_MAT_s105
                                 + 0.944993 w_k1_MSFT_s105
                                 + 0.941524 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.969206 w_k1_ADM_s105
                                 + 0.901377 w_k1_AXP_s105
                                 + 1.15936 w_k1_DIS_s105
                                 + 1.01669 w_k1_FDX_s105
                                 + 0.872538 w_k1_IBM_s105
                                 + 0.969862 w_k1_INTC_s105
                                 + 0.904341 w_k1_KO_s105
                                 + 0.928379 w_k1_MAT_s105
                                 + 0.948714 w_k1_MSFT_s105
                                 + 1.05681 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.25048 w_k1_ADM_s106
                                 + 1.43225 w_k1_AXP_s106
                                 + 1.54464 w_k1_DIS_s106
                                 + 1.42091 w_k1_FDX_s106
                                 + 1.03168 w_k1_IBM_s106
                                 + 1.47346 w_k1_INTC_s106 + 1.3228 w_k1_KO_s106
                                 + 1.26403 w_k1_MAT_s106
                                 + 1.42744 w_k1_MSFT_s106 + 1.1705 w_k1_PG_s106
                                 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.932826 w_k1_ADM_s106
                                 + 1.03336 w_k1_AXP_s106
                                 + 0.816001 w_k1_DIS_s106
                                 + 0.983663 w_k1_FDX_s106
                                 + 1.06927 w_k1_IBM_s106
                                 + 1.08863 w_k1_INTC_s106
                                 + 1.10808 w_k1_KO_s106 + 1.59501 w_k1_MAT_s106
                                 + 1.00658 w_k1_MSFT_s106
                                 + 0.985262 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.644189 w_k1_ADM_s106
                                 + 0.799736 w_k1_AXP_s106
                                 + 0.749002 w_k1_DIS_s106
                                 + 0.905414 w_k1_FDX_s106
                                 + 0.816723 w_k1_IBM_s106
                                 + 1.15152 w_k1_INTC_s106
                                 + 0.787932 w_k1_KO_s106
                                 + 0.949413 w_k1_MAT_s106
                                 + 0.845649 w_k1_MSFT_s106
                                 + 0.859228 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.0265 w_k1_ADM_s106
                                 + 1.24778 w_k1_AXP_s106
                                 + 1.13899 w_k1_DIS_s106 + 1.1595 w_k1_FDX_s106
                                 + 0.859439 w_k1_IBM_s106
                                 + 1.22926 w_k1_INTC_s106
                                 + 0.949414 w_k1_KO_s106
                                 + 0.928312 w_k1_MAT_s106
                                 + 1.15728 w_k1_MSFT_s106
                                 + 0.977797 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.93895 w_k1_ADM_s106
                                 + 1.54101 w_k1_AXP_s106 + 1.3743 w_k1_DIS_s106
                                 + 1.32141 w_k1_FDX_s106
                                 + 1.42295 w_k1_IBM_s106
                                 + 1.57811 w_k1_INTC_s106
                                 + 1.39044 w_k1_KO_s106 + 2.01355 w_k1_MAT_s106
                                 + 1.15657 w_k1_MSFT_s106
                                 + 1.20207 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.849994 w_k1_ADM_s106
                                 + 0.999801 w_k1_AXP_s106
                                 + 0.732031 w_k1_DIS_s106
                                 + 0.857927 w_k1_FDX_s106
                                 + 1.32863 w_k1_IBM_s106
                                 + 1.18959 w_k1_INTC_s106
                                 + 1.13306 w_k1_KO_s106 + 1.59299 w_k1_MAT_s106
                                 + 0.957821 w_k1_MSFT_s106
                                 + 0.996401 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.758993 w_k1_ADM_s106
                                 + 0.929527 w_k1_AXP_s106
                                 + 1.11468 w_k1_DIS_s106
                                 + 0.942389 w_k1_FDX_s106
                                 + 1.58363 w_k1_IBM_s106
                                 + 1.30849 w_k1_INTC_s106
                                 + 1.28843 w_k1_KO_s106 + 1.71816 w_k1_MAT_s106
                                 + 0.980898 w_k1_MSFT_s106
                                 + 1.08077 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.517885 w_k1_ADM_s106
                                 + 0.680954 w_k1_AXP_s106
                                 + 0.494949 w_k1_DIS_s106
                                 + 0.650987 w_k1_FDX_s106
                                 + 0.910172 w_k1_IBM_s106
                                 + 0.85657 w_k1_INTC_s106
                                 + 0.745304 w_k1_KO_s106
                                 + 0.93196 w_k1_MAT_s106
                                 + 0.676602 w_k1_MSFT_s106
                                 + 0.772696 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.779235 w_k1_ADM_s107
                                 + 0.517444 w_k1_AXP_s107
                                 + 0.697083 w_k1_DIS_s107
                                 + 0.725027 w_k1_FDX_s107
                                 + 0.97578 w_k1_IBM_s107
                                 + 0.673463 w_k1_INTC_s107
                                 + 0.937122 w_k1_KO_s107
                                 + 0.761427 w_k1_MAT_s107
                                 + 0.93018 w_k1_MSFT_s107
                                 + 0.956592 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.987903 w_k1_ADM_s107
                                 + 0.783885 w_k1_AXP_s107
                                 + 0.909945 w_k1_DIS_s107
                                 + 0.909292 w_k1_FDX_s107
                                 + 0.954942 w_k1_IBM_s107
                                 + 0.688242 w_k1_INTC_s107
                                 + 0.893569 w_k1_KO_s107
                                 + 0.656721 w_k1_MAT_s107
                                 + 0.923107 w_k1_MSFT_s107
                                 + 1.1034 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.15618 w_k1_ADM_s107
                                 + 0.95665 w_k1_AXP_s107 + 1.038 w_k1_DIS_s107
                                 + 0.968384 w_k1_FDX_s107
                                 + 1.51108 w_k1_IBM_s107
                                 + 0.820993 w_k1_INTC_s107
                                 + 1.18027 w_k1_KO_s107 + 1.19717 w_k1_MAT_s107
                                 + 1.01864 w_k1_MSFT_s107
                                 + 1.03911 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.30421 w_k1_ADM_s107
                                 + 1.46125 w_k1_AXP_s107
                                 + 1.31889 w_k1_DIS_s107
                                 + 1.30194 w_k1_FDX_s107
                                 + 1.20752 w_k1_IBM_s107
                                 + 1.15759 w_k1_INTC_s107
                                 + 1.13247 w_k1_KO_s107 + 1.242 w_k1_MAT_s107
                                 + 1.20988 w_k1_MSFT_s107
                                 + 1.14723 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.15417 w_k1_ADM_s107
                                 + 0.796415 w_k1_AXP_s107
                                 + 1.18509 w_k1_DIS_s107
                                 + 0.997444 w_k1_FDX_s107
                                 + 1.06378 w_k1_IBM_s107
                                 + 0.847544 w_k1_INTC_s107
                                 + 1.06211 w_k1_KO_s107
                                 + 0.871658 w_k1_MAT_s107
                                 + 1.06876 w_k1_MSFT_s107
                                 + 1.09077 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.870436 w_k1_ADM_s107
                                 + 0.992225 w_k1_AXP_s107
                                 + 0.818729 w_k1_DIS_s107
                                 + 1.00731 w_k1_FDX_s107
                                 + 0.729183 w_k1_IBM_s107
                                 + 1.0008 w_k1_INTC_s107
                                 + 0.812903 w_k1_KO_s107
                                 + 0.784566 w_k1_MAT_s107
                                 + 0.987996 w_k1_MSFT_s107
                                 + 0.940525 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.07133 w_k1_ADM_s107
                                 + 0.801124 w_k1_AXP_s107
                                 + 0.822497 w_k1_DIS_s107
                                 + 0.930767 w_k1_FDX_s107
                                 + 0.679209 w_k1_IBM_s107
                                 + 0.842023 w_k1_INTC_s107
                                 + 0.822268 w_k1_KO_s107
                                 + 0.799404 w_k1_MAT_s107
                                 + 1.03566 w_k1_MSFT_s107 + 1.0057 w_k1_PG_s107
                                 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.912976 w_k1_ADM_s107
                                 + 0.5972 w_k1_AXP_s107
                                 + 0.537231 w_k1_DIS_s107
                                 + 0.7203 w_k1_FDX_s107
                                 + 0.643541 w_k1_IBM_s107
                                 + 0.656063 w_k1_INTC_s107
                                 + 0.637306 w_k1_KO_s107
                                 + 0.58204 w_k1_MAT_s107
                                 + 0.731265 w_k1_MSFT_s107
                                 + 0.852477 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.789062 w_k1_ADM_s110
                                 + 0.780751 w_k1_AXP_s110
                                 + 0.705581 w_k1_DIS_s110
                                 + 0.767088 w_k1_FDX_s110
                                 + 1.15687 w_k1_IBM_s110
                                 + 0.84606 w_k1_INTC_s110
                                 + 0.748252 w_k1_KO_s110 + 1.0374 w_k1_MAT_s110
                                 + 0.803481 w_k1_MSFT_s110
                                 + 0.873954 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.9513 w_k1_ADM_s110
                                 + 0.775975 w_k1_AXP_s110
                                 + 0.808473 w_k1_DIS_s110
                                 + 0.900787 w_k1_FDX_s110
                                 + 0.67738 w_k1_IBM_s110
                                 + 1.02696 w_k1_INTC_s110
                                 + 0.817477 w_k1_KO_s110
                                 + 0.64889 w_k1_MAT_s110
                                 + 0.928923 w_k1_MSFT_s110
                                 + 0.994594 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.941961 w_k1_ADM_s110
                                 + 1.14269 w_k1_AXP_s110
                                 + 1.03775 w_k1_DIS_s110
                                 + 1.05703 w_k1_FDX_s110
                                 + 0.817078 w_k1_IBM_s110
                                 + 1.0842 w_k1_INTC_s110
                                 + 0.743706 w_k1_KO_s110
                                 + 0.819735 w_k1_MAT_s110
                                 + 1.02842 w_k1_MSFT_s110
                                 + 1.00689 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.939248 w_k1_ADM_s110
                                 + 1.11059 w_k1_AXP_s110 + 1.6975 w_k1_DIS_s110
                                 + 1.31505 w_k1_FDX_s110
                                 + 1.17534 w_k1_IBM_s110
                                 + 1.21498 w_k1_INTC_s110
                                 + 1.19219 w_k1_KO_s110
                                 + 0.954869 w_k1_MAT_s110
                                 + 1.45474 w_k1_MSFT_s110
                                 + 1.13609 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.02115 w_k1_ADM_s110
                                 + 0.8404 w_k1_AXP_s110 + 1.19891 w_k1_DIS_s110
                                 + 1.04227 w_k1_FDX_s110
                                 + 1.06056 w_k1_IBM_s110
                                 + 0.993844 w_k1_INTC_s110
                                 + 0.958609 w_k1_KO_s110
                                 + 0.941365 w_k1_MAT_s110
                                 + 1.00431 w_k1_MSFT_s110
                                 + 0.985566 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.81385 w_k1_ADM_s110
                                 + 0.715899 w_k1_AXP_s110
                                 + 0.64172 w_k1_DIS_s110
                                 + 0.70882 w_k1_FDX_s110
                                 + 0.760793 w_k1_IBM_s110
                                 + 0.709725 w_k1_INTC_s110
                                 + 0.648925 w_k1_KO_s110
                                 + 0.893707 w_k1_MAT_s110
                                 + 0.718396 w_k1_MSFT_s110
                                 + 0.767604 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.19047 w_k1_ADM_s110
                                 + 1.15464 w_k1_AXP_s110
                                 + 1.08206 w_k1_DIS_s110
                                 + 0.962863 w_k1_FDX_s110
                                 + 1.48624 w_k1_IBM_s110
                                 + 1.01734 w_k1_INTC_s110
                                 + 1.04699 w_k1_KO_s110 + 1.89218 w_k1_MAT_s110
                                 + 1.00564 w_k1_MSFT_s110
                                 + 0.976179 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.45442 w_k1_ADM_s110
                                 + 1.44201 w_k1_AXP_s110
                                 + 1.19095 w_k1_DIS_s110
                                 + 1.21871 w_k1_FDX_s110
                                 + 1.16585 w_k1_IBM_s110
                                 + 1.12275 w_k1_INTC_s110
                                 + 1.14576 w_k1_KO_s110 + 1.0452 w_k1_MAT_s110
                                 + 1.23987 w_k1_MSFT_s110
                                 + 1.07155 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.877998 w_k1_ADM_s111
                                 + 0.806169 w_k1_AXP_s111
                                 + 0.627788 w_k1_DIS_s111
                                 + 0.803833 w_k1_FDX_s111
                                 + 0.852631 w_k1_IBM_s111
                                 + 0.934067 w_k1_INTC_s111
                                 + 0.747474 w_k1_KO_s111
                                 + 0.750454 w_k1_MAT_s111
                                 + 0.786084 w_k1_MSFT_s111
                                 + 0.882539 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.08401 w_k1_ADM_s111
                                 + 1.29282 w_k1_AXP_s111
                                 + 0.938544 w_k1_DIS_s111
                                 + 1.07986 w_k1_FDX_s111
                                 + 0.859259 w_k1_IBM_s111
                                 + 1.09916 w_k1_INTC_s111
                                 + 0.940593 w_k1_KO_s111
                                 + 1.30458 w_k1_MAT_s111
                                 + 1.05913 w_k1_MSFT_s111
                                 + 1.11324 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.74321 w_k1_ADM_s111
                                 + 0.658201 w_k1_AXP_s111
                                 + 0.777896 w_k1_DIS_s111
                                 + 0.820655 w_k1_FDX_s111
                                 + 0.570823 w_k1_IBM_s111
                                 + 1.04418 w_k1_INTC_s111
                                 + 0.795654 w_k1_KO_s111
                                 + 0.443233 w_k1_MAT_s111
                                 + 1.00099 w_k1_MSFT_s111
                                 + 0.885224 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.02762 w_k1_ADM_s111
                                 + 0.92213 w_k1_AXP_s111
                                 + 1.02238 w_k1_DIS_s111
                                 + 0.933156 w_k1_FDX_s111 + 1.044 w_k1_IBM_s111
                                 + 1.05684 w_k1_INTC_s111
                                 + 1.03566 w_k1_KO_s111
                                 + 0.957257 w_k1_MAT_s111
                                 + 1.03085 w_k1_MSFT_s111
                                 + 1.03643 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.859178 w_k1_ADM_s111
                                 + 0.989966 w_k1_AXP_s111
                                 + 0.766879 w_k1_DIS_s111
                                 + 0.867726 w_k1_FDX_s111
                                 + 1.36334 w_k1_IBM_s111
                                 + 1.18732 w_k1_INTC_s111 + 1.0764 w_k1_KO_s111
                                 + 1.55386 w_k1_MAT_s111
                                 + 1.01252 w_k1_MSFT_s111
                                 + 1.13557 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.12163 w_k1_ADM_s111
                                 + 0.893137 w_k1_AXP_s111
                                 + 1.0929 w_k1_DIS_s111 + 1.14613 w_k1_FDX_s111
                                 + 0.685206 w_k1_IBM_s111
                                 + 1.04308 w_k1_INTC_s111
                                 + 0.71968 w_k1_KO_s111
                                 + 0.658369 w_k1_MAT_s111
                                 + 1.04745 w_k1_MSFT_s111 + 1.0188 w_k1_PG_s111
                                 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.01423 w_k1_ADM_s111
                                 + 1.15445 w_k1_AXP_s111
                                 + 1.23509 w_k1_DIS_s111
                                 + 1.07633 w_k1_FDX_s111
                                 + 1.27459 w_k1_IBM_s111
                                 + 1.43247 w_k1_INTC_s111
                                 + 1.15421 w_k1_KO_s111 + 1.33 w_k1_MAT_s111
                                 + 1.19496 w_k1_MSFT_s111
                                 + 1.14206 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.29155 w_k1_ADM_s111
                                 + 1.52878 w_k1_AXP_s111
                                 + 1.48694 w_k1_DIS_s111
                                 + 1.44745 w_k1_FDX_s111
                                 + 1.05995 w_k1_IBM_s111
                                 + 1.62051 w_k1_INTC_s111
                                 + 1.27486 w_k1_KO_s111 + 1.69023 w_k1_MAT_s111
                                 + 1.45865 w_k1_MSFT_s111 + 1.3813 w_k1_PG_s111
                                 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.898887 w_k1_ADM_s112
                                 + 1.38545 w_k1_AXP_s112
                                 + 0.956982 w_k1_DIS_s112
                                 + 1.04653 w_k1_FDX_s112
                                 + 1.05331 w_k1_IBM_s112
                                 + 1.06959 w_k1_INTC_s112
                                 + 1.04383 w_k1_KO_s112 + 1.37208 w_k1_MAT_s112
                                 + 1.01486 w_k1_MSFT_s112 + 1.0861 w_k1_PG_s112
                                 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.902407 w_k1_ADM_s112
                                 + 1.18953 w_k1_AXP_s112
                                 + 1.13768 w_k1_DIS_s112
                                 + 1.10161 w_k1_FDX_s112
                                 + 1.31686 w_k1_IBM_s112
                                 + 1.22267 w_k1_INTC_s112
                                 + 1.16912 w_k1_KO_s112 + 1.65275 w_k1_MAT_s112
                                 + 1.12372 w_k1_MSFT_s112
                                 + 1.03102 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.780197 w_k1_ADM_s112
                                 + 0.7708 w_k1_AXP_s112
                                 + 0.768802 w_k1_DIS_s112
                                 + 0.860926 w_k1_FDX_s112
                                 + 0.866083 w_k1_IBM_s112
                                 + 0.94737 w_k1_INTC_s112
                                 + 0.78482 w_k1_KO_s112
                                 + 0.934024 w_k1_MAT_s112
                                 + 0.81058 w_k1_MSFT_s112
                                 + 0.922612 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.44152 w_k1_ADM_s112
                                 + 1.82959 w_k1_AXP_s112
                                 + 1.55733 w_k1_DIS_s112
                                 + 1.43395 w_k1_FDX_s112 + 1.3595 w_k1_IBM_s112
                                 + 1.61424 w_k1_INTC_s112
                                 + 1.34962 w_k1_KO_s112 + 1.58246 w_k1_MAT_s112
                                 + 1.41186 w_k1_MSFT_s112
                                 + 1.27743 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.01041 w_k1_ADM_s112
                                 + 1.04702 w_k1_AXP_s112
                                 + 1.26102 w_k1_DIS_s112
                                 + 1.21183 w_k1_FDX_s112
                                 + 0.888769 w_k1_IBM_s112
                                 + 1.38921 w_k1_INTC_s112
                                 + 1.14325 w_k1_KO_s112
                                 + 0.964524 w_k1_MAT_s112
                                 + 1.32665 w_k1_MSFT_s112
                                 + 1.03998 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.881276 w_k1_ADM_s112
                                 + 0.843368 w_k1_AXP_s112
                                 + 1.03613 w_k1_DIS_s112
                                 + 1.01899 w_k1_FDX_s112
                                 + 1.12006 w_k1_IBM_s112
                                 + 1.16512 w_k1_INTC_s112
                                 + 1.12082 w_k1_KO_s112 + 1.06541 w_k1_MAT_s112
                                 + 1.15622 w_k1_MSFT_s112
                                 + 0.977919 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.825284 w_k1_ADM_s112
                                 + 0.584829 w_k1_AXP_s112
                                 + 0.572393 w_k1_DIS_s112
                                 + 0.699671 w_k1_FDX_s112
                                 + 0.701661 w_k1_IBM_s112
                                 + 0.845368 w_k1_INTC_s112
                                 + 0.673342 w_k1_KO_s112
                                 + 0.59634 w_k1_MAT_s112
                                 + 0.890054 w_k1_MSFT_s112
                                 + 0.796845 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.771635 w_k1_ADM_s112
                                 + 0.777559 w_k1_AXP_s112
                                 + 1.00952 w_k1_DIS_s112
                                 + 0.864891 w_k1_FDX_s112
                                 + 1.39302 w_k1_IBM_s112
                                 + 1.02169 w_k1_INTC_s112
                                 + 1.08062 w_k1_KO_s112 + 1.13208 w_k1_MAT_s112
                                 + 1.03947 w_k1_MSFT_s112
                                 + 0.882514 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.814721 w_k1_ADM_s113
                                 + 0.683004 w_k1_AXP_s113
                                 + 0.638356 w_k1_DIS_s113
                                 + 0.776655 w_k1_FDX_s113
                                 + 1.09338 w_k1_IBM_s113
                                 + 0.804829 w_k1_INTC_s113
                                 + 0.907422 w_k1_KO_s113
                                 + 0.656051 w_k1_MAT_s113
                                 + 0.803179 w_k1_MSFT_s113
                                 + 0.874162 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.827492 w_k1_ADM_s113
                                 + 0.722091 w_k1_AXP_s113
                                 + 0.555822 w_k1_DIS_s113
                                 + 0.734742 w_k1_FDX_s113
                                 + 0.703229 w_k1_IBM_s113
                                 + 0.587982 w_k1_INTC_s113
                                 + 0.693326 w_k1_KO_s113
                                 + 0.722503 w_k1_MAT_s113
                                 + 0.738812 w_k1_MSFT_s113
                                 + 0.797146 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.12234 w_k1_ADM_s113
                                 + 0.874969 w_k1_AXP_s113
                                 + 0.891471 w_k1_DIS_s113
                                 + 0.98839 w_k1_FDX_s113
                                 + 0.772671 w_k1_IBM_s113
                                 + 0.85423 w_k1_INTC_s113
                                 + 0.862099 w_k1_KO_s113
                                 + 0.758876 w_k1_MAT_s113
                                 + 0.987841 w_k1_MSFT_s113
                                 + 0.90068 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.41425 w_k1_ADM_s113
                                 + 1.31231 w_k1_AXP_s113
                                 + 1.47985 w_k1_DIS_s113
                                 + 1.21564 w_k1_FDX_s113
                                 + 1.27231 w_k1_IBM_s113
                                 + 1.06213 w_k1_INTC_s113
                                 + 1.22111 w_k1_KO_s113 + 1.23796 w_k1_MAT_s113
                                 + 1.18177 w_k1_MSFT_s113
                                 + 1.02186 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.839977 w_k1_ADM_s113
                                 + 1.25786 w_k1_AXP_s113
                                 + 0.880339 w_k1_DIS_s113
                                 + 1.10503 w_k1_FDX_s113
                                 + 0.986904 w_k1_IBM_s113
                                 + 1.09713 w_k1_INTC_s113
                                 + 0.891496 w_k1_KO_s113
                                 + 1.00772 w_k1_MAT_s113
                                 + 1.02618 w_k1_MSFT_s113
                                 + 0.971823 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.17442 w_k1_ADM_s113
                                 + 1.43864 w_k1_AXP_s113
                                 + 1.39673 w_k1_DIS_s113
                                 + 1.36648 w_k1_FDX_s113
                                 + 0.910496 w_k1_IBM_s113
                                 + 1.24498 w_k1_INTC_s113
                                 + 1.08322 w_k1_KO_s113 + 1.01172 w_k1_MAT_s113
                                 + 1.30981 w_k1_MSFT_s113
                                 + 1.09778 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.28176 w_k1_ADM_s113
                                 + 1.20572 w_k1_AXP_s113
                                 + 1.10467 w_k1_DIS_s113
                                 + 1.06081 w_k1_FDX_s113
                                 + 1.73753 w_k1_IBM_s113
                                 + 1.00317 w_k1_INTC_s113
                                 + 1.40359 w_k1_KO_s113 + 1.59561 w_k1_MAT_s113
                                 + 1.17918 w_k1_MSFT_s113 + 1.269 w_k1_PG_s113
                                 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.958638 w_k1_ADM_s113
                                 + 0.923175 w_k1_AXP_s113
                                 + 0.830865 w_k1_DIS_s113
                                 + 0.936017 w_k1_FDX_s113
                                 + 1.30892 w_k1_IBM_s113
                                 + 0.878278 w_k1_INTC_s113
                                 + 1.02438 w_k1_KO_s113 + 1.07116 w_k1_MAT_s113
                                 + 0.77045 w_k1_MSFT_s113
                                 + 0.974191 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.920062 w_k1_ADM_s114
                                 + 0.805592 w_k1_AXP_s114
                                 + 0.878756 w_k1_DIS_s114
                                 + 0.986562 w_k1_FDX_s114
                                 + 0.636793 w_k1_IBM_s114
                                 + 0.946414 w_k1_INTC_s114
                                 + 0.798508 w_k1_KO_s114
                                 + 0.637101 w_k1_MAT_s114
                                 + 0.983114 w_k1_MSFT_s114
                                 + 0.965688 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.3483 w_k1_ADM_s114
                                 + 1.46671 w_k1_AXP_s114
                                 + 1.29172 w_k1_DIS_s114
                                 + 1.14719 w_k1_FDX_s114
                                 + 1.46218 w_k1_IBM_s114
                                 + 1.07516 w_k1_INTC_s114
                                 + 1.27488 w_k1_KO_s114 + 1.33269 w_k1_MAT_s114
                                 + 1.08416 w_k1_MSFT_s114
                                 + 1.11843 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.31117 w_k1_ADM_s114
                                 + 1.3316 w_k1_AXP_s114 + 1.26614 w_k1_DIS_s114
                                 + 1.21461 w_k1_FDX_s114
                                 + 1.16755 w_k1_IBM_s114
                                 + 1.12163 w_k1_INTC_s114
                                 + 1.28542 w_k1_KO_s114 + 1.625 w_k1_MAT_s114
                                 + 1.29199 w_k1_MSFT_s114
                                 + 1.14092 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.16666 w_k1_ADM_s114
                                 + 1.20466 w_k1_AXP_s114
                                 + 0.973117 w_k1_DIS_s114
                                 + 1.01514 w_k1_FDX_s114 + 1.0145 w_k1_IBM_s114
                                 + 0.992537 w_k1_INTC_s114
                                 + 1.06362 w_k1_KO_s114 + 1.30259 w_k1_MAT_s114
                                 + 1.16101 w_k1_MSFT_s114
                                 + 1.13068 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.876576 w_k1_ADM_s114
                                 + 0.813182 w_k1_AXP_s114
                                 + 0.727034 w_k1_DIS_s114
                                 + 0.829995 w_k1_FDX_s114
                                 + 0.83189 w_k1_IBM_s114
                                 + 0.925899 w_k1_INTC_s114
                                 + 0.963152 w_k1_KO_s114
                                 + 0.992078 w_k1_MAT_s114
                                 + 1.05672 w_k1_MSFT_s114
                                 + 1.02683 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.97029 w_k1_ADM_s114
                                 + 0.955976 w_k1_AXP_s114
                                 + 0.960014 w_k1_DIS_s114
                                 + 0.99309 w_k1_FDX_s114
                                 + 1.04824 w_k1_IBM_s114
                                 + 0.771585 w_k1_INTC_s114
                                 + 1.03543 w_k1_KO_s114 + 1.10286 w_k1_MAT_s114
                                 + 0.933841 w_k1_MSFT_s114
                                 + 1.00192 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.79323 w_k1_ADM_s114
                                 + 0.75325 w_k1_AXP_s114
                                 + 0.656668 w_k1_DIS_s114
                                 + 0.666749 w_k1_FDX_s114
                                 + 0.820535 w_k1_IBM_s114
                                 + 0.711194 w_k1_INTC_s114
                                 + 0.814802 w_k1_KO_s114
                                 + 0.765878 w_k1_MAT_s114
                                 + 0.764236 w_k1_MSFT_s114
                                 + 0.793036 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.24093 w_k1_ADM_s114
                                 + 1.09584 w_k1_AXP_s114
                                 + 1.30509 w_k1_DIS_s114
                                 + 1.17553 w_k1_FDX_s114
                                 + 0.84621 w_k1_IBM_s114
                                 + 1.2765 w_k1_INTC_s114 + 1.11117 w_k1_KO_s114
                                 + 0.692776 w_k1_MAT_s114
                                 + 1.33101 w_k1_MSFT_s114
                                 + 1.15961 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.690169 w_k1_ADM_s115
                                 + 0.624435 w_k1_AXP_s115
                                 + 0.73032 w_k1_DIS_s115
                                 + 0.835172 w_k1_FDX_s115
                                 + 0.67242 w_k1_IBM_s115
                                 + 0.986697 w_k1_INTC_s115
                                 + 0.736986 w_k1_KO_s115
                                 + 0.553035 w_k1_MAT_s115
                                 + 0.887074 w_k1_MSFT_s115
                                 + 0.969443 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.894843 w_k1_ADM_s115
                                 + 0.990745 w_k1_AXP_s115
                                 + 0.795058 w_k1_DIS_s115
                                 + 0.893314 w_k1_FDX_s115
                                 + 1.2688 w_k1_IBM_s115
                                 + 0.90679 w_k1_INTC_s115
                                 + 0.870245 w_k1_KO_s115
                                 + 1.54473 w_k1_MAT_s115
                                 + 0.84794 w_k1_MSFT_s115 + 1.0682 w_k1_PG_s115
                                 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.08283 w_k1_ADM_s115
                                 + 1.14593 w_k1_AXP_s115
                                 + 1.16498 w_k1_DIS_s115
                                 + 1.00836 w_k1_FDX_s115
                                 + 1.58658 w_k1_IBM_s115
                                 + 0.973349 w_k1_INTC_s115
                                 + 1.27004 w_k1_KO_s115 + 1.54043 w_k1_MAT_s115
                                 + 1.0674 w_k1_MSFT_s115 + 1.24934 w_k1_PG_s115
                                 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.2551 w_k1_ADM_s115
                                 + 1.19151 w_k1_AXP_s115
                                 + 1.01723 w_k1_DIS_s115
                                 + 1.09739 w_k1_FDX_s115
                                 + 1.02302 w_k1_IBM_s115
                                 + 0.955543 w_k1_INTC_s115
                                 + 0.908329 w_k1_KO_s115
                                 + 1.19681 w_k1_MAT_s115
                                 + 1.07476 w_k1_MSFT_s115
                                 + 1.04687 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.17061 w_k1_ADM_s115
                                 + 1.13644 w_k1_AXP_s115
                                 + 1.14083 w_k1_DIS_s115
                                 + 1.34481 w_k1_FDX_s115
                                 + 0.697897 w_k1_IBM_s115
                                 + 1.03929 w_k1_INTC_s115
                                 + 0.854673 w_k1_KO_s115
                                 + 0.972866 w_k1_MAT_s115
                                 + 1.19842 w_k1_MSFT_s115
                                 + 1.12846 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.38359 w_k1_ADM_s115
                                 + 1.55197 w_k1_AXP_s115
                                 + 1.69596 w_k1_DIS_s115
                                 + 1.35747 w_k1_FDX_s115
                                 + 1.78538 w_k1_IBM_s115
                                 + 1.10859 w_k1_INTC_s115
                                 + 1.46155 w_k1_KO_s115 + 1.92151 w_k1_MAT_s115
                                 + 1.14747 w_k1_MSFT_s115
                                 + 1.47749 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.19534 w_k1_ADM_s115
                                 + 1.09265 w_k1_AXP_s115
                                 + 1.56685 w_k1_DIS_s115
                                 + 1.18852 w_k1_FDX_s115
                                 + 0.970563 w_k1_IBM_s115
                                 + 1.09734 w_k1_INTC_s115
                                 + 1.09369 w_k1_KO_s115
                                 + 0.840894 w_k1_MAT_s115
                                 + 1.22782 w_k1_MSFT_s115
                                 + 1.27268 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.945103 w_k1_ADM_s115
                                 + 0.752696 w_k1_AXP_s115
                                 + 0.820659 w_k1_DIS_s115
                                 + 0.743018 w_k1_FDX_s115
                                 + 1.10619 w_k1_IBM_s115
                                 + 0.884981 w_k1_INTC_s115
                                 + 1.0046 w_k1_KO_s115 + 0.801725 w_k1_MAT_s115
                                 + 0.879749 w_k1_MSFT_s115
                                 + 1.04336 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.899408 w_k1_ADM_s116
                                 + 1.03316 w_k1_AXP_s116
                                 + 0.858539 w_k1_DIS_s116
                                 + 0.945697 w_k1_FDX_s116
                                 + 1.12073 w_k1_IBM_s116
                                 + 0.877991 w_k1_INTC_s116
                                 + 0.998382 w_k1_KO_s116
                                 + 1.35917 w_k1_MAT_s116
                                 + 0.925024 w_k1_MSFT_s116
                                 + 0.990588 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.27991 w_k1_ADM_s116
                                 + 1.2033 w_k1_AXP_s116 + 1.32114 w_k1_DIS_s116
                                 + 1.23219 w_k1_FDX_s116
                                 + 1.00921 w_k1_IBM_s116
                                 + 1.04102 w_k1_INTC_s116
                                 + 1.13826 w_k1_KO_s116 + 1.00625 w_k1_MAT_s116
                                 + 1.08338 w_k1_MSFT_s116
                                 + 1.07393 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.894402 w_k1_ADM_s116
                                 + 0.729849 w_k1_AXP_s116
                                 + 0.621239 w_k1_DIS_s116
                                 + 0.722028 w_k1_FDX_s116
                                 + 1.0158 w_k1_IBM_s116
                                 + 0.612745 w_k1_INTC_s116
                                 + 0.827397 w_k1_KO_s116
                                 + 0.735925 w_k1_MAT_s116
                                 + 0.699597 w_k1_MSFT_s116
                                 + 0.892378 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.15621 w_k1_ADM_s116
                                 + 0.872229 w_k1_AXP_s116
                                 + 0.990998 w_k1_DIS_s116
                                 + 1.00389 w_k1_FDX_s116
                                 + 0.989598 w_k1_IBM_s116
                                 + 0.701079 w_k1_INTC_s116
                                 + 1.08363 w_k1_KO_s116
                                 + 0.951279 w_k1_MAT_s116
                                 + 0.880262 w_k1_MSFT_s116
                                 + 1.07753 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.17578 w_k1_ADM_s116
                                 + 1.40769 w_k1_AXP_s116 + 1.4264 w_k1_DIS_s116
                                 + 1.25937 w_k1_FDX_s116
                                 + 1.35404 w_k1_IBM_s116
                                 + 0.98173 w_k1_INTC_s116
                                 + 1.22917 w_k1_KO_s116 + 1.4914 w_k1_MAT_s116
                                 + 1.17243 w_k1_MSFT_s116
                                 + 1.19384 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.95722 w_k1_ADM_s116
                                 + 0.774775 w_k1_AXP_s116
                                 + 0.670619 w_k1_DIS_s116
                                 + 0.807525 w_k1_FDX_s116
                                 + 0.767853 w_k1_IBM_s116
                                 + 0.666303 w_k1_INTC_s116
                                 + 0.741485 w_k1_KO_s116
                                 + 0.524615 w_k1_MAT_s116
                                 + 0.625962 w_k1_MSFT_s116
                                 + 0.79523 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.979214 w_k1_ADM_s116
                                 + 0.820446 w_k1_AXP_s116
                                 + 0.84304 w_k1_DIS_s116
                                 + 0.927985 w_k1_FDX_s116
                                 + 0.680678 w_k1_IBM_s116
                                 + 0.846105 w_k1_INTC_s116
                                 + 0.771264 w_k1_KO_s116
                                 + 0.525482 w_k1_MAT_s116
                                 + 0.937081 w_k1_MSFT_s116
                                 + 0.850896 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.15443 w_k1_ADM_s116
                                 + 1.08437 w_k1_AXP_s116
                                 + 1.51964 w_k1_DIS_s116
                                 + 1.31128 w_k1_FDX_s116
                                 + 0.742299 w_k1_IBM_s116
                                 + 1.0174 w_k1_INTC_s116
                                 + 0.962138 w_k1_KO_s116
                                 + 0.737356 w_k1_MAT_s116
                                 + 1.08348 w_k1_MSFT_s116
                                 + 1.08158 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.702479 w_k1_ADM_s117
                                 + 0.81395 w_k1_AXP_s117
                                 + 0.690546 w_k1_DIS_s117
                                 + 0.7883 w_k1_FDX_s117
                                 + 0.935442 w_k1_IBM_s117
                                 + 0.756772 w_k1_INTC_s117
                                 + 0.814886 w_k1_KO_s117
                                 + 0.928761 w_k1_MAT_s117
                                 + 0.752238 w_k1_MSFT_s117
                                 + 0.785171 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.859402 w_k1_ADM_s117
                                 + 1.7198 w_k1_AXP_s117 + 1.61654 w_k1_DIS_s117
                                 + 1.46257 w_k1_FDX_s117
                                 + 1.33497 w_k1_IBM_s117
                                 + 1.25196 w_k1_INTC_s117
                                 + 1.09767 w_k1_KO_s117 + 1.34906 w_k1_MAT_s117
                                 + 1.13511 w_k1_MSFT_s117 + 1.2094 w_k1_PG_s117
                                 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.711072 w_k1_ADM_s117
                                 + 1.21435 w_k1_AXP_s117
                                 + 1.14578 w_k1_DIS_s117
                                 + 1.09411 w_k1_FDX_s117
                                 + 1.10261 w_k1_IBM_s117
                                 + 1.11594 w_k1_INTC_s117 + 1.0783 w_k1_KO_s117
                                 + 1.46257 w_k1_MAT_s117
                                 + 1.16612 w_k1_MSFT_s117
                                 + 1.00456 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.1987 w_k1_ADM_s117
                                 + 1.47348 w_k1_AXP_s117
                                 + 1.70864 w_k1_DIS_s117
                                 + 1.43642 w_k1_FDX_s117
                                 + 1.83769 w_k1_IBM_s117
                                 + 1.04066 w_k1_INTC_s117
                                 + 1.44192 w_k1_KO_s117 + 1.9139 w_k1_MAT_s117
                                 + 1.13599 w_k1_MSFT_s117
                                 + 1.32425 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.972708 w_k1_ADM_s117
                                 + 0.888782 w_k1_AXP_s117
                                 + 1.03343 w_k1_DIS_s117
                                 + 0.991866 w_k1_FDX_s117
                                 + 1.00297 w_k1_IBM_s117
                                 + 1.05999 w_k1_INTC_s117 + 1.0383 w_k1_KO_s117
                                 + 1.02163 w_k1_MAT_s117
                                 + 1.15194 w_k1_MSFT_s117 + 1.1054 w_k1_PG_s117
                                 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.03718 w_k1_ADM_s117
                                 + 1.27241 w_k1_AXP_s117
                                 + 1.58353 w_k1_DIS_s117
                                 + 1.42784 w_k1_FDX_s117
                                 + 1.01005 w_k1_IBM_s117
                                 + 1.34864 w_k1_INTC_s117
                                 + 1.15747 w_k1_KO_s117 + 1.00153 w_k1_MAT_s117
                                 + 1.25749 w_k1_MSFT_s117
                                 + 1.02355 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.759238 w_k1_ADM_s117
                                 + 0.833542 w_k1_AXP_s117
                                 + 0.789204 w_k1_DIS_s117
                                 + 0.97144 w_k1_FDX_s117
                                 + 0.636693 w_k1_IBM_s117
                                 + 0.912117 w_k1_INTC_s117
                                 + 0.732066 w_k1_KO_s117
                                 + 0.740593 w_k1_MAT_s117
                                 + 0.862271 w_k1_MSFT_s117
                                 + 0.874167 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.911526 w_k1_ADM_s117
                                 + 0.910675 w_k1_AXP_s117
                                 + 1.01595 w_k1_DIS_s117
                                 + 0.948253 w_k1_FDX_s117
                                 + 1.49464 w_k1_IBM_s117
                                 + 0.896757 w_k1_INTC_s117
                                 + 1.12433 w_k1_KO_s117 + 1.43059 w_k1_MAT_s117
                                 + 0.953399 w_k1_MSFT_s117
                                 + 1.01336 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0658222211571397 w_k1_ADM_s000
                                 + 0.0237267564154661 w_k1_ADM_s001
                                 + 0.0545738594209376 w_k1_ADM_s002
                                 + 0.0546851142748929 w_k1_ADM_s003
                                 + 0.0327767201042177 w_k1_ADM_s004
                                 + 0.0875849330874581 w_k1_ADM_s005
                                 + 0.106825761325057 w_k1_ADM_s006
                                 + 0.0315603344714606 w_k1_ADM_s007
                                 + 0.0166594979984888 w_k1_ADM_s010
                                 + 0.0418637238663853 w_k1_ADM_s011
                                 + 0.0265172972716455 w_k1_ADM_s012
                                 + 0.0176176562555802 w_k1_ADM_s013
                                 + 0.0372044882259096 w_k1_ADM_s014
                                 + 0.0319537714114555 w_k1_ADM_s015
                                 + 0.0512351567891003 w_k1_ADM_s016
                                 + 0.0678207755905394 w_k1_ADM_s017
                                 + 0.0073440575777977 w_k1_ADM_s100
                                 + 0.0169416766464554 w_k1_ADM_s101
                                 + 0.00497558394587892 w_k1_ADM_s102
                                 + 0.00877480076807137 w_k1_ADM_s103
                                 + 0.00931357859726834 w_k1_ADM_s104
                                 + 0.018585448971336 w_k1_ADM_s105
                                 + 0.0202875175475477 w_k1_ADM_s106
                                 + 0.00643987025200183 w_k1_ADM_s107
                                 + 0.0119350383822898 w_k1_ADM_s110
                                 + 0.0357466766167122 w_k1_ADM_s111
                                 + 0.0180749974461687 w_k1_ADM_s112
                                 + 0.0102277183119798 w_k1_ADM_s113
                                 + 0.0289811011370908 w_k1_ADM_s114
                                 + 0.0336807264737505 w_k1_ADM_s115
                                 + 0.00984860814808304 w_k1_ADM_s116
                                 + 0.0133575731430781 w_k1_ADM_s117
                                 + 0.0668055498484807 w_k1_AXP_s000
                                 + 0.0245089836133425 w_k1_AXP_s001
                                 + 0.0567523579371914 w_k1_AXP_s002
                                 + 0.0571406193767203 w_k1_AXP_s003
                                 + 0.0343324759474119 w_k1_AXP_s004
                                 + 0.0814977572705199 w_k1_AXP_s005
                                 + 0.101636224316512 w_k1_AXP_s006
                                 + 0.0340201838448615 w_k1_AXP_s007
                                 + 0.0188458003554163 w_k1_AXP_s010
                                 + 0.0475959695617018 w_k1_AXP_s011
                                 + 0.0271874054304555 w_k1_AXP_s012
                                 + 0.0168929423544504 w_k1_AXP_s013
                                 + 0.0385945932331618 w_k1_AXP_s014
                                 + 0.0301366675796877 w_k1_AXP_s015
                                 + 0.0529462341636864 w_k1_AXP_s016
                                 + 0.072256705327711 w_k1_AXP_s017
                                 + 0.00827911435211335 w_k1_AXP_s100
                                 + 0.0177999167025452 w_k1_AXP_s101
                                 + 0.00497748044225699 w_k1_AXP_s102
                                 + 0.00923459836327326 w_k1_AXP_s103
                                 + 0.00971547338684134 w_k1_AXP_s104
                                 + 0.0179973185921615 w_k1_AXP_s105
                                 + 0.0255045884257257 w_k1_AXP_s106
                                 + 0.00544207537398391 w_k1_AXP_s107
                                 + 0.0115044862549405 w_k1_AXP_s110
                                 + 0.0378661729510425 w_k1_AXP_s111
                                 + 0.0199007030678367 w_k1_AXP_s112
                                 + 0.0103994062394421 w_k1_AXP_s113
                                 + 0.0284401481072327 w_k1_AXP_s114
                                 + 0.0320862304631944 w_k1_AXP_s115
                                 + 0.00947593569473499 w_k1_AXP_s116
                                 + 0.0163232045468977 w_k1_AXP_s117
                                 + 0.0656562854115189 w_k1_DIS_s000
                                 + 0.0252768177369965 w_k1_DIS_s001
                                 + 0.0542753864014191 w_k1_DIS_s002
                                 + 0.052782494249035 w_k1_DIS_s003
                                 + 0.0344646722646995 w_k1_DIS_s004
                                 + 0.0891881678915341 w_k1_DIS_s005
                                 + 0.10993244002906 w_k1_DIS_s006
                                 + 0.0343600662720787 w_k1_DIS_s007
                                 + 0.0194997318420783 w_k1_DIS_s010
                                 + 0.0444215048450168 w_k1_DIS_s011
                                 + 0.0272292363031324 w_k1_DIS_s012
                                 + 0.0171462905040499 w_k1_DIS_s013
                                 + 0.0409671829541903 w_k1_DIS_s014
                                 + 0.0343784826488634 w_k1_DIS_s015
                                 + 0.0561013862734077 w_k1_DIS_s016
                                 + 0.0694219171959148 w_k1_DIS_s017
                                 + 0.00682836404528837 w_k1_DIS_s100
                                 + 0.0182543163583807 w_k1_DIS_s101
                                 + 0.00521363038984164 w_k1_DIS_s102
                                 + 0.00993815933285732 w_k1_DIS_s103
                                 + 0.0104434758366063 w_k1_DIS_s104
                                 + 0.0189463843569994 w_k1_DIS_s105
                                 + 0.0235174808152517 w_k1_DIS_s106
                                 + 0.0059698985027302 w_k1_DIS_s107
                                 + 0.0124901425036478 w_k1_DIS_s110
                                 + 0.0353897255186189 w_k1_DIS_s111
                                 + 0.0198900937511018 w_k1_DIS_s112
                                 + 0.00959768792017929 w_k1_DIS_s113
                                 + 0.0277562263593138 w_k1_DIS_s114
                                 + 0.0341175001383385 w_k1_DIS_s115
                                 + 0.00957313510620288 w_k1_DIS_s116
                                 + 0.0168303874326227 w_k1_DIS_s117
                                 + 0.0643097793376714 w_k1_FDX_s000
                                 + 0.0247871705503981 w_k1_FDX_s001
                                 + 0.0512470363963285 w_k1_FDX_s002
                                 + 0.055275509303123 w_k1_FDX_s003
                                 + 0.0344891390138658 w_k1_FDX_s004
                                 + 0.0889079518795636 w_k1_FDX_s005
                                 + 0.107961249835786 w_k1_FDX_s006
                                 + 0.0322170682938737 w_k1_FDX_s007
                                 + 0.0179662621189185 w_k1_FDX_s010
                                 + 0.0426503499919847 w_k1_FDX_s011
                                 + 0.0263756277786304 w_k1_FDX_s012
                                 + 0.0178505189808347 w_k1_FDX_s013
                                 + 0.0398302838599851 w_k1_FDX_s014
                                 + 0.0338065303741665 w_k1_FDX_s015
                                 + 0.0561829255944844 w_k1_FDX_s016
                                 + 0.068090410645448 w_k1_FDX_s017
                                 + 0.0071464553811525 w_k1_FDX_s100
                                 + 0.0179314996000306 w_k1_FDX_s101
                                 + 0.00508118225462903 w_k1_FDX_s102
                                 + 0.0092293208559763 w_k1_FDX_s103
                                 + 0.00971716253143573 w_k1_FDX_s104
                                 + 0.0178007214825436 w_k1_FDX_s105
                                 + 0.0247748215605835 w_k1_FDX_s106
                                 + 0.00588572723165936 w_k1_FDX_s107
                                 + 0.0118760274384722 w_k1_FDX_s110
                                 + 0.0364434708610489 w_k1_FDX_s111
                                 + 0.0200154963551369 w_k1_FDX_s112
                                 + 0.00995635169511541 w_k1_FDX_s113
                                 + 0.0268374197196892 w_k1_FDX_s114
                                 + 0.0327788591613108 w_k1_FDX_s115
                                 + 0.00960732327575257 w_k1_FDX_s116
                                 + 0.016709564701141 w_k1_FDX_s117
                                 + 0.0679573975928424 w_k1_IBM_s000
                                 + 0.0263086299200715 w_k1_IBM_s001
                                 + 0.0538936611907403 w_k1_IBM_s002
                                 + 0.0547536636792111 w_k1_IBM_s003
                                 + 0.0337542756179407 w_k1_IBM_s004
                                 + 0.0862520169961761 w_k1_IBM_s005
                                 + 0.109119731144946 w_k1_IBM_s006
                                 + 0.0342217041437016 w_k1_IBM_s007
                                 + 0.0179968007485038 w_k1_IBM_s010
                                 + 0.0439290618555223 w_k1_IBM_s011
                                 + 0.027066845589001 w_k1_IBM_s012
                                 + 0.0171721062972076 w_k1_IBM_s013
                                 + 0.0410751150427871 w_k1_IBM_s014
                                 + 0.0326725246766206 w_k1_IBM_s015
                                 + 0.0560768972822413 w_k1_IBM_s016
                                 + 0.0715320399654953 w_k1_IBM_s017
                                 + 0.00751756619375223 w_k1_IBM_s100
                                 + 0.0180268474350722 w_k1_IBM_s101
                                 + 0.00526777819069995 w_k1_IBM_s102
                                 + 0.0095866169360917 w_k1_IBM_s103
                                 + 0.0101389953938892 w_k1_IBM_s104
                                 + 0.0174443349689834 w_k1_IBM_s105
                                 + 0.0244911062652665 w_k1_IBM_s106
                                 + 0.00629724929128409 w_k1_IBM_s107
                                 + 0.0121004801474409 w_k1_IBM_s110
                                 + 0.0354410969274024 w_k1_IBM_s111
                                 + 0.0201578996354877 w_k1_IBM_s112
                                 + 0.0104506974200282 w_k1_IBM_s113
                                 + 0.0268795238776607 w_k1_IBM_s114
                                 + 0.034692992452462 w_k1_IBM_s115
                                 + 0.00935813642400937 w_k1_IBM_s116
                                 + 0.0163059558115045 w_k1_IBM_s117
                                 + 0.0631314165095391 w_k1_INTC_s000
                                 + 0.0274892516395271 w_k1_INTC_s001
                                 + 0.0504635965383771 w_k1_INTC_s002
                                 + 0.0558562318861183 w_k1_INTC_s003
                                 + 0.0349702511424127 w_k1_INTC_s004
                                 + 0.0917069205812401 w_k1_INTC_s005
                                 + 0.10091904579538 w_k1_INTC_s006
                                 + 0.0362191295536801 w_k1_INTC_s007
                                 + 0.0183273083861303 w_k1_INTC_s010
                                 + 0.0424125120083665 w_k1_INTC_s011
                                 + 0.0266807178777232 w_k1_INTC_s012
                                 + 0.0193096206506028 w_k1_INTC_s013
                                 + 0.0417434537136879 w_k1_INTC_s014
                                 + 0.0339252837988788 w_k1_INTC_s015
                                 + 0.0547661088699142 w_k1_INTC_s016
                                 + 0.078056871971222 w_k1_INTC_s017
                                 + 0.00735503906014556 w_k1_INTC_s100
                                 + 0.0179970909786939 w_k1_INTC_s101
                                 + 0.00623244158826864 w_k1_INTC_s102
                                 + 0.00940236139541813 w_k1_INTC_s103
                                 + 0.00993524310502782 w_k1_INTC_s104
                                 + 0.0184890390717 w_k1_INTC_s105
                                 + 0.0293233875385799 w_k1_INTC_s106
                                 + 0.00508610899495882 w_k1_INTC_s107
                                 + 0.0118801090099942 w_k1_INTC_s110
                                 + 0.0420825730613331 w_k1_INTC_s111
                                 + 0.0224762279468644 w_k1_INTC_s112
                                 + 0.009175283060729 w_k1_INTC_s113
                                 + 0.0259383935697881 w_k1_INTC_s114
                                 + 0.0319024716427273 w_k1_INTC_s115
                                 + 0.00797022612436013 w_k1_INTC_s116
                                 + 0.0157926863620283 w_k1_INTC_s117
                                 + 0.0697538054995165 w_k1_KO_s000
                                 + 0.0254989115424086 w_k1_KO_s001
                                 + 0.052590211386829 w_k1_KO_s002
                                 + 0.047929567260899 w_k1_KO_s003
                                 + 0.0345292386227835 w_k1_KO_s004
                                 + 0.0861997773222586 w_k1_KO_s005
                                 + 0.102728181824971 w_k1_KO_s006
                                 + 0.0307080453604382 w_k1_KO_s007
                                 + 0.0187173195324252 w_k1_KO_s010
                                 + 0.044397143049951 w_k1_KO_s011
                                 + 0.0287970017160806 w_k1_KO_s012
                                 + 0.0142285878612699 w_k1_KO_s013
                                 + 0.0367739995254479 w_k1_KO_s014
                                 + 0.0340697150545894 w_k1_KO_s015
                                 + 0.0540671610785253 w_k1_KO_s016
                                 + 0.0692783228427441 w_k1_KO_s017
                                 + 0.007242375384389 w_k1_KO_s100
                                 + 0.018670743390777 w_k1_KO_s101
                                 + 0.00597750607507791 w_k1_KO_s102
                                 + 0.00981602409312657 w_k1_KO_s103
                                 + 0.00991338305274134 w_k1_KO_s104
                                 + 0.0171466333690369 w_k1_KO_s105
                                 + 0.024251006475169 w_k1_KO_s106
                                 + 0.00591782673815979 w_k1_KO_s107
                                 + 0.0107934354571863 w_k1_KO_s110
                                 + 0.0348106022764767 w_k1_KO_s111
                                 + 0.0198306626440585 w_k1_KO_s112
                                 + 0.00967340590556146 w_k1_KO_s113
                                 + 0.028130426797864 w_k1_KO_s114
                                 + 0.0320267641396022 w_k1_KO_s115
                                 + 0.00928928966014078 w_k1_KO_s116
                                 + 0.0154825800991687 w_k1_KO_s117
                                 + 0.0645910525111621 w_k1_MAT_s000
                                 + 0.0325483478899815 w_k1_MAT_s001
                                 + 0.0555037839313271 w_k1_MAT_s002
                                 + 0.0466614008588759 w_k1_MAT_s003
                                 + 0.0360775007493036 w_k1_MAT_s004
                                 + 0.102741729612664 w_k1_MAT_s005
                                 + 0.123667465413874 w_k1_MAT_s006
                                 + 0.0359153763748274 w_k1_MAT_s007
                                 + 0.0160807540107403 w_k1_MAT_s010
                                 + 0.0401592326644937 w_k1_MAT_s011
                                 + 0.0301398877351716 w_k1_MAT_s012
                                 + 0.0184137209935422 w_k1_MAT_s013
                                 + 0.044645953813056 w_k1_MAT_s014
                                 + 0.0294199900861352 w_k1_MAT_s015
                                 + 0.0497842263090623 w_k1_MAT_s016
                                 + 0.0713863631664853 w_k1_MAT_s017
                                 + 0.00637117386306033 w_k1_MAT_s100
                                 + 0.0187216425607432 w_k1_MAT_s101
                                 + 0.00383007478239049 w_k1_MAT_s102
                                 + 0.00807725808023824 w_k1_MAT_s103
                                 + 0.01001984585357 w_k1_MAT_s104
                                 + 0.0166929017901148 w_k1_MAT_s105
                                 + 0.0294833459905816 w_k1_MAT_s106
                                 + 0.00542598052235764 w_k1_MAT_s107
                                 + 0.0117062716355453 w_k1_MAT_s110
                                 + 0.0414923797160411 w_k1_MAT_s111
                                 + 0.0216961641427702 w_k1_MAT_s112
                                 + 0.00971728517859848 w_k1_MAT_s113
                                 + 0.027493912872567 w_k1_MAT_s114
                                 + 0.0341651659830231 w_k1_MAT_s115
                                 + 0.0094438065638638 w_k1_MAT_s116
                                 + 0.0174559589363069 w_k1_MAT_s117
                                 + 0.0663183354981761 w_k1_MSFT_s000
                                 + 0.0250119029110972 w_k1_MSFT_s001
                                 + 0.0491084598362731 w_k1_MSFT_s002
                                 + 0.0500979483567297 w_k1_MSFT_s003
                                 + 0.0321402752357402 w_k1_MSFT_s004
                                 + 0.0897335152094726 w_k1_MSFT_s005
                                 + 0.110759443713723 w_k1_MSFT_s006
                                 + 0.0306952705810427 w_k1_MSFT_s007
                                 + 0.0177219578654402 w_k1_MSFT_s010
                                 + 0.0445878515905299 w_k1_MSFT_s011
                                 + 0.0278120392975412 w_k1_MSFT_s012
                                 + 0.0156643515319829 w_k1_MSFT_s013
                                 + 0.0419678012845081 w_k1_MSFT_s014
                                 + 0.0313691492863747 w_k1_MSFT_s015
                                 + 0.0569102196684204 w_k1_MSFT_s016
                                 + 0.0706130034434117 w_k1_MSFT_s017
                                 + 0.00763497299333116 w_k1_MSFT_s100
                                 + 0.0177800605683033 w_k1_MSFT_s101
                                 + 0.00562677670511325 w_k1_MSFT_s102
                                 + 0.00866244437697614 w_k1_MSFT_s103
                                 + 0.0100635576795197 w_k1_MSFT_s104
                                 + 0.0170048479094423 w_k1_MSFT_s105
                                 + 0.0242527095727726 w_k1_MSFT_s106
                                 + 0.00608972363409112 w_k1_MSFT_s107
                                 + 0.0121024833695883 w_k1_MSFT_s110
                                 + 0.0379295922137692 w_k1_MSFT_s111
                                 + 0.0209932714950407 w_k1_MSFT_s112
                                 + 0.00974144563845381 w_k1_MSFT_s113
                                 + 0.0282765044432868 w_k1_MSFT_s114
                                 + 0.0330093030370055 w_k1_MSFT_s115
                                 + 0.00873537995672536 w_k1_MSFT_s116
                                 + 0.0158506030667246 w_k1_MSFT_s117
                                 + 0.0674466059699917 w_k1_PG_s000
                                 + 0.0259516896678386 w_k1_PG_s001
                                 + 0.0566463741943607 w_k1_PG_s002
                                 + 0.0527308498705373 w_k1_PG_s003
                                 + 0.0340652197466445 w_k1_PG_s004
                                 + 0.0868698045569399 w_k1_PG_s005
                                 + 0.10597934563259 w_k1_PG_s006
                                 + 0.0310844957656438 w_k1_PG_s007
                                 + 0.0194346187055002 w_k1_PG_s010
                                 + 0.0433193432219588 w_k1_PG_s011
                                 + 0.0259649350518997 w_k1_PG_s012
                                 + 0.016967858984869 w_k1_PG_s013
                                 + 0.0389733202437471 w_k1_PG_s014
                                 + 0.0304445985238262 w_k1_PG_s015
                                 + 0.053885495984475 w_k1_PG_s016
                                 + 0.0677917944937504 w_k1_PG_s017
                                 + 0.00687929315635564 w_k1_PG_s100
                                 + 0.0171428725794913 w_k1_PG_s101
                                 + 0.00531475414538727 w_k1_PG_s102
                                 + 0.00912727107441967 w_k1_PG_s103
                                 + 0.0108738038893291 w_k1_PG_s104
                                 + 0.0176656995505528 w_k1_PG_s105
                                 + 0.0233156263263225 w_k1_PG_s106
                                 + 0.00629639345674505 w_k1_PG_s107
                                 + 0.0116358579478936 w_k1_PG_s110
                                 + 0.0384112015512787 w_k1_PG_s111
                                 + 0.0194881927224185 w_k1_PG_s112
                                 + 0.0093920101683784 w_k1_PG_s113
                                 + 0.0274459254465963 w_k1_PG_s114
                                 + 0.0364546135006694 w_k1_PG_s115
                                 + 0.00941182937300125 w_k1_PG_s116
                                 + 0.0154714847439358 w_k1_PG_s117 >= 1.209
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
