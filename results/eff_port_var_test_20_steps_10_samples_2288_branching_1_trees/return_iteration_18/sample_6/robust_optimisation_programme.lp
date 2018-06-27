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
 w_k1_ADM_s0_return_constraint:  1.17851 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.971846 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.915721 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.07465 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.04213 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.873192 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.1151 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.35052 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.09794 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.941415 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.16537 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.892313 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.874878 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.764001 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.953242 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.12222 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.35524 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.916783 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.1071 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.3549 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.16194 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.07519 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.28949 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.918499 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.754894 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.10512 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.00655 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.21333 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.11834 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.808581 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.773333 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.660817 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.842141 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.936363 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.05003 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.862163 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.99498 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.02255 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.23611 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.927286 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.941794 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.28809 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.19053 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.768668 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.931051 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.08455 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.49203 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.87541 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.20785 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.821371 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.88424 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.60777 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.745114 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.27715 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.72999 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.781245 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.14392 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.10721 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.930141 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.972748 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.18862 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.00298 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.732495 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.27736 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.829653 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.0348 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.04079 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.597509 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.767192 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.59781 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.911085 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.0807 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.07591 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.932941 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.32113 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.38102 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.32551 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.869164 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.860941 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.26543 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.25592 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.780349 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.97573 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.40869 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.80456 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.706576 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.21021 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.914166 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.829133 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.667096 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.76302 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.13112 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.96373 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.706834 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.36735 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.37405 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.00829 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.972218 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.59716 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.01141 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.594821 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.836803 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.805899 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.18965 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.31678 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.704059 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.634808 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.527068 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.799819 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.03189 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.15918 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.894816 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.42055 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.43367 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.17594 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.920404 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.941834 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.1271 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.18864 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.811922 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.12672 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.33249 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.46401 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.875993 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.05757 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.965735 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.798098 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.736802 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.768014 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.08 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.64422 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.793292 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.12552 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.37473 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.0691 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.852506 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.39375 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.912087 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.701492 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.02525 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.892948 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.17912 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.09439 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.793573 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.757625 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.772835 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.942748 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.13495 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.06631 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.901683 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.20861 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.41002 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.23103 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.916737 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.81312 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.23588 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.06075 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.99586 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.632545 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.837912 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.29439 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.836896 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.24465 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.977731 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.631 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.73152 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.13123 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.08534 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.23672 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.769513 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.41228 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.668178 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.770567 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.67722 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.818415 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.55563 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.01954 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.939712 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.70894 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.833106 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.26296 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.6202 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.844965 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.472219 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.634342 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.741425 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.0486 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.06957 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.65366 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.947386 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.22301 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.978438 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.827813 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.00571 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.06121 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.791249 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.09814 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.11147 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.50466 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.986596 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.09026 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.08836 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.8923 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.8711 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.792071 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.13502 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.43706 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.797945 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.02788 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.05476 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.959863 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.825468 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.09946 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.08168 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.712799 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.871849 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.03263 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.13971 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.04487 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.886672 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.811749 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.925323 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.999871 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.12856 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.09378 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.904095 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.23873 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.22852 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.21227 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.930141 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.82794 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.20757 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.05322 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.90627 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.891401 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.14137 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.5406 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.859948 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.24762 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.04329 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.11274 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.770987 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.00056 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.0595 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.42199 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.797328 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.24657 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.929489 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.808894 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.17445 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.17469 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.39778 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.830276 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.04862 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.837179 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.05682 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.35781 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.740792 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.683699 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.567644 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.619961 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.858866 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.02503 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.962479 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.28995 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.10547 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.15114 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.850287 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.856813 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.36481 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.27079 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.919538 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.823593 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.02668 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.55998 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.942334 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.38592 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.849772 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.74123 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.702362 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.18296 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.25674 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.26264 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.938029 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.70126 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.525744 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.854733 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.82867 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.988442 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.52026 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.852941 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.11342 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.540415 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.924987 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.24813 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.567306 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.873169 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.455627 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.730148 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.09098 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.912634 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.22869 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.68015 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.1253 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.21872 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.913725 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.875831 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.999496 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.16788 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.818964 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.06362 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.31576 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.49253 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.826546 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.11488 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.04321 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.813162 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.814512 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.956059 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.07569 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.66398 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.874297 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.1655 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.22567 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.11825 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.886647 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.3405 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.04448 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.723874 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.871803 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.928429 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.18093 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.07824 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.838672 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.834445 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.857519 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.956208 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.1605 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.14114 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.900579 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.24861 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.30458 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.11234 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.907396 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.919651 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.08472 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.11604 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.887073 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.95805 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.13544 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.2715 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.9022 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.13575 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.990548 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.0227 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.85986 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.9606 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.08708 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.29907 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.906384 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.18605 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.16573 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.997946 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.08521 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.12861 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.05765 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.833911 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.990706 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.933868 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.03235 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.14459 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.807634 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.873269 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.774692 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.94561 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.03501 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.10227 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.00787 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.30872 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.2565 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.325000227705072 z_k1_s0000
                                 - 0.155000035767016 z_k1_s0001
                                 - 0.0769999495577908 z_k1_s0002
                                 - 0.104000009745607 z_k1_s0003
                                 - 0.186999359011647 z_k1_s0004
                                 - 0.349000035738625 z_k1_s0005
                                 - 0.249000572598755 z_k1_s0006
                                 - 0.132000048784818 z_k1_s0007
                                 - 0.208999997609295 z_k1_s0010
                                 - 0.345999977429741 z_k1_s0011
                                 - 0.209999727717445 z_k1_s0012
                                 - 0.489000282899292 z_k1_s0013
                                 - 0.225999299448254 z_k1_s0014
                                 - 0.113999963332339 z_k1_s0015
                                 - 0.134000011495913 z_k1_s0016
                                 - 0.2169997834747 z_k1_s0017
                                 - 0.224999736960794 z_k1_s0020
                                 - 0.214999429540205 z_k1_s0021
                                 - 0.0680000800593168 z_k1_s0022
                                 - 0.103000007230985 z_k1_s0023
                                 - 0.338000051612368 z_k1_s0024
                                 - 0.117000006819662 z_k1_s0025
                                 - 0.0870000897582392 z_k1_s0026
                                 - 0.28299979679962 z_k1_s0027
                                 - 0.449001054078385 z_k1_s0030
                                 - 0.300999903895325 z_k1_s0031
                                 - 0.379000854288902 z_k1_s0032
                                 - 0.745998035473127 z_k1_s0033
                                 - 0.228999834854636 z_k1_s0034
                                 - 0.381000405140292 z_k1_s0035
                                 - 0.241000005561326 z_k1_s0036
                                 - 0.457999032909638 z_k1_s0037
                                 - 0.775998568647347 z_k1_s0040
                                 - 0.472998260101808 z_k1_s0041
                                 - 0.333000282614388 z_k1_s0042
                                 - 0.183000031970034 z_k1_s0043
                                 - 0.895002078638147 z_k1_s0044
                                 - 0.324000100243679 z_k1_s0045
                                 - 0.593998345662887 z_k1_s0046
                                 - 0.224999995666423 z_k1_s0047
                                 - 1.08100356670309 z_k1_s0050
                                 - 0.540001030410861 z_k1_s0051
                                 - 0.526000988111291 z_k1_s0052
                                 - 0.565000805373815 z_k1_s0053
                                 - 0.426999186132936 z_k1_s0054
                                 - 0.328000750560906 z_k1_s0055
                                 - 0.449002859467608 z_k1_s0056
                                 - 0.292000401332988 z_k1_s0057
                                 - 0.104000020684098 z_k1_s0060
                                 - 0.308999139904969 z_k1_s0061
                                 - 0.256999990412647 z_k1_s0062
                                 - 0.292000227301453 z_k1_s0063
                                 - 0.170999954914264 z_k1_s0064
                                 - 0.450000586142575 z_k1_s0065
                                 - 0.130999902812701 z_k1_s0066
                                 - 0.19899976182543 z_k1_s0067
                                 - 0.313000412363775 z_k1_s0070
                                 - 0.195999821121965 z_k1_s0071
                                 - 0.18999987961208 z_k1_s0072
                                 - 0.357000478235995 z_k1_s0073
                                 - 0.112999888035627 z_k1_s0074
                                 - 0.514999927595761 z_k1_s0075
                                 - 0.399998574657652 z_k1_s0076
                                 - 0.389998672141177 z_k1_s0077
                                 - 0.411999633492741 z_k1_s0100
                                 - 0.1330001387103 z_k1_s0101
                                 - 0.4659999049994 z_k1_s0102
                                 - 0.638999396824871 z_k1_s0103
                                 - 0.136000090193937 z_k1_s0104
                                 - 0.468999284082449 z_k1_s0105
                                 - 0.180000237221509 z_k1_s0106
                                 - 0.426999390910931 z_k1_s0107
                                 - 0.118999992311993 z_k1_s0110
                                 - 0.297000135719867 z_k1_s0111
                                 - 0.105000076628239 z_k1_s0112
                                 - 0.190000678851165 z_k1_s0113
                                 - 0.202999486414517 z_k1_s0114
                                 - 0.288000088637442 z_k1_s0115
                                 - 0.0659999671377407 z_k1_s0116
                                 - 0.150000312040366 z_k1_s0117
                                 - 0.0719999334496341 z_k1_s0120
                                 - 0.158999817684617 z_k1_s0121
                                 - 0.0650000291364538 z_k1_s0122
                                 - 0.152000519371291 z_k1_s0123
                                 - 0.238000382647256 z_k1_s0124
                                 - 0.145000009058256 z_k1_s0125
                                 - 0.170999826792885 z_k1_s0126
                                 - 0.209999553394827 z_k1_s0127
                                 - 0.247000028116617 z_k1_s0130
                                 - 0.399998925208634 z_k1_s0131
                                 - 0.586999460444099 z_k1_s0132
                                 - 0.242000041551509 z_k1_s0133
                                 - 0.565000211357909 z_k1_s0134
                                 - 0.223000092604099 z_k1_s0135
                                 - 0.398999273795755 z_k1_s0136
                                 - 0.795999244765753 z_k1_s0137
                                 - 0.227998883316639 z_k1_s0140
                                 - 0.207999541173587 z_k1_s0141
                                 - 0.353999250857861 z_k1_s0142
                                 - 0.125999707242686 z_k1_s0143
                                 - 0.235999500571907 z_k1_s0144
                                 - 0.483998826529158 z_k1_s0145
                                 - 0.110999705411495 z_k1_s0146
                                 - 0.4539992630658 z_k1_s0147
                                 - 0.172999860180041 z_k1_s0150
                                 - 0.223999617666605 z_k1_s0151
                                 - 0.255000238433091 z_k1_s0152
                                 - 0.0869999447830292 z_k1_s0153
                                 - 0.178000007110128 z_k1_s0154
                                 - 0.148000576529876 z_k1_s0155
                                 - 0.16399930550583 z_k1_s0156
                                 - 0.222000139294678 z_k1_s0157
                                 - 0.27499988238771 z_k1_s0160
                                 - 0.554999689387952 z_k1_s0161
                                 - 0.527000235517718 z_k1_s0162
                                 - 0.193000065082588 z_k1_s0163
                                 - 0.588000927269979 z_k1_s0164
                                 - 0.414999321038015 z_k1_s0165
                                 - 0.223999969560145 z_k1_s0166
                                 - 0.321998368005839 z_k1_s0167
                                 - 0.166000837047899 z_k1_s0170
                                 - 0.126000080120571 z_k1_s0171
                                 - 0.350000678112798 z_k1_s0172
                                 - 0.162000056155009 z_k1_s0173
                                 - 0.115000105665608 z_k1_s0174
                                 - 0.256999537356898 z_k1_s0175
                                 - 0.338999555657579 z_k1_s0176
                                 - 0.125000007897376 z_k1_s0177
                                 - 0.44799908992498 z_k1_s1000
                                 - 0.181000095822881 z_k1_s1001
                                 - 0.269999962323609 z_k1_s1002
                                 - 0.402000791464159 z_k1_s1003
                                 - 0.786998921540161 z_k1_s1004
                                 - 0.243000026631257 z_k1_s1005
                                 - 0.168000037096921 z_k1_s1006
                                 - 0.52799968416169 z_k1_s1007
                                 - 0.595002733421505 z_k1_s1010
                                 - 0.483000376881669 z_k1_s1011
                                 - 0.269000019871813 z_k1_s1012
                                 - 0.298000234763019 z_k1_s1013
                                 - 1.30700190291041 z_k1_s1014
                                 - 1.02000160071978 z_k1_s1015
                                 - 0.953998765882497 z_k1_s1016
                                 - 0.47300034001577 z_k1_s1017
                                 - 0.637000912837583 z_k1_s1020
                                 - 0.696000613065814 z_k1_s1021
                                 - 0.16900013356914 z_k1_s1022
                                 - 0.71600069185507 z_k1_s1023
                                 - 0.300999943978131 z_k1_s1024
                                 - 0.600999351816724 z_k1_s1025
                                 - 0.239999881070924 z_k1_s1026
                                 - 0.188000031018908 z_k1_s1027
                                 - 0.117000067681671 z_k1_s1030
                                 - 0.157999956301713 z_k1_s1031
                                 - 0.340999662347311 z_k1_s1032
                                 - 0.260000387737101 z_k1_s1033
                                 - 0.539000499839536 z_k1_s1034
                                 - 0.166999994369541 z_k1_s1035
                                 - 0.211000109278908 z_k1_s1036
                                 - 0.343001094807156 z_k1_s1037
                                 - 0.60899938521641 z_k1_s1040
                                 - 0.571999268365945 z_k1_s1041
                                 - 0.382999797064034 z_k1_s1042
                                 - 0.864999200533923 z_k1_s1043
                                 - 0.365999831656994 z_k1_s1044
                                 - 0.489000109754573 z_k1_s1045
                                 - 0.293999858080214 z_k1_s1046
                                 - 0.505001374148412 z_k1_s1047
                                 - 0.354998779251658 z_k1_s1050
                                 - 0.338998607036915 z_k1_s1051
                                 - 0.316998675803034 z_k1_s1052
                                 - 1.16699517446967 z_k1_s1053
                                 - 0.900995428125754 z_k1_s1054
                                 - 0.568999310367776 z_k1_s1055
                                 - 0.52699806384446 z_k1_s1056
                                 - 0.713996745985107 z_k1_s1057
                                 - 1.20300198936172 z_k1_s1060
                                 - 1.42399811265749 z_k1_s1061
                                 - 1.43300146877724 z_k1_s1062
                                 - 0.974999462867554 z_k1_s1063
                                 - 1.2619974404099 z_k1_s1064
                                 - 0.722998511971158 z_k1_s1065
                                 - 0.790999025936455 z_k1_s1066
                                 - 0.722998511971158 z_k1_s1067
                                 - 0.523999806563306 z_k1_s1070
                                 - 0.434000607918632 z_k1_s1071
                                 - 0.429999749088006 z_k1_s1072
                                 - 0.308000522313424 z_k1_s1073
                                 - 0.86100172538811 z_k1_s1074
                                 - 0.36000081958758 z_k1_s1075
                                 - 0.660000831742416 z_k1_s1076
                                 - 0.447999588816941 z_k1_s1077
                                 - 0.203000621595989 z_k1_s1100
                                 - 0.218000621319708 z_k1_s1101
                                 - 0.262000709842854 z_k1_s1102
                                 - 0.350000886889147 z_k1_s1103
                                 - 0.616002311326886 z_k1_s1104
                                 - 0.585002669232146 z_k1_s1105
                                 - 0.158000354424122 z_k1_s1106
                                 - 0.288001602699667 z_k1_s1107
                                 - 0.266999310868108 z_k1_s1110
                                 - 0.292000415073043 z_k1_s1111
                                 - 0.151000075777492 z_k1_s1112
                                 - 0.142999933991923 z_k1_s1113
                                 - 0.149999925829289 z_k1_s1114
                                 - 0.186999656012502 z_k1_s1115
                                 - 0.441000367254138 z_k1_s1116
                                 - 0.13200004756178 z_k1_s1117
                                 - 0.674000090644513 z_k1_s1120
                                 - 0.333000717964113 z_k1_s1121
                                 - 0.569001695965322 z_k1_s1122
                                 - 0.646002658933786 z_k1_s1123
                                 - 0.153000310594289 z_k1_s1124
                                 - 0.679001033345295 z_k1_s1125
                                 - 0.374000600696874 z_k1_s1126
                                 - 0.139000167936443 z_k1_s1127
                                 - 0.187000298450108 z_k1_s1130
                                 - 0.291000668901916 z_k1_s1131
                                 - 0.214000388175083 z_k1_s1132
                                 - 0.58900047908631 z_k1_s1133
                                 - 0.525002676320846 z_k1_s1134
                                 - 0.533001982415685 z_k1_s1135
                                 - 0.330000723971175 z_k1_s1136
                                 - 0.685002214244628 z_k1_s1137
                                 - 0.353999661873109 z_k1_s1140
                                 - 0.384000987662478 z_k1_s1141
                                 - 0.236000725917873 z_k1_s1142
                                 - 0.17200046617388 z_k1_s1143
                                 - 0.356000321060547 z_k1_s1144
                                 - 0.453002324573586 z_k1_s1145
                                 - 0.529001687629626 z_k1_s1146
                                 - 0.370002081365213 z_k1_s1147
                                 - 0.505001325503972 z_k1_s1150
                                 - 0.228000818317331 z_k1_s1151
                                 - 0.313001099501349 z_k1_s1152
                                 - 0.493001128065737 z_k1_s1153
                                 - 0.726003564987126 z_k1_s1154
                                 - 0.816000855760822 z_k1_s1155
                                 - 0.82700383008792 z_k1_s1156
                                 - 0.282000868786778 z_k1_s1157
                                 - 0.653999822722707 z_k1_s1160
                                 - 1.27500292117737 z_k1_s1161
                                 - 0.663001768892765 z_k1_s1162
                                 - 0.648002689951537 z_k1_s1163
                                 - 0.745999587640438 z_k1_s1164
                                 - 0.312000832420125 z_k1_s1165
                                 - 0.980000211955532 z_k1_s1166
                                 - 0.968999221528417 z_k1_s1167
                                 - 0.40599947220673 z_k1_s1170
                                 - 0.170000101041923 z_k1_s1171
                                 - 0.31799959557572 z_k1_s1172
                                 - 0.181000040095773 z_k1_s1173
                                 - 0.199000817757132 z_k1_s1174
                                 - 0.088000058731117 z_k1_s1175
                                 - 0.334000731963856 z_k1_s1176
                                 - 0.125000068939646 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.829103 w_k1_ADM_s000
                                 + 0.721384 w_k1_AXP_s000
                                 + 0.861935 w_k1_DIS_s000
                                 + 0.848151 w_k1_FDX_s000
                                 + 0.815165 w_k1_IBM_s000
                                 + 0.86577 w_k1_INTC_s000
                                 + 0.865406 w_k1_KO_s000
                                 + 0.615097 w_k1_MAT_s000
                                 + 0.794044 w_k1_MSFT_s000
                                 + 0.967973 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.935014 w_k1_ADM_s000
                                 + 1.03353 w_k1_AXP_s000
                                 + 0.961216 w_k1_DIS_s000
                                 + 0.889856 w_k1_FDX_s000
                                 + 1.32384 w_k1_IBM_s000
                                 + 1.01449 w_k1_INTC_s000
                                 + 1.08489 w_k1_KO_s000 + 1.10904 w_k1_MAT_s000
                                 + 1.01458 w_k1_MSFT_s000
                                 + 1.01133 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.785161 w_k1_ADM_s000
                                 + 0.684101 w_k1_AXP_s000
                                 + 0.671062 w_k1_DIS_s000
                                 + 0.677196 w_k1_FDX_s000
                                 + 1.07058 w_k1_IBM_s000
                                 + 0.678784 w_k1_INTC_s000
                                 + 0.832271 w_k1_KO_s000
                                 + 1.11284 w_k1_MAT_s000
                                 + 0.801945 w_k1_MSFT_s000
                                 + 0.999167 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.02232 w_k1_ADM_s000
                                 + 0.913991 w_k1_AXP_s000
                                 + 1.23552 w_k1_DIS_s000
                                 + 0.951541 w_k1_FDX_s000
                                 + 1.16066 w_k1_IBM_s000
                                 + 0.913356 w_k1_INTC_s000
                                 + 1.16002 w_k1_KO_s000 + 1.31529 w_k1_MAT_s000
                                 + 1.09297 w_k1_MSFT_s000
                                 + 1.17023 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.827714 w_k1_ADM_s000
                                 + 1.1745 w_k1_AXP_s000
                                 + 0.814151 w_k1_DIS_s000
                                 + 0.979788 w_k1_FDX_s000
                                 + 0.897495 w_k1_IBM_s000
                                 + 1.10194 w_k1_INTC_s000
                                 + 0.840031 w_k1_KO_s000
                                 + 1.30295 w_k1_MAT_s000
                                 + 0.88785 w_k1_MSFT_s000
                                 + 0.932154 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.19255 w_k1_ADM_s000
                                 + 1.38086 w_k1_AXP_s000
                                 + 1.34252 w_k1_DIS_s000
                                 + 1.23355 w_k1_FDX_s000
                                 + 1.18443 w_k1_IBM_s000
                                 + 1.34224 w_k1_INTC_s000
                                 + 1.27251 w_k1_KO_s000 + 1.20125 w_k1_MAT_s000
                                 + 1.25058 w_k1_MSFT_s000
                                 + 1.17282 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.973973 w_k1_ADM_s000
                                 + 1.0693 w_k1_AXP_s000 + 1.12243 w_k1_DIS_s000
                                 + 1.09452 w_k1_FDX_s000
                                 + 0.859416 w_k1_IBM_s000
                                 + 0.936492 w_k1_INTC_s000
                                 + 0.990373 w_k1_KO_s000
                                 + 1.04348 w_k1_MAT_s000
                                 + 1.05883 w_k1_MSFT_s000
                                 + 1.06627 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.01051 w_k1_ADM_s000
                                 + 1.37605 w_k1_AXP_s000
                                 + 1.59252 w_k1_DIS_s000
                                 + 1.10578 w_k1_FDX_s000
                                 + 1.68684 w_k1_IBM_s000
                                 + 1.30385 w_k1_INTC_s000
                                 + 1.58405 w_k1_KO_s000 + 1.18576 w_k1_MAT_s000
                                 + 1.2257 w_k1_MSFT_s000 + 1.20512 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.990984 w_k1_ADM_s001
                                 + 1.14634 w_k1_AXP_s001
                                 + 0.759854 w_k1_DIS_s001
                                 + 1.01517 w_k1_FDX_s001
                                 + 0.645686 w_k1_IBM_s001
                                 + 1.1781 w_k1_INTC_s001
                                 + 0.772037 w_k1_KO_s001
                                 + 0.597314 w_k1_MAT_s001
                                 + 0.898221 w_k1_MSFT_s001
                                 + 0.970818 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.737268 w_k1_ADM_s001
                                 + 0.83816 w_k1_AXP_s001
                                 + 0.77539 w_k1_DIS_s001
                                 + 0.811341 w_k1_FDX_s001
                                 + 1.01788 w_k1_IBM_s001
                                 + 1.04816 w_k1_INTC_s001
                                 + 0.978531 w_k1_KO_s001
                                 + 0.938238 w_k1_MAT_s001
                                 + 0.783917 w_k1_MSFT_s001
                                 + 0.93662 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.922998 w_k1_ADM_s001
                                 + 1.11605 w_k1_AXP_s001
                                 + 1.39188 w_k1_DIS_s001
                                 + 1.15957 w_k1_FDX_s001
                                 + 0.908653 w_k1_IBM_s001
                                 + 1.10456 w_k1_INTC_s001
                                 + 0.999574 w_k1_KO_s001
                                 + 0.793652 w_k1_MAT_s001
                                 + 1.18136 w_k1_MSFT_s001
                                 + 1.00692 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.19365 w_k1_ADM_s001
                                 + 1.34953 w_k1_AXP_s001
                                 + 1.01284 w_k1_DIS_s001 + 1.1387 w_k1_FDX_s001
                                 + 1.05529 w_k1_IBM_s001
                                 + 1.01268 w_k1_INTC_s001
                                 + 1.00547 w_k1_KO_s001 + 1.24528 w_k1_MAT_s001
                                 + 0.9697 w_k1_MSFT_s001 + 1.14022 w_k1_PG_s001
                                 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.18461 w_k1_ADM_s001
                                 + 1.43053 w_k1_AXP_s001
                                 + 1.63628 w_k1_DIS_s001
                                 + 1.33246 w_k1_FDX_s001
                                 + 1.41055 w_k1_IBM_s001
                                 + 1.59674 w_k1_INTC_s001
                                 + 1.59209 w_k1_KO_s001 + 1.11309 w_k1_MAT_s001
                                 + 1.37596 w_k1_MSFT_s001
                                 + 1.30065 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.79307 w_k1_ADM_s001
                                 + 0.757077 w_k1_AXP_s001
                                 + 0.709778 w_k1_DIS_s001
                                 + 0.85412 w_k1_FDX_s001
                                 + 0.612163 w_k1_IBM_s001
                                 + 0.861287 w_k1_INTC_s001
                                 + 0.732246 w_k1_KO_s001
                                 + 0.712952 w_k1_MAT_s001
                                 + 0.893277 w_k1_MSFT_s001
                                 + 0.77216 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.768971 w_k1_ADM_s001
                                 + 1.01083 w_k1_AXP_s001
                                 + 0.755192 w_k1_DIS_s001
                                 + 0.908056 w_k1_FDX_s001
                                 + 0.83044 w_k1_IBM_s001
                                 + 1.25235 w_k1_INTC_s001
                                 + 0.865324 w_k1_KO_s001
                                 + 0.834461 w_k1_MAT_s001
                                 + 0.931087 w_k1_MSFT_s001
                                 + 0.928585 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.80879 w_k1_ADM_s001
                                 + 1.0067 w_k1_AXP_s001 + 1.05612 w_k1_DIS_s001
                                 + 0.921708 w_k1_FDX_s001
                                 + 1.36673 w_k1_IBM_s001
                                 + 1.01271 w_k1_INTC_s001
                                 + 1.07078 w_k1_KO_s001 + 1.13223 w_k1_MAT_s001
                                 + 1.00477 w_k1_MSFT_s001
                                 + 1.02847 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.945501 w_k1_ADM_s002
                                 + 0.928921 w_k1_AXP_s002
                                 + 0.975662 w_k1_DIS_s002
                                 + 0.876531 w_k1_FDX_s002
                                 + 1.04663 w_k1_IBM_s002
                                 + 0.953863 w_k1_INTC_s002
                                 + 1.02693 w_k1_KO_s002 + 0.78827 w_k1_MAT_s002
                                 + 0.922894 w_k1_MSFT_s002
                                 + 1.01454 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.917126 w_k1_ADM_s002
                                 + 0.825274 w_k1_AXP_s002
                                 + 0.652634 w_k1_DIS_s002
                                 + 0.719754 w_k1_FDX_s002
                                 + 1.03158 w_k1_IBM_s002
                                 + 0.686879 w_k1_INTC_s002
                                 + 0.767732 w_k1_KO_s002
                                 + 0.760363 w_k1_MAT_s002
                                 + 0.659902 w_k1_MSFT_s002
                                 + 0.86666 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.43057 w_k1_ADM_s002
                                 + 1.74791 w_k1_AXP_s002
                                 + 1.85274 w_k1_DIS_s002 + 1.5505 w_k1_FDX_s002
                                 + 1.33078 w_k1_IBM_s002
                                 + 1.30701 w_k1_INTC_s002
                                 + 1.28417 w_k1_KO_s002 + 1.63534 w_k1_MAT_s002
                                 + 1.4318 w_k1_MSFT_s002 + 1.41595 w_k1_PG_s002
                                 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.22804 w_k1_ADM_s002
                                 + 1.08212 w_k1_AXP_s002
                                 + 1.12086 w_k1_DIS_s002
                                 + 1.21297 w_k1_FDX_s002
                                 + 0.759699 w_k1_IBM_s002
                                 + 1.32667 w_k1_INTC_s002
                                 + 0.967013 w_k1_KO_s002
                                 + 0.748982 w_k1_MAT_s002
                                 + 1.09599 w_k1_MSFT_s002
                                 + 1.09272 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.26653 w_k1_ADM_s002
                                 + 0.927218 w_k1_AXP_s002
                                 + 0.905838 w_k1_DIS_s002
                                 + 0.978795 w_k1_FDX_s002
                                 + 0.775389 w_k1_IBM_s002
                                 + 0.804429 w_k1_INTC_s002
                                 + 0.812671 w_k1_KO_s002
                                 + 0.55494 w_k1_MAT_s002
                                 + 0.852574 w_k1_MSFT_s002
                                 + 1.00236 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.52714 w_k1_ADM_s002
                                 + 1.21493 w_k1_AXP_s002
                                 + 1.23498 w_k1_DIS_s002
                                 + 1.02998 w_k1_FDX_s002
                                 + 1.00013 w_k1_IBM_s002
                                 + 0.974569 w_k1_INTC_s002
                                 + 1.04405 w_k1_KO_s002
                                 + 0.912348 w_k1_MAT_s002
                                 + 0.957337 w_k1_MSFT_s002
                                 + 1.20584 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.49629 w_k1_ADM_s002
                                 + 1.3257 w_k1_AXP_s002 + 1.47619 w_k1_DIS_s002
                                 + 1.35521 w_k1_FDX_s002
                                 + 0.897533 w_k1_IBM_s002
                                 + 1.0447 w_k1_INTC_s002 + 1.0713 w_k1_KO_s002
                                 + 0.819173 w_k1_MAT_s002
                                 + 1.23315 w_k1_MSFT_s002
                                 + 1.21993 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.14014 w_k1_ADM_s002
                                 + 0.972351 w_k1_AXP_s002
                                 + 1.29607 w_k1_DIS_s002
                                 + 1.03464 w_k1_FDX_s002
                                 + 1.47039 w_k1_IBM_s002
                                 + 0.97913 w_k1_INTC_s002
                                 + 1.27025 w_k1_KO_s002 + 1.35952 w_k1_MAT_s002
                                 + 1.06944 w_k1_MSFT_s002
                                 + 1.07475 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.789532 w_k1_ADM_s003
                                 + 0.996233 w_k1_AXP_s003
                                 + 0.742669 w_k1_DIS_s003
                                 + 0.8297 w_k1_FDX_s003 + 0.85834 w_k1_IBM_s003
                                 + 0.765334 w_k1_INTC_s003
                                 + 0.764901 w_k1_KO_s003
                                 + 0.909694 w_k1_MAT_s003
                                 + 0.861414 w_k1_MSFT_s003
                                 + 0.906316 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.02304 w_k1_ADM_s003
                                 + 0.9301 w_k1_AXP_s003 + 1.14024 w_k1_DIS_s003
                                 + 0.955658 w_k1_FDX_s003
                                 + 1.2232 w_k1_IBM_s003
                                 + 0.837062 w_k1_INTC_s003
                                 + 1.02937 w_k1_KO_s003
                                 + 0.827058 w_k1_MAT_s003
                                 + 1.10271 w_k1_MSFT_s003
                                 + 1.05137 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.34172 w_k1_ADM_s003
                                 + 1.6081 w_k1_AXP_s003 + 1.39092 w_k1_DIS_s003
                                 + 1.15347 w_k1_FDX_s003
                                 + 1.62082 w_k1_IBM_s003
                                 + 0.899633 w_k1_INTC_s003
                                 + 1.34337 w_k1_KO_s003 + 1.99831 w_k1_MAT_s003
                                 + 1.07891 w_k1_MSFT_s003
                                 + 1.22417 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.847611 w_k1_ADM_s003
                                 + 0.836504 w_k1_AXP_s003
                                 + 0.789076 w_k1_DIS_s003
                                 + 0.80104 w_k1_FDX_s003 + 1.1192 w_k1_IBM_s003
                                 + 0.874547 w_k1_INTC_s003
                                 + 0.98614 w_k1_KO_s003 + 0.99222 w_k1_MAT_s003
                                 + 0.801305 w_k1_MSFT_s003
                                 + 0.970194 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.16527 w_k1_ADM_s003
                                 + 1.15506 w_k1_AXP_s003
                                 + 1.08782 w_k1_DIS_s003
                                 + 1.21435 w_k1_FDX_s003
                                 + 0.633307 w_k1_IBM_s003
                                 + 0.933777 w_k1_INTC_s003
                                 + 0.954071 w_k1_KO_s003
                                 + 0.831484 w_k1_MAT_s003
                                 + 1.13259 w_k1_MSFT_s003
                                 + 1.09843 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.24793 w_k1_ADM_s003
                                 + 1.395 w_k1_AXP_s003 + 1.3328 w_k1_DIS_s003
                                 + 1.29604 w_k1_FDX_s003 + 1.0737 w_k1_IBM_s003
                                 + 1.04373 w_k1_INTC_s003
                                 + 1.21129 w_k1_KO_s003 + 1.41644 w_k1_MAT_s003
                                 + 1.27803 w_k1_MSFT_s003
                                 + 1.24668 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.0277 w_k1_ADM_s003
                                 + 1.19952 w_k1_AXP_s003
                                 + 1.19766 w_k1_DIS_s003
                                 + 1.04679 w_k1_FDX_s003
                                 + 1.17875 w_k1_IBM_s003
                                 + 1.02933 w_k1_INTC_s003
                                 + 1.21497 w_k1_KO_s003 + 1.4232 w_k1_MAT_s003
                                 + 1.03678 w_k1_MSFT_s003
                                 + 1.25513 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.785714 w_k1_ADM_s003
                                 + 0.766009 w_k1_AXP_s003
                                 + 0.639757 w_k1_DIS_s003
                                 + 0.777234 w_k1_FDX_s003
                                 + 0.673717 w_k1_IBM_s003
                                 + 0.76171 w_k1_INTC_s003
                                 + 0.712909 w_k1_KO_s003
                                 + 0.773457 w_k1_MAT_s003
                                 + 0.863435 w_k1_MSFT_s003
                                 + 0.85656 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.05322 w_k1_ADM_s004
                                 + 0.845756 w_k1_AXP_s004
                                 + 0.95899 w_k1_DIS_s004
                                 + 1.02419 w_k1_FDX_s004
                                 + 0.827414 w_k1_IBM_s004
                                 + 0.959479 w_k1_INTC_s004
                                 + 0.825645 w_k1_KO_s004
                                 + 0.668787 w_k1_MAT_s004
                                 + 0.930004 w_k1_MSFT_s004
                                 + 0.953938 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.863623 w_k1_ADM_s004
                                 + 1.08282 w_k1_AXP_s004
                                 + 1.02527 w_k1_DIS_s004
                                 + 1.01927 w_k1_FDX_s004
                                 + 1.60151 w_k1_IBM_s004
                                 + 1.08317 w_k1_INTC_s004
                                 + 1.12458 w_k1_KO_s004 + 1.38641 w_k1_MAT_s004
                                 + 0.979103 w_k1_MSFT_s004
                                 + 1.1144 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.815112 w_k1_ADM_s004
                                 + 0.736619 w_k1_AXP_s004
                                 + 0.641994 w_k1_DIS_s004
                                 + 0.734384 w_k1_FDX_s004
                                 + 1.09883 w_k1_IBM_s004
                                 + 0.896884 w_k1_INTC_s004
                                 + 0.72868 w_k1_KO_s004
                                 + 0.932555 w_k1_MAT_s004
                                 + 0.770282 w_k1_MSFT_s004
                                 + 0.872269 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.785353 w_k1_ADM_s004
                                 + 0.606027 w_k1_AXP_s004
                                 + 0.65687 w_k1_DIS_s004
                                 + 0.790132 w_k1_FDX_s004
                                 + 0.621643 w_k1_IBM_s004
                                 + 0.82909 w_k1_INTC_s004
                                 + 0.70973 w_k1_KO_s004
                                 + 0.658831 w_k1_MAT_s004
                                 + 0.818291 w_k1_MSFT_s004
                                 + 0.887279 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.0869 w_k1_ADM_s004
                                 + 1.27252 w_k1_AXP_s004 + 1.309 w_k1_DIS_s004
                                 + 1.35592 w_k1_FDX_s004
                                 + 1.03335 w_k1_IBM_s004
                                 + 1.44702 w_k1_INTC_s004
                                 + 1.15427 w_k1_KO_s004 + 1.17455 w_k1_MAT_s004
                                 + 1.27888 w_k1_MSFT_s004
                                 + 1.14458 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.823774 w_k1_ADM_s004
                                 + 0.928779 w_k1_AXP_s004
                                 + 0.791805 w_k1_DIS_s004
                                 + 0.974207 w_k1_FDX_s004
                                 + 1.05511 w_k1_IBM_s004
                                 + 0.857201 w_k1_INTC_s004
                                 + 0.934219 w_k1_KO_s004
                                 + 1.25588 w_k1_MAT_s004
                                 + 0.809708 w_k1_MSFT_s004
                                 + 0.91073 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.93525 w_k1_ADM_s004
                                 + 1.22798 w_k1_AXP_s004
                                 + 1.07962 w_k1_DIS_s004
                                 + 1.07412 w_k1_FDX_s004
                                 + 1.34498 w_k1_IBM_s004
                                 + 1.09551 w_k1_INTC_s004
                                 + 1.06717 w_k1_KO_s004 + 1.48688 w_k1_MAT_s004
                                 + 0.906616 w_k1_MSFT_s004
                                 + 1.11319 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.921162 w_k1_ADM_s004
                                 + 0.951136 w_k1_AXP_s004
                                 + 0.714794 w_k1_DIS_s004
                                 + 0.828508 w_k1_FDX_s004
                                 + 0.817625 w_k1_IBM_s004
                                 + 0.681838 w_k1_INTC_s004
                                 + 0.731739 w_k1_KO_s004
                                 + 0.974906 w_k1_MAT_s004
                                 + 0.734735 w_k1_MSFT_s004
                                 + 0.824367 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.18117 w_k1_ADM_s005
                                 + 1.07131 w_k1_AXP_s005
                                 + 1.21277 w_k1_DIS_s005
                                 + 1.07369 w_k1_FDX_s005
                                 + 0.950304 w_k1_IBM_s005
                                 + 1.00215 w_k1_INTC_s005
                                 + 0.979407 w_k1_KO_s005
                                 + 0.970069 w_k1_MAT_s005
                                 + 1.057 w_k1_MSFT_s005 + 1.06905 w_k1_PG_s005
                                 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.31836 w_k1_ADM_s005
                                 + 1.55479 w_k1_AXP_s005
                                 + 1.49186 w_k1_DIS_s005
                                 + 1.37731 w_k1_FDX_s005
                                 + 1.04035 w_k1_IBM_s005
                                 + 1.16254 w_k1_INTC_s005 + 1.233 w_k1_KO_s005
                                 + 1.356 w_k1_MAT_s005 + 1.33508 w_k1_MSFT_s005
                                 + 1.23871 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.09713 w_k1_ADM_s005
                                 + 1.21424 w_k1_AXP_s005
                                 + 1.16039 w_k1_DIS_s005
                                 + 1.05189 w_k1_FDX_s005
                                 + 1.44357 w_k1_IBM_s005
                                 + 0.922141 w_k1_INTC_s005
                                 + 1.18492 w_k1_KO_s005 + 2.10076 w_k1_MAT_s005
                                 + 0.995271 w_k1_MSFT_s005
                                 + 1.10536 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.16928 w_k1_ADM_s005
                                 + 0.941618 w_k1_AXP_s005
                                 + 0.863279 w_k1_DIS_s005
                                 + 0.958394 w_k1_FDX_s005
                                 + 0.804555 w_k1_IBM_s005
                                 + 0.95754 w_k1_INTC_s005
                                 + 0.883401 w_k1_KO_s005
                                 + 0.870961 w_k1_MAT_s005
                                 + 1.0254 w_k1_MSFT_s005
                                 + 0.986879 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.810191 w_k1_ADM_s005
                                 + 0.762273 w_k1_AXP_s005
                                 + 0.640333 w_k1_DIS_s005
                                 + 0.72874 w_k1_FDX_s005
                                 + 0.926891 w_k1_IBM_s005
                                 + 0.668256 w_k1_INTC_s005
                                 + 0.796114 w_k1_KO_s005
                                 + 0.867812 w_k1_MAT_s005
                                 + 0.713476 w_k1_MSFT_s005
                                 + 0.793965 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.84528 w_k1_ADM_s005
                                 + 0.828679 w_k1_AXP_s005
                                 + 1.00533 w_k1_DIS_s005
                                 + 0.82168 w_k1_FDX_s005
                                 + 1.34962 w_k1_IBM_s005
                                 + 0.989893 w_k1_INTC_s005
                                 + 1.22292 w_k1_KO_s005 + 1.7859 w_k1_MAT_s005
                                 + 0.939502 w_k1_MSFT_s005
                                 + 1.01272 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.918094 w_k1_ADM_s005
                                 + 0.785589 w_k1_AXP_s005
                                 + 0.718897 w_k1_DIS_s005
                                 + 0.903427 w_k1_FDX_s005
                                 + 0.648805 w_k1_IBM_s005
                                 + 0.775637 w_k1_INTC_s005
                                 + 0.711567 w_k1_KO_s005
                                 + 0.80226 w_k1_MAT_s005
                                 + 0.847755 w_k1_MSFT_s005
                                 + 0.839812 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.16147 w_k1_ADM_s005
                                 + 0.793955 w_k1_AXP_s005
                                 + 0.850861 w_k1_DIS_s005
                                 + 0.788766 w_k1_FDX_s005
                                 + 1.1481 w_k1_IBM_s005
                                 + 0.743589 w_k1_INTC_s005
                                 + 0.88434 w_k1_KO_s005 + 1.29701 w_k1_MAT_s005
                                 + 0.807118 w_k1_MSFT_s005
                                 + 0.966512 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.723257 w_k1_ADM_s006
                                 + 0.713704 w_k1_AXP_s006
                                 + 0.669141 w_k1_DIS_s006
                                 + 0.722805 w_k1_FDX_s006
                                 + 0.914854 w_k1_IBM_s006
                                 + 0.877748 w_k1_INTC_s006
                                 + 0.885103 w_k1_KO_s006
                                 + 0.743435 w_k1_MAT_s006
                                 + 0.771696 w_k1_MSFT_s006
                                 + 0.812282 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.39584 w_k1_ADM_s006
                                 + 1.1847 w_k1_AXP_s006 + 1.39692 w_k1_DIS_s006
                                 + 1.16515 w_k1_FDX_s006
                                 + 1.00979 w_k1_IBM_s006
                                 + 0.986928 w_k1_INTC_s006
                                 + 1.24869 w_k1_KO_s006 + 1.03803 w_k1_MAT_s006
                                 + 1.23068 w_k1_MSFT_s006
                                 + 1.05742 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.670628 w_k1_ADM_s006
                                 + 0.753028 w_k1_AXP_s006
                                 + 0.630243 w_k1_DIS_s006
                                 + 0.731589 w_k1_FDX_s006
                                 + 0.687399 w_k1_IBM_s006
                                 + 0.936392 w_k1_INTC_s006
                                 + 0.720173 w_k1_KO_s006
                                 + 0.643989 w_k1_MAT_s006
                                 + 0.745412 w_k1_MSFT_s006
                                 + 0.896486 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.829613 w_k1_ADM_s006
                                 + 0.998183 w_k1_AXP_s006
                                 + 0.868978 w_k1_DIS_s006
                                 + 0.885322 w_k1_FDX_s006
                                 + 1.07857 w_k1_IBM_s006
                                 + 0.960872 w_k1_INTC_s006
                                 + 0.979675 w_k1_KO_s006
                                 + 0.970801 w_k1_MAT_s006
                                 + 0.898398 w_k1_MSFT_s006
                                 + 0.950308 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.22812 w_k1_ADM_s006
                                 + 1.11176 w_k1_AXP_s006
                                 + 1.86466 w_k1_DIS_s006 + 1.3007 w_k1_FDX_s006
                                 + 1.45831 w_k1_IBM_s006
                                 + 1.5143 w_k1_INTC_s006 + 1.69569 w_k1_KO_s006
                                 + 1.44263 w_k1_MAT_s006
                                 + 1.58477 w_k1_MSFT_s006
                                 + 1.17687 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.02787 w_k1_ADM_s006
                                 + 0.775407 w_k1_AXP_s006
                                 + 0.990791 w_k1_DIS_s006
                                 + 1.00542 w_k1_FDX_s006
                                 + 0.733771 w_k1_IBM_s006
                                 + 1.0692 w_k1_INTC_s006 + 1.03521 w_k1_KO_s006
                                 + 0.588683 w_k1_MAT_s006
                                 + 1.09455 w_k1_MSFT_s006
                                 + 0.933066 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.20728 w_k1_ADM_s006
                                 + 1.67562 w_k1_AXP_s006
                                 + 1.89204 w_k1_DIS_s006
                                 + 1.57532 w_k1_FDX_s006
                                 + 1.16183 w_k1_IBM_s006
                                 + 1.66087 w_k1_INTC_s006
                                 + 1.27146 w_k1_KO_s006 + 1.08995 w_k1_MAT_s006
                                 + 1.50846 w_k1_MSFT_s006
                                 + 1.31707 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.818548 w_k1_ADM_s006
                                 + 1.12087 w_k1_AXP_s006
                                 + 1.04225 w_k1_DIS_s006
                                 + 0.982444 w_k1_FDX_s006
                                 + 1.46459 w_k1_IBM_s006
                                 + 1.07506 w_k1_INTC_s006 + 1.1155 w_k1_KO_s006
                                 + 1.05649 w_k1_MAT_s006
                                 + 1.0834 w_k1_MSFT_s006 + 1.09952 w_k1_PG_s006
                                 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.14357 w_k1_ADM_s007
                                 + 1.00647 w_k1_AXP_s007
                                 + 0.963071 w_k1_DIS_s007
                                 + 1.0036 w_k1_FDX_s007 + 1.10138 w_k1_IBM_s007
                                 + 1.00851 w_k1_INTC_s007
                                 + 1.10767 w_k1_KO_s007 + 1.00266 w_k1_MAT_s007
                                 + 0.950697 w_k1_MSFT_s007
                                 + 1.02926 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.22485 w_k1_ADM_s007
                                 + 1.28103 w_k1_AXP_s007
                                 + 1.66639 w_k1_DIS_s007
                                 + 1.31376 w_k1_FDX_s007
                                 + 1.12376 w_k1_IBM_s007
                                 + 1.19149 w_k1_INTC_s007
                                 + 1.40076 w_k1_KO_s007 + 1.06096 w_k1_MAT_s007
                                 + 1.37161 w_k1_MSFT_s007
                                 + 1.12384 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.956751 w_k1_ADM_s007
                                 + 1.11691 w_k1_AXP_s007
                                 + 1.15697 w_k1_DIS_s007
                                 + 1.03804 w_k1_FDX_s007
                                 + 1.03733 w_k1_IBM_s007
                                 + 0.978879 w_k1_INTC_s007
                                 + 1.08933 w_k1_KO_s007
                                 + 0.912785 w_k1_MAT_s007
                                 + 1.06706 w_k1_MSFT_s007
                                 + 1.08781 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.00548 w_k1_ADM_s007
                                 + 1.08918 w_k1_AXP_s007
                                 + 1.25489 w_k1_DIS_s007
                                 + 1.21751 w_k1_FDX_s007
                                 + 0.812623 w_k1_IBM_s007
                                 + 1.05427 w_k1_INTC_s007
                                 + 0.905831 w_k1_KO_s007
                                 + 0.67493 w_k1_MAT_s007
                                 + 1.12935 w_k1_MSFT_s007
                                 + 0.948892 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.03792 w_k1_ADM_s007
                                 + 1.14864 w_k1_AXP_s007
                                 + 1.50712 w_k1_DIS_s007
                                 + 1.24801 w_k1_FDX_s007
                                 + 1.87768 w_k1_IBM_s007
                                 + 1.11975 w_k1_INTC_s007
                                 + 1.43843 w_k1_KO_s007 + 1.48312 w_k1_MAT_s007
                                 + 1.17278 w_k1_MSFT_s007
                                 + 1.10176 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.827131 w_k1_ADM_s007
                                 + 0.90755 w_k1_AXP_s007
                                 + 0.864354 w_k1_DIS_s007
                                 + 0.774166 w_k1_FDX_s007
                                 + 1.42971 w_k1_IBM_s007
                                 + 0.990028 w_k1_INTC_s007
                                 + 1.02295 w_k1_KO_s007
                                 + 0.962507 w_k1_MAT_s007
                                 + 0.828376 w_k1_MSFT_s007
                                 + 0.931723 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.04284 w_k1_ADM_s007
                                 + 0.920527 w_k1_AXP_s007
                                 + 0.834209 w_k1_DIS_s007
                                 + 1.00141 w_k1_FDX_s007
                                 + 0.759327 w_k1_IBM_s007
                                 + 1.08715 w_k1_INTC_s007
                                 + 0.852336 w_k1_KO_s007
                                 + 0.560713 w_k1_MAT_s007
                                 + 0.974562 w_k1_MSFT_s007
                                 + 0.899219 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.802282 w_k1_ADM_s007
                                 + 0.60202 w_k1_AXP_s007
                                 + 0.653931 w_k1_DIS_s007
                                 + 0.732263 w_k1_FDX_s007
                                 + 0.71702 w_k1_IBM_s007
                                 + 0.76712 w_k1_INTC_s007
                                 + 0.715541 w_k1_KO_s007
                                 + 0.580906 w_k1_MAT_s007
                                 + 0.833164 w_k1_MSFT_s007
                                 + 0.779274 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.760302 w_k1_ADM_s010
                                 + 1.11984 w_k1_AXP_s010
                                 + 1.05754 w_k1_DIS_s010
                                 + 1.07969 w_k1_FDX_s010
                                 + 0.863189 w_k1_IBM_s010
                                 + 1.269 w_k1_INTC_s010 + 0.939404 w_k1_KO_s010
                                 + 0.855329 w_k1_MAT_s010
                                 + 0.891681 w_k1_MSFT_s010
                                 + 1.05515 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.03994 w_k1_ADM_s010
                                 + 1.68134 w_k1_AXP_s010
                                 + 1.02424 w_k1_DIS_s010 + 1.2964 w_k1_FDX_s010
                                 + 1.19762 w_k1_IBM_s010
                                 + 1.22348 w_k1_INTC_s010
                                 + 0.900341 w_k1_KO_s010
                                 + 1.97992 w_k1_MAT_s010
                                 + 1.02295 w_k1_MSFT_s010 + 1.1069 w_k1_PG_s010
                                 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.932033 w_k1_ADM_s010
                                 + 1.36469 w_k1_AXP_s010
                                 + 1.23936 w_k1_DIS_s010
                                 + 1.08504 w_k1_FDX_s010
                                 + 1.42662 w_k1_IBM_s010
                                 + 1.33579 w_k1_INTC_s010
                                 + 1.25555 w_k1_KO_s010 + 1.15075 w_k1_MAT_s010
                                 + 0.971553 w_k1_MSFT_s010
                                 + 1.09572 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.842332 w_k1_ADM_s010
                                 + 1.26637 w_k1_AXP_s010
                                 + 1.45299 w_k1_DIS_s010
                                 + 1.33562 w_k1_FDX_s010 + 1.0595 w_k1_IBM_s010
                                 + 1.48166 w_k1_INTC_s010 + 1.1681 w_k1_KO_s010
                                 + 1.04136 w_k1_MAT_s010
                                 + 1.24925 w_k1_MSFT_s010
                                 + 1.13342 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.805283 w_k1_ADM_s010
                                 + 0.855536 w_k1_AXP_s010
                                 + 0.826549 w_k1_DIS_s010
                                 + 0.737495 w_k1_FDX_s010
                                 + 1.3923 w_k1_IBM_s010
                                 + 0.952198 w_k1_INTC_s010
                                 + 0.974048 w_k1_KO_s010
                                 + 1.30099 w_k1_MAT_s010
                                 + 0.767487 w_k1_MSFT_s010
                                 + 0.910955 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.697643 w_k1_ADM_s010
                                 + 0.829863 w_k1_AXP_s010
                                 + 0.57808 w_k1_DIS_s010
                                 + 0.784943 w_k1_FDX_s010
                                 + 0.692313 w_k1_IBM_s010
                                 + 0.93627 w_k1_INTC_s010
                                 + 0.683379 w_k1_KO_s010
                                 + 0.752008 w_k1_MAT_s010
                                 + 0.734557 w_k1_MSFT_s010
                                 + 0.826234 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.727217 w_k1_ADM_s010
                                 + 0.96591 w_k1_AXP_s010
                                 + 0.704956 w_k1_DIS_s010
                                 + 0.783866 w_k1_FDX_s010
                                 + 1.06685 w_k1_IBM_s010
                                 + 0.855021 w_k1_INTC_s010
                                 + 0.765926 w_k1_KO_s010
                                 + 0.848589 w_k1_MAT_s010
                                 + 0.720423 w_k1_MSFT_s010
                                 + 0.896077 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.907794 w_k1_ADM_s010
                                 + 0.786855 w_k1_AXP_s010
                                 + 0.83818 w_k1_DIS_s010
                                 + 0.927509 w_k1_FDX_s010
                                 + 0.898089 w_k1_IBM_s010
                                 + 1.15159 w_k1_INTC_s010
                                 + 0.87273 w_k1_KO_s010
                                 + 0.582665 w_k1_MAT_s010
                                 + 0.887989 w_k1_MSFT_s010
                                 + 0.908653 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.831424 w_k1_ADM_s011
                                 + 0.793306 w_k1_AXP_s011
                                 + 0.854395 w_k1_DIS_s011
                                 + 0.863256 w_k1_FDX_s011
                                 + 0.897183 w_k1_IBM_s011
                                 + 0.779543 w_k1_INTC_s011
                                 + 0.869179 w_k1_KO_s011
                                 + 0.873156 w_k1_MAT_s011
                                 + 0.933662 w_k1_MSFT_s011
                                 + 0.805536 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.09945 w_k1_ADM_s011
                                 + 1.22398 w_k1_AXP_s011
                                 + 0.973434 w_k1_DIS_s011
                                 + 1.09924 w_k1_FDX_s011
                                 + 1.10734 w_k1_IBM_s011
                                 + 1.08125 w_k1_INTC_s011
                                 + 1.03359 w_k1_KO_s011 + 1.29465 w_k1_MAT_s011
                                 + 1.13707 w_k1_MSFT_s011
                                 + 1.02653 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.793194 w_k1_ADM_s011
                                 + 0.846974 w_k1_AXP_s011
                                 + 0.810644 w_k1_DIS_s011
                                 + 0.935119 w_k1_FDX_s011
                                 + 0.695182 w_k1_IBM_s011
                                 + 0.865752 w_k1_INTC_s011
                                 + 0.705036 w_k1_KO_s011
                                 + 0.702727 w_k1_MAT_s011
                                 + 0.885538 w_k1_MSFT_s011
                                 + 0.835718 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.846504 w_k1_ADM_s011
                                 + 0.900035 w_k1_AXP_s011
                                 + 1.16094 w_k1_DIS_s011
                                 + 1.01338 w_k1_FDX_s011
                                 + 1.15116 w_k1_IBM_s011
                                 + 1.16975 w_k1_INTC_s011
                                 + 1.15414 w_k1_KO_s011
                                 + 0.927804 w_k1_MAT_s011
                                 + 1.10076 w_k1_MSFT_s011
                                 + 0.943879 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.929311 w_k1_ADM_s011
                                 + 0.927096 w_k1_AXP_s011
                                 + 1.13517 w_k1_DIS_s011
                                 + 1.10119 w_k1_FDX_s011
                                 + 0.815267 w_k1_IBM_s011
                                 + 1.05088 w_k1_INTC_s011
                                 + 0.934032 w_k1_KO_s011
                                 + 0.837024 w_k1_MAT_s011
                                 + 1.12101 w_k1_MSFT_s011
                                 + 1.04197 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.19495 w_k1_ADM_s011
                                 + 1.13922 w_k1_AXP_s011
                                 + 1.20581 w_k1_DIS_s011 + 1.1128 w_k1_FDX_s011
                                 + 1.4469 w_k1_IBM_s011
                                 + 0.855194 w_k1_INTC_s011
                                 + 1.07784 w_k1_KO_s011 + 1.42564 w_k1_MAT_s011
                                 + 0.975061 w_k1_MSFT_s011
                                 + 1.03717 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.902044 w_k1_ADM_s011
                                 + 0.687797 w_k1_AXP_s011
                                 + 0.713061 w_k1_DIS_s011
                                 + 0.685953 w_k1_FDX_s011
                                 + 0.944573 w_k1_IBM_s011
                                 + 0.8664 w_k1_INTC_s011
                                 + 0.918938 w_k1_KO_s011
                                 + 0.750224 w_k1_MAT_s011
                                 + 0.791906 w_k1_MSFT_s011
                                 + 0.842389 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.706993 w_k1_ADM_s011
                                 + 0.827965 w_k1_AXP_s011
                                 + 0.609889 w_k1_DIS_s011
                                 + 0.714498 w_k1_FDX_s011
                                 + 0.87742 w_k1_IBM_s011
                                 + 0.65156 w_k1_INTC_s011
                                 + 0.771494 w_k1_KO_s011
                                 + 0.951036 w_k1_MAT_s011
                                 + 0.727567 w_k1_MSFT_s011
                                 + 0.84423 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.45893 w_k1_ADM_s012
                                 + 1.61315 w_k1_AXP_s012 + 1.4589 w_k1_DIS_s012
                                 + 1.34738 w_k1_FDX_s012
                                 + 1.37827 w_k1_IBM_s012
                                 + 1.33492 w_k1_INTC_s012
                                 + 1.58873 w_k1_KO_s012 + 2.14445 w_k1_MAT_s012
                                 + 1.14583 w_k1_MSFT_s012
                                 + 1.42616 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.21431 w_k1_ADM_s012
                                 + 0.982799 w_k1_AXP_s012
                                 + 1.24068 w_k1_DIS_s012
                                 + 0.966441 w_k1_FDX_s012
                                 + 1.57996 w_k1_IBM_s012
                                 + 1.12157 w_k1_INTC_s012
                                 + 1.25017 w_k1_KO_s012 + 1.89468 w_k1_MAT_s012
                                 + 1.07648 w_k1_MSFT_s012
                                 + 1.24271 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.3501 w_k1_ADM_s012
                                 + 1.36389 w_k1_AXP_s012
                                 + 2.07416 w_k1_DIS_s012
                                 + 1.39263 w_k1_FDX_s012
                                 + 1.32675 w_k1_IBM_s012
                                 + 1.29496 w_k1_INTC_s012
                                 + 1.65676 w_k1_KO_s012 + 2.53263 w_k1_MAT_s012
                                 + 1.29762 w_k1_MSFT_s012
                                 + 1.35865 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.821573 w_k1_ADM_s012
                                 + 0.76269 w_k1_AXP_s012
                                 + 0.581771 w_k1_DIS_s012
                                 + 0.717933 w_k1_FDX_s012
                                 + 0.931396 w_k1_IBM_s012
                                 + 0.951608 w_k1_INTC_s012
                                 + 0.791612 w_k1_KO_s012
                                 + 0.840739 w_k1_MAT_s012
                                 + 0.717459 w_k1_MSFT_s012
                                 + 0.902404 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.02864 w_k1_ADM_s012
                                 + 0.974114 w_k1_AXP_s012
                                 + 0.823019 w_k1_DIS_s012
                                 + 0.907036 w_k1_FDX_s012
                                 + 0.893841 w_k1_IBM_s012
                                 + 1.05778 w_k1_INTC_s012
                                 + 0.888063 w_k1_KO_s012
                                 + 1.02725 w_k1_MAT_s012
                                 + 0.849794 w_k1_MSFT_s012
                                 + 1.07877 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.721421 w_k1_ADM_s012
                                 + 0.651598 w_k1_AXP_s012
                                 + 0.847783 w_k1_DIS_s012
                                 + 0.844962 w_k1_FDX_s012
                                 + 0.717542 w_k1_IBM_s012
                                 + 0.933071 w_k1_INTC_s012
                                 + 0.804074 w_k1_KO_s012
                                 + 0.716549 w_k1_MAT_s012
                                 + 0.880274 w_k1_MSFT_s012
                                 + 0.915451 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.22792 w_k1_ADM_s012
                                 + 1.1895 w_k1_AXP_s012 + 1.11483 w_k1_DIS_s012
                                 + 1.14274 w_k1_FDX_s012
                                 + 1.24643 w_k1_IBM_s012
                                 + 1.19153 w_k1_INTC_s012
                                 + 0.994011 w_k1_KO_s012
                                 + 1.43741 w_k1_MAT_s012
                                 + 1.01902 w_k1_MSFT_s012
                                 + 1.14758 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.32764 w_k1_ADM_s012
                                 + 0.851426 w_k1_AXP_s012
                                 + 1.12702 w_k1_DIS_s012
                                 + 1.08774 w_k1_FDX_s012
                                 + 0.753582 w_k1_IBM_s012
                                 + 1.15816 w_k1_INTC_s012
                                 + 0.916906 w_k1_KO_s012
                                 + 0.847946 w_k1_MAT_s012
                                 + 1.04784 w_k1_MSFT_s012
                                 + 1.05232 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.877819 w_k1_ADM_s013
                                 + 0.790444 w_k1_AXP_s013
                                 + 0.796662 w_k1_DIS_s013
                                 + 0.820035 w_k1_FDX_s013
                                 + 1.4339 w_k1_IBM_s013
                                 + 0.934484 w_k1_INTC_s013
                                 + 1.01609 w_k1_KO_s013 + 1.1945 w_k1_MAT_s013
                                 + 0.845954 w_k1_MSFT_s013
                                 + 1.06005 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.04591 w_k1_ADM_s013
                                 + 0.940649 w_k1_AXP_s013
                                 + 1.07842 w_k1_DIS_s013
                                 + 1.12914 w_k1_FDX_s013
                                 + 1.10525 w_k1_IBM_s013
                                 + 1.04856 w_k1_INTC_s013
                                 + 1.01552 w_k1_KO_s013
                                 + 0.895853 w_k1_MAT_s013
                                 + 1.05673 w_k1_MSFT_s013
                                 + 1.05577 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.724031 w_k1_ADM_s013
                                 + 0.607373 w_k1_AXP_s013
                                 + 0.749852 w_k1_DIS_s013
                                 + 0.887437 w_k1_FDX_s013
                                 + 0.660142 w_k1_IBM_s013
                                 + 0.872014 w_k1_INTC_s013
                                 + 0.695459 w_k1_KO_s013
                                 + 0.417498 w_k1_MAT_s013
                                 + 0.840134 w_k1_MSFT_s013
                                 + 0.829835 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.990171 w_k1_ADM_s013
                                 + 1.46384 w_k1_AXP_s013
                                 + 1.41631 w_k1_DIS_s013
                                 + 1.30618 w_k1_FDX_s013
                                 + 1.68206 w_k1_IBM_s013
                                 + 1.33256 w_k1_INTC_s013 + 1.3294 w_k1_KO_s013
                                 + 1.57432 w_k1_MAT_s013
                                 + 1.17617 w_k1_MSFT_s013
                                 + 1.13319 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.06487 w_k1_ADM_s013
                                 + 1.37846 w_k1_AXP_s013
                                 + 1.51115 w_k1_DIS_s013
                                 + 1.37385 w_k1_FDX_s013
                                 + 1.03028 w_k1_IBM_s013
                                 + 1.31216 w_k1_INTC_s013
                                 + 1.09163 w_k1_KO_s013
                                 + 0.770262 w_k1_MAT_s013
                                 + 1.32424 w_k1_MSFT_s013 + 1.1598 w_k1_PG_s013
                                 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.11146 w_k1_ADM_s013
                                 + 1.09246 w_k1_AXP_s013
                                 + 1.14007 w_k1_DIS_s013
                                 + 1.13586 w_k1_FDX_s013
                                 + 1.44949 w_k1_IBM_s013
                                 + 1.17932 w_k1_INTC_s013
                                 + 1.21753 w_k1_KO_s013 + 1.1589 w_k1_MAT_s013
                                 + 1.14395 w_k1_MSFT_s013
                                 + 1.08656 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.940291 w_k1_ADM_s013
                                 + 1.09894 w_k1_AXP_s013
                                 + 0.944399 w_k1_DIS_s013
                                 + 1.07464 w_k1_FDX_s013
                                 + 1.01052 w_k1_IBM_s013
                                 + 1.0003 w_k1_INTC_s013 + 0.94044 w_k1_KO_s013
                                 + 0.960611 w_k1_MAT_s013
                                 + 0.962369 w_k1_MSFT_s013
                                 + 1.01564 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.726144 w_k1_ADM_s013
                                 + 0.757783 w_k1_AXP_s013
                                 + 0.78521 w_k1_DIS_s013
                                 + 0.833968 w_k1_FDX_s013
                                 + 1.00828 w_k1_IBM_s013
                                 + 1.03866 w_k1_INTC_s013
                                 + 0.839002 w_k1_KO_s013
                                 + 0.891429 w_k1_MAT_s013
                                 + 0.914768 w_k1_MSFT_s013
                                 + 0.788535 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.35025 w_k1_ADM_s014
                                 + 1.48153 w_k1_AXP_s014
                                 + 1.52681 w_k1_DIS_s014
                                 + 1.41697 w_k1_FDX_s014
                                 + 1.08009 w_k1_IBM_s014
                                 + 1.47409 w_k1_INTC_s014
                                 + 1.24742 w_k1_KO_s014 + 1.25323 w_k1_MAT_s014
                                 + 1.38708 w_k1_MSFT_s014
                                 + 1.38491 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.11495 w_k1_ADM_s014
                                 + 0.846134 w_k1_AXP_s014
                                 + 0.866447 w_k1_DIS_s014
                                 + 1.04026 w_k1_FDX_s014
                                 + 0.618045 w_k1_IBM_s014
                                 + 1.11926 w_k1_INTC_s014
                                 + 0.900627 w_k1_KO_s014
                                 + 0.512678 w_k1_MAT_s014
                                 + 0.98267 w_k1_MSFT_s014
                                 + 0.992023 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.958007 w_k1_ADM_s014
                                 + 1.04389 w_k1_AXP_s014
                                 + 0.823968 w_k1_DIS_s014
                                 + 0.914369 w_k1_FDX_s014
                                 + 1.16899 w_k1_IBM_s014
                                 + 0.95429 w_k1_INTC_s014
                                 + 0.963685 w_k1_KO_s014 + 1.0161 w_k1_MAT_s014
                                 + 0.925195 w_k1_MSFT_s014
                                 + 1.09911 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.1193 w_k1_ADM_s014
                                 + 1.11003 w_k1_AXP_s014
                                 + 1.23327 w_k1_DIS_s014 + 1.2275 w_k1_FDX_s014
                                 + 1.10464 w_k1_IBM_s014
                                 + 1.39848 w_k1_INTC_s014
                                 + 1.14947 w_k1_KO_s014 + 1.17562 w_k1_MAT_s014
                                 + 1.24873 w_k1_MSFT_s014
                                 + 1.11335 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.985662 w_k1_ADM_s014
                                 + 1.42422 w_k1_AXP_s014
                                 + 1.44054 w_k1_DIS_s014
                                 + 1.14879 w_k1_FDX_s014
                                 + 1.64345 w_k1_IBM_s014
                                 + 1.05321 w_k1_INTC_s014 + 1.5647 w_k1_KO_s014
                                 + 1.62269 w_k1_MAT_s014
                                 + 1.12298 w_k1_MSFT_s014
                                 + 1.14083 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.06599 w_k1_ADM_s014
                                 + 1.03242 w_k1_AXP_s014
                                 + 1.00922 w_k1_DIS_s014
                                 + 1.13319 w_k1_FDX_s014
                                 + 0.81768 w_k1_IBM_s014
                                 + 1.01761 w_k1_INTC_s014
                                 + 1.09111 w_k1_KO_s014
                                 + 0.922399 w_k1_MAT_s014
                                 + 1.13344 w_k1_MSFT_s014
                                 + 1.09331 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.04281 w_k1_ADM_s014
                                 + 0.975536 w_k1_AXP_s014
                                 + 1.14516 w_k1_DIS_s014
                                 + 1.00905 w_k1_FDX_s014
                                 + 1.28879 w_k1_IBM_s014
                                 + 1.00882 w_k1_INTC_s014
                                 + 1.29676 w_k1_KO_s014 + 1.13782 w_k1_MAT_s014
                                 + 1.09908 w_k1_MSFT_s014
                                 + 1.19698 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.851707 w_k1_ADM_s014
                                 + 0.802987 w_k1_AXP_s014
                                 + 0.653461 w_k1_DIS_s014
                                 + 0.792062 w_k1_FDX_s014
                                 + 0.829846 w_k1_IBM_s014
                                 + 1.08403 w_k1_INTC_s014
                                 + 0.911421 w_k1_KO_s014
                                 + 0.756783 w_k1_MAT_s014
                                 + 0.856243 w_k1_MSFT_s014
                                 + 0.925284 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.950272 w_k1_ADM_s015
                                 + 1.32946 w_k1_AXP_s015
                                 + 0.867017 w_k1_DIS_s015
                                 + 1.11237 w_k1_FDX_s015
                                 + 0.90088 w_k1_IBM_s015
                                 + 1.05238 w_k1_INTC_s015
                                 + 0.918844 w_k1_KO_s015
                                 + 2.30814 w_k1_MAT_s015
                                 + 1.13233 w_k1_MSFT_s015
                                 + 1.01822 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.79438 w_k1_ADM_s015
                                 + 0.706822 w_k1_AXP_s015
                                 + 0.618552 w_k1_DIS_s015
                                 + 0.763067 w_k1_FDX_s015
                                 + 0.918429 w_k1_IBM_s015
                                 + 0.997204 w_k1_INTC_s015
                                 + 0.819348 w_k1_KO_s015
                                 + 0.987996 w_k1_MAT_s015
                                 + 0.879593 w_k1_MSFT_s015
                                 + 0.71788 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.945141 w_k1_ADM_s015
                                 + 1.00043 w_k1_AXP_s015
                                 + 0.996226 w_k1_DIS_s015
                                 + 0.998607 w_k1_FDX_s015
                                 + 1.21428 w_k1_IBM_s015
                                 + 1.19583 w_k1_INTC_s015
                                 + 1.04546 w_k1_KO_s015 + 1.22081 w_k1_MAT_s015
                                 + 1.07422 w_k1_MSFT_s015
                                 + 0.938029 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.00281 w_k1_ADM_s015
                                 + 1.28119 w_k1_AXP_s015
                                 + 1.14784 w_k1_DIS_s015
                                 + 1.17917 w_k1_FDX_s015 + 1.0475 w_k1_IBM_s015
                                 + 1.55745 w_k1_INTC_s015
                                 + 1.14262 w_k1_KO_s015 + 2.02215 w_k1_MAT_s015
                                 + 1.28829 w_k1_MSFT_s015
                                 + 1.09315 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.983834 w_k1_ADM_s015
                                 + 1.57273 w_k1_AXP_s015
                                 + 1.50091 w_k1_DIS_s015
                                 + 1.39721 w_k1_FDX_s015
                                 + 0.960011 w_k1_IBM_s015
                                 + 1.61822 w_k1_INTC_s015
                                 + 1.38862 w_k1_KO_s015 + 1.73693 w_k1_MAT_s015
                                 + 1.34975 w_k1_MSFT_s015 + 1.1575 w_k1_PG_s015
                                 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.921815 w_k1_ADM_s015
                                 + 0.950092 w_k1_AXP_s015
                                 + 1.43588 w_k1_DIS_s015
                                 + 1.07901 w_k1_FDX_s015
                                 + 1.57256 w_k1_IBM_s015
                                 + 1.21981 w_k1_INTC_s015
                                 + 1.29883 w_k1_KO_s015 + 2.06637 w_k1_MAT_s015
                                 + 1.31442 w_k1_MSFT_s015
                                 + 1.05162 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.866186 w_k1_ADM_s015
                                 + 0.84785 w_k1_AXP_s015
                                 + 1.00375 w_k1_DIS_s015
                                 + 0.871786 w_k1_FDX_s015
                                 + 1.17272 w_k1_IBM_s015
                                 + 1.15893 w_k1_INTC_s015
                                 + 1.22939 w_k1_KO_s015 + 1.39727 w_k1_MAT_s015
                                 + 1.06429 w_k1_MSFT_s015
                                 + 0.977515 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.956432 w_k1_ADM_s015
                                 + 0.664601 w_k1_AXP_s015
                                 + 0.777868 w_k1_DIS_s015
                                 + 0.837891 w_k1_FDX_s015
                                 + 0.692061 w_k1_IBM_s015
                                 + 0.981237 w_k1_INTC_s015
                                 + 0.753919 w_k1_KO_s015
                                 + 0.935534 w_k1_MAT_s015
                                 + 1.02999 w_k1_MSFT_s015
                                 + 0.790315 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.792891 w_k1_ADM_s016
                                 + 0.642654 w_k1_AXP_s016
                                 + 0.769597 w_k1_DIS_s016
                                 + 0.773442 w_k1_FDX_s016
                                 + 1.12419 w_k1_IBM_s016
                                 + 0.832651 w_k1_INTC_s016
                                 + 1.08311 w_k1_KO_s016
                                 + 0.891088 w_k1_MAT_s016
                                 + 0.760228 w_k1_MSFT_s016
                                 + 0.872635 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.621297 w_k1_ADM_s016
                                 + 0.864386 w_k1_AXP_s016
                                 + 0.71722 w_k1_DIS_s016
                                 + 0.815886 w_k1_FDX_s016
                                 + 0.777045 w_k1_IBM_s016
                                 + 0.875869 w_k1_INTC_s016
                                 + 0.897417 w_k1_KO_s016
                                 + 0.887658 w_k1_MAT_s016
                                 + 0.791722 w_k1_MSFT_s016
                                 + 0.868897 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.976774 w_k1_ADM_s016
                                 + 1.02155 w_k1_AXP_s016
                                 + 0.909299 w_k1_DIS_s016
                                 + 0.99394 w_k1_FDX_s016
                                 + 0.977132 w_k1_IBM_s016
                                 + 1.02099 w_k1_INTC_s016
                                 + 0.977213 w_k1_KO_s016
                                 + 1.03908 w_k1_MAT_s016
                                 + 1.03646 w_k1_MSFT_s016
                                 + 1.03461 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.00631 w_k1_ADM_s016
                                 + 1.19939 w_k1_AXP_s016
                                 + 1.71079 w_k1_DIS_s016
                                 + 1.19184 w_k1_FDX_s016
                                 + 1.72456 w_k1_IBM_s016
                                 + 1.14973 w_k1_INTC_s016
                                 + 1.65906 w_k1_KO_s016 + 1.6509 w_k1_MAT_s016
                                 + 1.41358 w_k1_MSFT_s016
                                 + 1.29257 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.788361 w_k1_ADM_s016
                                 + 0.815216 w_k1_AXP_s016
                                 + 1.17754 w_k1_DIS_s016
                                 + 1.07113 w_k1_FDX_s016
                                 + 0.844671 w_k1_IBM_s016
                                 + 1.07393 w_k1_INTC_s016
                                 + 1.13799 w_k1_KO_s016
                                 + 0.700184 w_k1_MAT_s016
                                 + 0.988789 w_k1_MSFT_s016
                                 + 1.0601 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.945742 w_k1_ADM_s016
                                 + 1.42609 w_k1_AXP_s016
                                 + 1.38827 w_k1_DIS_s016
                                 + 1.31775 w_k1_FDX_s016
                                 + 1.27466 w_k1_IBM_s016
                                 + 1.23904 w_k1_INTC_s016 + 1.509 w_k1_KO_s016
                                 + 1.48179 w_k1_MAT_s016
                                 + 1.25329 w_k1_MSFT_s016
                                 + 1.35377 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.09384 w_k1_ADM_s016
                                 + 1.22602 w_k1_AXP_s016
                                 + 1.13162 w_k1_DIS_s016
                                 + 1.04155 w_k1_FDX_s016
                                 + 1.41281 w_k1_IBM_s016
                                 + 1.15005 w_k1_INTC_s016
                                 + 1.42058 w_k1_KO_s016 + 1.43174 w_k1_MAT_s016
                                 + 1.06372 w_k1_MSFT_s016
                                 + 1.18506 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.0944 w_k1_ADM_s016
                                 + 1.35779 w_k1_AXP_s016
                                 + 1.09617 w_k1_DIS_s016
                                 + 1.28571 w_k1_FDX_s016
                                 + 0.925759 w_k1_IBM_s016
                                 + 1.1717 w_k1_INTC_s016 + 1.15291 w_k1_KO_s016
                                 + 0.77961 w_k1_MAT_s016
                                 + 1.11016 w_k1_MSFT_s016
                                 + 0.974848 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.640559 w_k1_ADM_s017
                                 + 0.721005 w_k1_AXP_s017
                                 + 0.755871 w_k1_DIS_s017
                                 + 0.759477 w_k1_FDX_s017
                                 + 1.12797 w_k1_IBM_s017
                                 + 0.943585 w_k1_INTC_s017
                                 + 0.900779 w_k1_KO_s017
                                 + 0.942554 w_k1_MAT_s017
                                 + 0.813236 w_k1_MSFT_s017
                                 + 0.762522 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.580398 w_k1_ADM_s017
                                 + 0.676497 w_k1_AXP_s017
                                 + 1.00942 w_k1_DIS_s017
                                 + 0.868337 w_k1_FDX_s017
                                 + 0.779804 w_k1_IBM_s017
                                 + 1.11342 w_k1_INTC_s017
                                 + 0.998527 w_k1_KO_s017
                                 + 0.656122 w_k1_MAT_s017
                                 + 1.03333 w_k1_MSFT_s017
                                 + 0.830247 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.69926 w_k1_ADM_s017
                                 + 0.75896 w_k1_AXP_s017
                                 + 0.647679 w_k1_DIS_s017
                                 + 0.808467 w_k1_FDX_s017
                                 + 0.862355 w_k1_IBM_s017
                                 + 1.10653 w_k1_INTC_s017
                                 + 0.847711 w_k1_KO_s017
                                 + 0.721184 w_k1_MAT_s017
                                 + 0.984229 w_k1_MSFT_s017
                                 + 0.921102 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.898289 w_k1_ADM_s017
                                 + 1.12195 w_k1_AXP_s017
                                 + 1.50248 w_k1_DIS_s017
                                 + 1.19822 w_k1_FDX_s017
                                 + 1.08104 w_k1_IBM_s017
                                 + 1.21682 w_k1_INTC_s017
                                 + 1.19151 w_k1_KO_s017
                                 + 0.950288 w_k1_MAT_s017
                                 + 1.40503 w_k1_MSFT_s017
                                 + 0.989301 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.910041 w_k1_ADM_s017
                                 + 1.02639 w_k1_AXP_s017
                                 + 1.15983 w_k1_DIS_s017
                                 + 1.15649 w_k1_FDX_s017
                                 + 0.726228 w_k1_IBM_s017
                                 + 1.31661 w_k1_INTC_s017
                                 + 0.904279 w_k1_KO_s017
                                 + 0.624517 w_k1_MAT_s017
                                 + 1.21363 w_k1_MSFT_s017
                                 + 0.986869 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.917383 w_k1_ADM_s017
                                 + 1.3564 w_k1_AXP_s017
                                 + 0.940536 w_k1_DIS_s017
                                 + 1.01124 w_k1_FDX_s017 + 1.121 w_k1_IBM_s017
                                 + 1.26492 w_k1_INTC_s017
                                 + 0.966818 w_k1_KO_s017
                                 + 1.19276 w_k1_MAT_s017
                                 + 0.988164 w_k1_MSFT_s017
                                 + 1.06216 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.12273 w_k1_ADM_s017
                                 + 1.20727 w_k1_AXP_s017
                                 + 1.43482 w_k1_DIS_s017
                                 + 1.14438 w_k1_FDX_s017
                                 + 1.52736 w_k1_IBM_s017
                                 + 1.31951 w_k1_INTC_s017
                                 + 1.45426 w_k1_KO_s017 + 1.08413 w_k1_MAT_s017
                                 + 1.21253 w_k1_MSFT_s017
                                 + 1.05701 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.523924 w_k1_ADM_s017
                                 + 0.684464 w_k1_AXP_s017
                                 + 0.569313 w_k1_DIS_s017
                                 + 0.757957 w_k1_FDX_s017
                                 + 0.606819 w_k1_IBM_s017
                                 + 1.06277 w_k1_INTC_s017
                                 + 0.663537 w_k1_KO_s017
                                 + 0.472651 w_k1_MAT_s017
                                 + 0.878599 w_k1_MSFT_s017
                                 + 0.808351 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.996645 w_k1_ADM_s100
                                 + 1.08979 w_k1_AXP_s100
                                 + 1.13163 w_k1_DIS_s100
                                 + 1.13058 w_k1_FDX_s100
                                 + 0.803059 w_k1_IBM_s100
                                 + 1.25346 w_k1_INTC_s100
                                 + 0.985775 w_k1_KO_s100
                                 + 0.888775 w_k1_MAT_s100
                                 + 1.09691 w_k1_MSFT_s100
                                 + 1.04676 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.1438 w_k1_ADM_s100
                                 + 1.81093 w_k1_AXP_s100
                                 + 1.36967 w_k1_DIS_s100
                                 + 1.40543 w_k1_FDX_s100
                                 + 1.10492 w_k1_IBM_s100
                                 + 1.61255 w_k1_INTC_s100
                                 + 1.20374 w_k1_KO_s100
                                 + 0.712851 w_k1_MAT_s100
                                 + 1.24908 w_k1_MSFT_s100
                                 + 1.24164 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.01964 w_k1_ADM_s100
                                 + 1.05841 w_k1_AXP_s100
                                 + 1.04416 w_k1_DIS_s100
                                 + 1.00404 w_k1_FDX_s100
                                 + 1.60684 w_k1_IBM_s100
                                 + 1.06119 w_k1_INTC_s100
                                 + 1.15692 w_k1_KO_s100 + 1.39707 w_k1_MAT_s100
                                 + 0.931632 w_k1_MSFT_s100
                                 + 0.935939 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.29042 w_k1_ADM_s100
                                 + 1.14378 w_k1_AXP_s100
                                 + 0.906358 w_k1_DIS_s100
                                 + 1.02267 w_k1_FDX_s100
                                 + 1.02004 w_k1_IBM_s100
                                 + 1.08521 w_k1_INTC_s100
                                 + 0.99034 w_k1_KO_s100 + 0.97336 w_k1_MAT_s100
                                 + 0.807653 w_k1_MSFT_s100
                                 + 1.08415 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.809387 w_k1_ADM_s100
                                 + 0.792475 w_k1_AXP_s100
                                 + 0.627761 w_k1_DIS_s100
                                 + 0.846013 w_k1_FDX_s100
                                 + 0.791843 w_k1_IBM_s100
                                 + 1.06895 w_k1_INTC_s100
                                 + 0.720452 w_k1_KO_s100
                                 + 0.888117 w_k1_MAT_s100
                                 + 0.88666 w_k1_MSFT_s100
                                 + 0.911125 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.939527 w_k1_ADM_s100
                                 + 0.880167 w_k1_AXP_s100
                                 + 0.862926 w_k1_DIS_s100
                                 + 0.846308 w_k1_FDX_s100
                                 + 0.702993 w_k1_IBM_s100
                                 + 1.00544 w_k1_INTC_s100
                                 + 0.790373 w_k1_KO_s100
                                 + 0.524937 w_k1_MAT_s100
                                 + 0.959528 w_k1_MSFT_s100
                                 + 0.856609 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.871822 w_k1_ADM_s100
                                 + 0.541751 w_k1_AXP_s100
                                 + 0.48507 w_k1_DIS_s100
                                 + 0.692636 w_k1_FDX_s100
                                 + 0.52394 w_k1_IBM_s100
                                 + 0.712858 w_k1_INTC_s100
                                 + 0.610141 w_k1_KO_s100
                                 + 0.486482 w_k1_MAT_s100
                                 + 0.683249 w_k1_MSFT_s100
                                 + 0.742118 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.46006 w_k1_ADM_s100
                                 + 1.75265 w_k1_AXP_s100
                                 + 1.56203 w_k1_DIS_s100
                                 + 1.25505 w_k1_FDX_s100
                                 + 1.76878 w_k1_IBM_s100
                                 + 1.58059 w_k1_INTC_s100
                                 + 1.59594 w_k1_KO_s100 + 1.59998 w_k1_MAT_s100
                                 + 1.30348 w_k1_MSFT_s100
                                 + 1.24154 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.964916 w_k1_ADM_s101
                                 + 1.36314 w_k1_AXP_s101
                                 + 1.02791 w_k1_DIS_s101
                                 + 1.17582 w_k1_FDX_s101
                                 + 0.981256 w_k1_IBM_s101
                                 + 1.1663 w_k1_INTC_s101
                                 + 0.949942 w_k1_KO_s101
                                 + 1.40042 w_k1_MAT_s101
                                 + 0.981256 w_k1_MSFT_s101
                                 + 1.04568 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.16767 w_k1_ADM_s101
                                 + 1.59285 w_k1_AXP_s101
                                 + 1.64409 w_k1_DIS_s101
                                 + 1.21385 w_k1_FDX_s101 + 1.7958 w_k1_IBM_s101
                                 + 0.971161 w_k1_INTC_s101
                                 + 1.33954 w_k1_KO_s101 + 1.52125 w_k1_MAT_s101
                                 + 0.940146 w_k1_MSFT_s101
                                 + 1.20843 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.59138 w_k1_ADM_s101
                                 + 1.36645 w_k1_AXP_s101
                                 + 1.70521 w_k1_DIS_s101
                                 + 1.29759 w_k1_FDX_s101
                                 + 1.16243 w_k1_IBM_s101 + 1.039 w_k1_INTC_s101
                                 + 1.1344 w_k1_KO_s101 + 0.857882 w_k1_MAT_s101
                                 + 1.02712 w_k1_MSFT_s101
                                 + 1.03325 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.59488 w_k1_ADM_s101
                                 + 1.61704 w_k1_AXP_s101
                                 + 1.55047 w_k1_DIS_s101
                                 + 1.39467 w_k1_FDX_s101
                                 + 1.38331 w_k1_IBM_s101
                                 + 1.29099 w_k1_INTC_s101
                                 + 1.37935 w_k1_KO_s101 + 1.85066 w_k1_MAT_s101
                                 + 1.40318 w_k1_MSFT_s101
                                 + 1.32145 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.759842 w_k1_ADM_s101
                                 + 0.830691 w_k1_AXP_s101
                                 + 0.922776 w_k1_DIS_s101
                                 + 0.904569 w_k1_FDX_s101
                                 + 0.777329 w_k1_IBM_s101
                                 + 0.951739 w_k1_INTC_s101
                                 + 0.905648 w_k1_KO_s101
                                 + 0.822756 w_k1_MAT_s101
                                 + 0.988526 w_k1_MSFT_s101
                                 + 0.950386 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.953216 w_k1_ADM_s101
                                 + 0.930169 w_k1_AXP_s101
                                 + 1.11563 w_k1_DIS_s101
                                 + 0.944681 w_k1_FDX_s101
                                 + 1.10038 w_k1_IBM_s101
                                 + 0.95306 w_k1_INTC_s101
                                 + 0.942164 w_k1_KO_s101
                                 + 0.802716 w_k1_MAT_s101
                                 + 0.937539 w_k1_MSFT_s101
                                 + 0.998945 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.841179 w_k1_ADM_s101
                                 + 0.902155 w_k1_AXP_s101
                                 + 0.622319 w_k1_DIS_s101
                                 + 0.822838 w_k1_FDX_s101
                                 + 0.665529 w_k1_IBM_s101
                                 + 0.84592 w_k1_INTC_s101
                                 + 0.651367 w_k1_KO_s101
                                 + 0.647281 w_k1_MAT_s101
                                 + 0.793065 w_k1_MSFT_s101
                                 + 0.791592 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.04922 w_k1_ADM_s101
                                 + 1.16493 w_k1_AXP_s101
                                 + 1.04559 w_k1_DIS_s101
                                 + 0.891654 w_k1_FDX_s101
                                 + 1.62746 w_k1_IBM_s101
                                 + 0.961388 w_k1_INTC_s101
                                 + 1.07295 w_k1_KO_s101 + 1.12175 w_k1_MAT_s101
                                 + 0.813985 w_k1_MSFT_s101
                                 + 0.932483 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.834998 w_k1_ADM_s102
                                 + 0.69968 w_k1_AXP_s102
                                 + 0.935384 w_k1_DIS_s102
                                 + 0.815323 w_k1_FDX_s102
                                 + 1.0171 w_k1_IBM_s102
                                 + 0.999099 w_k1_INTC_s102
                                 + 0.966826 w_k1_KO_s102
                                 + 0.869287 w_k1_MAT_s102
                                 + 0.824541 w_k1_MSFT_s102
                                 + 0.866641 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.17079 w_k1_ADM_s102
                                 + 0.985892 w_k1_AXP_s102
                                 + 1.0821 w_k1_DIS_s102 + 1.10207 w_k1_FDX_s102
                                 + 0.837918 w_k1_IBM_s102
                                 + 1.12156 w_k1_INTC_s102
                                 + 0.933288 w_k1_KO_s102
                                 + 0.877277 w_k1_MAT_s102
                                 + 1.0517 w_k1_MSFT_s102
                                 + 0.997724 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.904821 w_k1_ADM_s102
                                 + 0.904488 w_k1_AXP_s102
                                 + 1.46419 w_k1_DIS_s102
                                 + 1.12924 w_k1_FDX_s102
                                 + 0.80723 w_k1_IBM_s102
                                 + 1.27883 w_k1_INTC_s102
                                 + 1.05422 w_k1_KO_s102
                                 + 0.688392 w_k1_MAT_s102
                                 + 1.12119 w_k1_MSFT_s102
                                 + 0.969214 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.06531 w_k1_ADM_s102
                                 + 1.13204 w_k1_AXP_s102
                                 + 1.31391 w_k1_DIS_s102
                                 + 1.16603 w_k1_FDX_s102
                                 + 1.51671 w_k1_IBM_s102
                                 + 1.31417 w_k1_INTC_s102
                                 + 1.32966 w_k1_KO_s102 + 1.68004 w_k1_MAT_s102
                                 + 1.22065 w_k1_MSFT_s102 + 1.1065 w_k1_PG_s102
                                 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.11765 w_k1_ADM_s102
                                 + 0.757197 w_k1_AXP_s102
                                 + 1.08252 w_k1_DIS_s102
                                 + 1.01308 w_k1_FDX_s102
                                 + 1.11644 w_k1_IBM_s102
                                 + 1.21332 w_k1_INTC_s102
                                 + 1.27444 w_k1_KO_s102 + 1.04367 w_k1_MAT_s102
                                 + 1.02375 w_k1_MSFT_s102
                                 + 1.05426 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.917709 w_k1_ADM_s102
                                 + 0.66291 w_k1_AXP_s102
                                 + 0.691354 w_k1_DIS_s102
                                 + 0.765858 w_k1_FDX_s102
                                 + 0.672553 w_k1_IBM_s102
                                 + 0.984193 w_k1_INTC_s102
                                 + 0.852281 w_k1_KO_s102
                                 + 0.753591 w_k1_MAT_s102
                                 + 0.798608 w_k1_MSFT_s102
                                 + 0.867693 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.57368 w_k1_ADM_s102
                                 + 1.23453 w_k1_AXP_s102
                                 + 1.83322 w_k1_DIS_s102
                                 + 1.43507 w_k1_FDX_s102
                                 + 1.15615 w_k1_IBM_s102
                                 + 1.59259 w_k1_INTC_s102
                                 + 1.33969 w_k1_KO_s102 + 1.40626 w_k1_MAT_s102
                                 + 1.51087 w_k1_MSFT_s102
                                 + 1.38026 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.05847 w_k1_ADM_s102
                                 + 0.891323 w_k1_AXP_s102
                                 + 0.902419 w_k1_DIS_s102
                                 + 0.985002 w_k1_FDX_s102
                                 + 0.581382 w_k1_IBM_s102
                                 + 0.9054 w_k1_INTC_s102
                                 + 0.686759 w_k1_KO_s102
                                 + 0.474488 w_k1_MAT_s102
                                 + 0.906721 w_k1_MSFT_s102
                                 + 0.867381 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.752654 w_k1_ADM_s103
                                 + 0.754947 w_k1_AXP_s103
                                 + 0.73077 w_k1_DIS_s103
                                 + 0.703524 w_k1_FDX_s103
                                 + 1.01253 w_k1_IBM_s103
                                 + 0.932541 w_k1_INTC_s103
                                 + 0.836688 w_k1_KO_s103
                                 + 0.830158 w_k1_MAT_s103
                                 + 0.863743 w_k1_MSFT_s103
                                 + 0.874154 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.99379 w_k1_ADM_s103
                                 + 1.20607 w_k1_AXP_s103
                                 + 1.22053 w_k1_DIS_s103
                                 + 1.07657 w_k1_FDX_s103
                                 + 1.51629 w_k1_IBM_s103
                                 + 1.2101 w_k1_INTC_s103 + 1.12635 w_k1_KO_s103
                                 + 1.69702 w_k1_MAT_s103
                                 + 1.10656 w_k1_MSFT_s103 + 1.1397 w_k1_PG_s103
                                 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.07612 w_k1_ADM_s103
                                 + 1.25643 w_k1_AXP_s103 + 1.2846 w_k1_DIS_s103
                                 + 1.1962 w_k1_FDX_s103 + 1.2752 w_k1_IBM_s103
                                 + 1.4334 w_k1_INTC_s103 + 1.12845 w_k1_KO_s103
                                 + 1.30103 w_k1_MAT_s103
                                 + 1.22204 w_k1_MSFT_s103
                                 + 1.16893 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.921542 w_k1_ADM_s103
                                 + 0.827306 w_k1_AXP_s103
                                 + 1.14388 w_k1_DIS_s103
                                 + 0.980749 w_k1_FDX_s103
                                 + 0.869479 w_k1_IBM_s103
                                 + 1.27269 w_k1_INTC_s103
                                 + 0.87521 w_k1_KO_s103
                                 + 0.539258 w_k1_MAT_s103
                                 + 1.19061 w_k1_MSFT_s103
                                 + 0.923722 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.943596 w_k1_ADM_s103
                                 + 1.0448 w_k1_AXP_s103
                                 + 0.862493 w_k1_DIS_s103
                                 + 0.93552 w_k1_FDX_s103
                                 + 0.972566 w_k1_IBM_s103
                                 + 0.970867 w_k1_INTC_s103
                                 + 0.834229 w_k1_KO_s103 + 1.0348 w_k1_MAT_s103
                                 + 0.94182 w_k1_MSFT_s103
                                 + 0.94955 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.54717 w_k1_ADM_s103
                                 + 1.79657 w_k1_AXP_s103
                                 + 1.96144 w_k1_DIS_s103 + 1.3932 w_k1_FDX_s103
                                 + 1.37057 w_k1_IBM_s103
                                 + 1.34116 w_k1_INTC_s103
                                 + 1.21231 w_k1_KO_s103 + 1.34037 w_k1_MAT_s103
                                 + 1.29543 w_k1_MSFT_s103 + 1.315 w_k1_PG_s103
                                 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.84586 w_k1_ADM_s103
                                 + 0.652561 w_k1_AXP_s103
                                 + 0.529429 w_k1_DIS_s103
                                 + 0.679105 w_k1_FDX_s103
                                 + 0.664584 w_k1_IBM_s103
                                 + 0.773828 w_k1_INTC_s103
                                 + 0.562671 w_k1_KO_s103
                                 + 0.51875 w_k1_MAT_s103
                                 + 0.730719 w_k1_MSFT_s103
                                 + 0.78725 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.888475 w_k1_ADM_s103
                                 + 0.936837 w_k1_AXP_s103
                                 + 0.912753 w_k1_DIS_s103
                                 + 1.0336 w_k1_FDX_s103
                                 + 0.664899 w_k1_IBM_s103
                                 + 1.1758 w_k1_INTC_s103
                                 + 0.742444 w_k1_KO_s103
                                 + 0.743907 w_k1_MAT_s103
                                 + 1.1088 w_k1_MSFT_s103 + 0.95167 w_k1_PG_s103
                                 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.2479 w_k1_ADM_s104
                                 + 1.12759 w_k1_AXP_s104
                                 + 1.02595 w_k1_DIS_s104
                                 + 1.14949 w_k1_FDX_s104
                                 + 0.884308 w_k1_IBM_s104
                                 + 0.960499 w_k1_INTC_s104
                                 + 1.01125 w_k1_KO_s104 + 1.2399 w_k1_MAT_s104
                                 + 1.01524 w_k1_MSFT_s104 + 1.0236 w_k1_PG_s104
                                 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.890546 w_k1_ADM_s104
                                 + 1.11578 w_k1_AXP_s104
                                 + 1.13694 w_k1_DIS_s104
                                 + 0.984472 w_k1_FDX_s104
                                 + 1.38165 w_k1_IBM_s104
                                 + 0.955674 w_k1_INTC_s104
                                 + 1.14703 w_k1_KO_s104 + 1.87885 w_k1_MAT_s104
                                 + 1.0477 w_k1_MSFT_s104 + 1.10881 w_k1_PG_s104
                                 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.96491 w_k1_ADM_s104
                                 + 0.940808 w_k1_AXP_s104
                                 + 0.878822 w_k1_DIS_s104
                                 + 1.04553 w_k1_FDX_s104
                                 + 1.01637 w_k1_IBM_s104
                                 + 0.820391 w_k1_INTC_s104
                                 + 1.0921 w_k1_KO_s104 + 1.04543 w_k1_MAT_s104
                                 + 0.941429 w_k1_MSFT_s104
                                 + 1.00275 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.0601 w_k1_ADM_s104
                                 + 1.3648 w_k1_AXP_s104 + 1.46546 w_k1_DIS_s104
                                 + 1.28941 w_k1_FDX_s104
                                 + 1.31165 w_k1_IBM_s104
                                 + 0.96801 w_k1_INTC_s104
                                 + 1.26804 w_k1_KO_s104 + 2.01011 w_k1_MAT_s104
                                 + 1.20758 w_k1_MSFT_s104
                                 + 1.06064 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.875384 w_k1_ADM_s104
                                 + 1.01432 w_k1_AXP_s104
                                 + 0.83521 w_k1_DIS_s104
                                 + 0.865568 w_k1_FDX_s104
                                 + 1.14068 w_k1_IBM_s104
                                 + 0.755372 w_k1_INTC_s104
                                 + 0.885027 w_k1_KO_s104
                                 + 1.99577 w_k1_MAT_s104
                                 + 0.843193 w_k1_MSFT_s104
                                 + 0.907324 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.863306 w_k1_ADM_s104
                                 + 0.831816 w_k1_AXP_s104
                                 + 0.712761 w_k1_DIS_s104
                                 + 0.844061 w_k1_FDX_s104
                                 + 0.838858 w_k1_IBM_s104
                                 + 0.746818 w_k1_INTC_s104
                                 + 0.874858 w_k1_KO_s104 + 1.2479 w_k1_MAT_s104
                                 + 0.821838 w_k1_MSFT_s104
                                 + 0.842686 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.924052 w_k1_ADM_s104
                                 + 0.891503 w_k1_AXP_s104
                                 + 1.03486 w_k1_DIS_s104
                                 + 1.03884 w_k1_FDX_s104
                                 + 0.778557 w_k1_IBM_s104
                                 + 0.87944 w_k1_INTC_s104
                                 + 0.948988 w_k1_KO_s104 + 1.1041 w_k1_MAT_s104
                                 + 0.961221 w_k1_MSFT_s104
                                 + 0.952639 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.800775 w_k1_ADM_s104
                                 + 0.712233 w_k1_AXP_s104
                                 + 0.591037 w_k1_DIS_s104
                                 + 0.831073 w_k1_FDX_s104
                                 + 0.619926 w_k1_IBM_s104
                                 + 0.830098 w_k1_INTC_s104
                                 + 0.677412 w_k1_KO_s104
                                 + 0.624045 w_k1_MAT_s104
                                 + 0.758746 w_k1_MSFT_s104
                                 + 0.764066 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.670287 w_k1_ADM_s105
                                 + 0.739902 w_k1_AXP_s105
                                 + 0.746826 w_k1_DIS_s105
                                 + 0.775524 w_k1_FDX_s105
                                 + 0.960362 w_k1_IBM_s105
                                 + 0.856499 w_k1_INTC_s105
                                 + 0.924362 w_k1_KO_s105
                                 + 0.644973 w_k1_MAT_s105
                                 + 0.882391 w_k1_MSFT_s105
                                 + 0.893336 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.789972 w_k1_ADM_s105
                                 + 0.642919 w_k1_AXP_s105
                                 + 0.556025 w_k1_DIS_s105
                                 + 0.701472 w_k1_FDX_s105
                                 + 0.636587 w_k1_IBM_s105
                                 + 0.798633 w_k1_INTC_s105
                                 + 0.677823 w_k1_KO_s105
                                 + 0.607268 w_k1_MAT_s105
                                 + 0.724668 w_k1_MSFT_s105
                                 + 0.843721 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.00447 w_k1_ADM_s105
                                 + 0.929009 w_k1_AXP_s105
                                 + 0.690701 w_k1_DIS_s105
                                 + 0.892312 w_k1_FDX_s105
                                 + 0.684679 w_k1_IBM_s105
                                 + 1.00276 w_k1_INTC_s105
                                 + 0.804482 w_k1_KO_s105
                                 + 0.761811 w_k1_MAT_s105
                                 + 0.818702 w_k1_MSFT_s105
                                 + 0.967088 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.0114 w_k1_ADM_s105
                                 + 1.10125 w_k1_AXP_s105
                                 + 0.969606 w_k1_DIS_s105
                                 + 0.947028 w_k1_FDX_s105
                                 + 1.33998 w_k1_IBM_s105
                                 + 0.935821 w_k1_INTC_s105
                                 + 1.24339 w_k1_KO_s105 + 1.35037 w_k1_MAT_s105
                                 + 0.933101 w_k1_MSFT_s105
                                 + 1.19485 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.06179 w_k1_ADM_s105
                                 + 1.17945 w_k1_AXP_s105
                                 + 1.43031 w_k1_DIS_s105
                                 + 1.12909 w_k1_FDX_s105
                                 + 1.17584 w_k1_IBM_s105
                                 + 1.0514 w_k1_INTC_s105 + 1.22774 w_k1_KO_s105
                                 + 0.845034 w_k1_MAT_s105
                                 + 1.16748 w_k1_MSFT_s105
                                 + 1.14432 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.27493 w_k1_ADM_s105
                                 + 1.59897 w_k1_AXP_s105
                                 + 1.38541 w_k1_DIS_s105
                                 + 1.31648 w_k1_FDX_s105
                                 + 1.22993 w_k1_IBM_s105
                                 + 1.32953 w_k1_INTC_s105
                                 + 1.19118 w_k1_KO_s105 + 1.38764 w_k1_MAT_s105
                                 + 1.31957 w_k1_MSFT_s105
                                 + 1.19576 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.874954 w_k1_ADM_s105
                                 + 0.578968 w_k1_AXP_s105
                                 + 0.707322 w_k1_DIS_s105
                                 + 0.866679 w_k1_FDX_s105
                                 + 0.585761 w_k1_IBM_s105
                                 + 0.846368 w_k1_INTC_s105
                                 + 0.731307 w_k1_KO_s105
                                 + 0.482417 w_k1_MAT_s105
                                 + 0.897609 w_k1_MSFT_s105
                                 + 0.887876 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.925045 w_k1_ADM_s105
                                 + 1.12668 w_k1_AXP_s105
                                 + 0.987018 w_k1_DIS_s105
                                 + 1.0528 w_k1_FDX_s105 + 0.95568 w_k1_IBM_s105
                                 + 1.24544 w_k1_INTC_s105
                                 + 0.981796 w_k1_KO_s105
                                 + 0.843242 w_k1_MAT_s105
                                 + 1.07461 w_k1_MSFT_s105
                                 + 0.998194 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.0272 w_k1_ADM_s106
                                 + 0.907738 w_k1_AXP_s106
                                 + 0.965822 w_k1_DIS_s106
                                 + 0.986181 w_k1_FDX_s106
                                 + 0.79503 w_k1_IBM_s106
                                 + 0.779622 w_k1_INTC_s106
                                 + 0.820168 w_k1_KO_s106
                                 + 0.773138 w_k1_MAT_s106
                                 + 1.00285 w_k1_MSFT_s106
                                 + 0.989483 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.0904 w_k1_ADM_s106
                                 + 1.14215 w_k1_AXP_s106
                                 + 1.18184 w_k1_DIS_s106
                                 + 1.04324 w_k1_FDX_s106
                                 + 1.20315 w_k1_IBM_s106
                                 + 1.17704 w_k1_INTC_s106
                                 + 1.16623 w_k1_KO_s106 + 1.24409 w_k1_MAT_s106
                                 + 1.19433 w_k1_MSFT_s106
                                 + 1.05701 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.925152 w_k1_ADM_s106
                                 + 0.819504 w_k1_AXP_s106
                                 + 0.792734 w_k1_DIS_s106
                                 + 0.786732 w_k1_FDX_s106
                                 + 1.03429 w_k1_IBM_s106
                                 + 0.849285 w_k1_INTC_s106
                                 + 0.878698 w_k1_KO_s106 + 1.0642 w_k1_MAT_s106
                                 + 0.906501 w_k1_MSFT_s106
                                 + 0.901643 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.44188 w_k1_ADM_s106
                                 + 1.76189 w_k1_AXP_s106
                                 + 1.76178 w_k1_DIS_s106
                                 + 1.39791 w_k1_FDX_s106
                                 + 1.32769 w_k1_IBM_s106
                                 + 1.3279 w_k1_INTC_s106 + 1.35682 w_k1_KO_s106
                                 + 1.58306 w_k1_MAT_s106
                                 + 1.55577 w_k1_MSFT_s106
                                 + 1.18537 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.926131 w_k1_ADM_s106
                                 + 0.857753 w_k1_AXP_s106
                                 + 0.628834 w_k1_DIS_s106
                                 + 0.819552 w_k1_FDX_s106
                                 + 0.727141 w_k1_IBM_s106
                                 + 0.821468 w_k1_INTC_s106
                                 + 0.662012 w_k1_KO_s106
                                 + 0.562198 w_k1_MAT_s106
                                 + 0.831194 w_k1_MSFT_s106
                                 + 0.881513 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.719381 w_k1_ADM_s106
                                 + 0.707132 w_k1_AXP_s106
                                 + 0.615912 w_k1_DIS_s106
                                 + 0.714988 w_k1_FDX_s106
                                 + 0.609063 w_k1_IBM_s106
                                 + 0.933644 w_k1_INTC_s106
                                 + 0.612986 w_k1_KO_s106
                                 + 0.484975 w_k1_MAT_s106
                                 + 0.843165 w_k1_MSFT_s106
                                 + 0.798259 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.19093 w_k1_ADM_s106
                                 + 1.19848 w_k1_AXP_s106
                                 + 1.22196 w_k1_DIS_s106
                                 + 1.17053 w_k1_FDX_s106
                                 + 0.906075 w_k1_IBM_s106
                                 + 1.10359 w_k1_INTC_s106
                                 + 1.01169 w_k1_KO_s106
                                 + 0.954378 w_k1_MAT_s106
                                 + 1.08816 w_k1_MSFT_s106
                                 + 0.99076 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.43037 w_k1_ADM_s106
                                 + 1.40719 w_k1_AXP_s106
                                 + 1.19529 w_k1_DIS_s106
                                 + 1.05475 w_k1_FDX_s106
                                 + 1.68672 w_k1_IBM_s106
                                 + 1.20821 w_k1_INTC_s106
                                 + 1.22367 w_k1_KO_s106 + 1.63583 w_k1_MAT_s106
                                 + 1.13266 w_k1_MSFT_s106
                                 + 1.19174 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.13383 w_k1_ADM_s107
                                 + 0.988647 w_k1_AXP_s107
                                 + 1.21654 w_k1_DIS_s107
                                 + 1.12029 w_k1_FDX_s107
                                 + 1.02708 w_k1_IBM_s107
                                 + 0.990588 w_k1_INTC_s107
                                 + 1.10217 w_k1_KO_s107 + 1.15479 w_k1_MAT_s107
                                 + 1.28018 w_k1_MSFT_s107
                                 + 1.13412 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.947938 w_k1_ADM_s107
                                 + 0.801545 w_k1_AXP_s107
                                 + 0.998812 w_k1_DIS_s107
                                 + 0.895004 w_k1_FDX_s107
                                 + 1.05561 w_k1_IBM_s107
                                 + 0.971216 w_k1_INTC_s107
                                 + 1.03809 w_k1_KO_s107
                                 + 0.887931 w_k1_MAT_s107
                                 + 1.04715 w_k1_MSFT_s107
                                 + 0.953953 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.08724 w_k1_ADM_s107
                                 + 0.934794 w_k1_AXP_s107
                                 + 0.690839 w_k1_DIS_s107
                                 + 0.774268 w_k1_FDX_s107
                                 + 1.01684 w_k1_IBM_s107
                                 + 0.717201 w_k1_INTC_s107
                                 + 0.846427 w_k1_KO_s107
                                 + 1.07561 w_k1_MAT_s107
                                 + 0.808219 w_k1_MSFT_s107
                                 + 0.942483 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.0741 w_k1_ADM_s107
                                 + 1.0479 w_k1_AXP_s107 + 1.33336 w_k1_DIS_s107
                                 + 1.25522 w_k1_FDX_s107
                                 + 0.778442 w_k1_IBM_s107
                                 + 1.08752 w_k1_INTC_s107
                                 + 0.984196 w_k1_KO_s107
                                 + 0.722652 w_k1_MAT_s107
                                 + 1.2733 w_k1_MSFT_s107
                                 + 0.963953 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.18054 w_k1_ADM_s107
                                 + 1.30019 w_k1_AXP_s107
                                 + 1.50109 w_k1_DIS_s107
                                 + 1.18313 w_k1_FDX_s107
                                 + 1.60887 w_k1_IBM_s107
                                 + 1.02797 w_k1_INTC_s107
                                 + 1.36638 w_k1_KO_s107 + 1.60924 w_k1_MAT_s107
                                 + 1.13156 w_k1_MSFT_s107
                                 + 1.27615 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.794495 w_k1_ADM_s107
                                 + 0.60918 w_k1_AXP_s107
                                 + 0.542418 w_k1_DIS_s107
                                 + 0.66728 w_k1_FDX_s107
                                 + 0.768291 w_k1_IBM_s107
                                 + 0.603663 w_k1_INTC_s107
                                 + 0.555931 w_k1_KO_s107
                                 + 0.607387 w_k1_MAT_s107
                                 + 0.657005 w_k1_MSFT_s107
                                 + 0.760702 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.08931 w_k1_ADM_s107
                                 + 1.10528 w_k1_AXP_s107
                                 + 0.891646 w_k1_DIS_s107
                                 + 0.973431 w_k1_FDX_s107
                                 + 1.03662 w_k1_IBM_s107
                                 + 0.756644 w_k1_INTC_s107
                                 + 0.962206 w_k1_KO_s107
                                 + 1.22774 w_k1_MAT_s107
                                 + 0.823442 w_k1_MSFT_s107
                                 + 0.977062 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.981887 w_k1_ADM_s107
                                 + 0.93862 w_k1_AXP_s107
                                 + 0.707579 w_k1_DIS_s107
                                 + 0.929266 w_k1_FDX_s107
                                 + 0.669889 w_k1_IBM_s107
                                 + 1.03363 w_k1_INTC_s107
                                 + 0.758343 w_k1_KO_s107
                                 + 0.525364 w_k1_MAT_s107
                                 + 0.886927 w_k1_MSFT_s107
                                 + 0.938329 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.25981 w_k1_ADM_s110
                                 + 1.41993 w_k1_AXP_s110
                                 + 1.64604 w_k1_DIS_s110
                                 + 1.31823 w_k1_FDX_s110
                                 + 1.39823 w_k1_IBM_s110
                                 + 1.06752 w_k1_INTC_s110
                                 + 1.26381 w_k1_KO_s110 + 1.39293 w_k1_MAT_s110
                                 + 1.10758 w_k1_MSFT_s110
                                 + 1.17788 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.33772 w_k1_ADM_s110
                                 + 1.20415 w_k1_AXP_s110
                                 + 1.10356 w_k1_DIS_s110
                                 + 0.981926 w_k1_FDX_s110
                                 + 1.6545 w_k1_IBM_s110
                                 + 0.909291 w_k1_INTC_s110
                                 + 1.39831 w_k1_KO_s110 + 1.79283 w_k1_MAT_s110
                                 + 0.89843 w_k1_MSFT_s110
                                 + 1.10568 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.885214 w_k1_ADM_s110
                                 + 1.11697 w_k1_AXP_s110
                                 + 1.18184 w_k1_DIS_s110
                                 + 0.991143 w_k1_FDX_s110
                                 + 1.2213 w_k1_IBM_s110 + 1.0277 w_k1_INTC_s110
                                 + 1.09527 w_k1_KO_s110 + 1.3129 w_k1_MAT_s110
                                 + 1.00928 w_k1_MSFT_s110
                                 + 1.08596 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.23034 w_k1_ADM_s110
                                 + 1.03712 w_k1_AXP_s110
                                 + 1.12331 w_k1_DIS_s110
                                 + 1.15996 w_k1_FDX_s110
                                 + 0.856095 w_k1_IBM_s110
                                 + 1.10225 w_k1_INTC_s110
                                 + 0.992925 w_k1_KO_s110
                                 + 0.689612 w_k1_MAT_s110
                                 + 1.04771 w_k1_MSFT_s110
                                 + 0.99011 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.74105 w_k1_ADM_s110
                                 + 0.647756 w_k1_AXP_s110
                                 + 0.705116 w_k1_DIS_s110
                                 + 0.735132 w_k1_FDX_s110
                                 + 0.820139 w_k1_IBM_s110
                                 + 0.790523 w_k1_INTC_s110
                                 + 0.865769 w_k1_KO_s110
                                 + 0.724076 w_k1_MAT_s110
                                 + 0.797061 w_k1_MSFT_s110
                                 + 0.888528 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.866014 w_k1_ADM_s110
                                 + 0.85211 w_k1_AXP_s110
                                 + 0.90819 w_k1_DIS_s110
                                 + 0.970169 w_k1_FDX_s110
                                 + 0.964685 w_k1_IBM_s110
                                 + 1.05046 w_k1_INTC_s110
                                 + 1.01093 w_k1_KO_s110
                                 + 0.779542 w_k1_MAT_s110
                                 + 1.05048 w_k1_MSFT_s110
                                 + 1.03313 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.978769 w_k1_ADM_s110
                                 + 0.874969 w_k1_AXP_s110
                                 + 0.749191 w_k1_DIS_s110
                                 + 0.752024 w_k1_FDX_s110
                                 + 1.2556 w_k1_IBM_s110
                                 + 0.861491 w_k1_INTC_s110
                                 + 1.0491 w_k1_KO_s110 + 1.28916 w_k1_MAT_s110
                                 + 0.836848 w_k1_MSFT_s110
                                 + 0.953523 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.23845 w_k1_ADM_s110
                                 + 1.30618 w_k1_AXP_s110
                                 + 1.32629 w_k1_DIS_s110
                                 + 1.29539 w_k1_FDX_s110
                                 + 0.911504 w_k1_IBM_s110
                                 + 1.07912 w_k1_INTC_s110 + 1.0883 w_k1_KO_s110
                                 + 1.14031 w_k1_MAT_s110
                                 + 1.12643 w_k1_MSFT_s110
                                 + 1.07297 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.897036 w_k1_ADM_s111
                                 + 0.708573 w_k1_AXP_s111
                                 + 0.684943 w_k1_DIS_s111
                                 + 0.648963 w_k1_FDX_s111
                                 + 0.948732 w_k1_IBM_s111
                                 + 0.76628 w_k1_INTC_s111
                                 + 0.835164 w_k1_KO_s111
                                 + 0.936494 w_k1_MAT_s111
                                 + 0.757197 w_k1_MSFT_s111
                                 + 0.924749 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.879927 w_k1_ADM_s111
                                 + 1.00151 w_k1_AXP_s111
                                 + 0.894777 w_k1_DIS_s111
                                 + 0.868259 w_k1_FDX_s111
                                 + 1.28222 w_k1_IBM_s111
                                 + 0.802486 w_k1_INTC_s111
                                 + 1.01103 w_k1_KO_s111 + 1.35855 w_k1_MAT_s111
                                 + 0.955109 w_k1_MSFT_s111
                                 + 1.02588 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.6595 w_k1_ADM_s111
                                 + 1.32537 w_k1_AXP_s111
                                 + 1.41022 w_k1_DIS_s111
                                 + 1.18388 w_k1_FDX_s111
                                 + 1.21822 w_k1_IBM_s111
                                 + 0.981844 w_k1_INTC_s111
                                 + 1.24726 w_k1_KO_s111 + 1.1157 w_k1_MAT_s111
                                 + 1.0242 w_k1_MSFT_s111 + 1.25903 w_k1_PG_s111
                                 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.11336 w_k1_ADM_s111
                                 + 1.18991 w_k1_AXP_s111
                                 + 1.18814 w_k1_DIS_s111
                                 + 1.18532 w_k1_FDX_s111
                                 + 0.882961 w_k1_IBM_s111
                                 + 1.0199 w_k1_INTC_s111
                                 + 0.991505 w_k1_KO_s111
                                 + 0.707274 w_k1_MAT_s111
                                 + 1.01555 w_k1_MSFT_s111
                                 + 1.04988 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.31336 w_k1_ADM_s111
                                 + 1.19577 w_k1_AXP_s111
                                 + 1.30475 w_k1_DIS_s111
                                 + 1.19519 w_k1_FDX_s111
                                 + 0.719665 w_k1_IBM_s111
                                 + 0.950009 w_k1_INTC_s111
                                 + 0.988275 w_k1_KO_s111
                                 + 0.877889 w_k1_MAT_s111
                                 + 1.01928 w_k1_MSFT_s111
                                 + 1.02755 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.28715 w_k1_ADM_s111
                                 + 1.17387 w_k1_AXP_s111
                                 + 1.36341 w_k1_DIS_s111
                                 + 1.12322 w_k1_FDX_s111
                                 + 1.53379 w_k1_IBM_s111
                                 + 0.806685 w_k1_INTC_s111
                                 + 1.20615 w_k1_KO_s111 + 1.57008 w_k1_MAT_s111
                                 + 1.07218 w_k1_MSFT_s111
                                 + 1.12734 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.03255 w_k1_ADM_s111
                                 + 1.03831 w_k1_AXP_s111
                                 + 1.04304 w_k1_DIS_s111
                                 + 0.941122 w_k1_FDX_s111
                                 + 0.826516 w_k1_IBM_s111
                                 + 0.795271 w_k1_INTC_s111
                                 + 0.918735 w_k1_KO_s111
                                 + 0.799036 w_k1_MAT_s111
                                 + 0.854453 w_k1_MSFT_s111
                                 + 0.940223 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.43374 w_k1_ADM_s111
                                 + 1.6519 w_k1_AXP_s111 + 1.91551 w_k1_DIS_s111
                                 + 1.51131 w_k1_FDX_s111
                                 + 1.04996 w_k1_IBM_s111
                                 + 1.16776 w_k1_INTC_s111
                                 + 1.43145 w_k1_KO_s111 + 1.91498 w_k1_MAT_s111
                                 + 1.38839 w_k1_MSFT_s111 + 1.3297 w_k1_PG_s111
                                 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.21068 w_k1_ADM_s112
                                 + 1.05507 w_k1_AXP_s112
                                 + 1.00418 w_k1_DIS_s112
                                 + 1.16674 w_k1_FDX_s112
                                 + 0.708307 w_k1_IBM_s112
                                 + 1.10005 w_k1_INTC_s112
                                 + 1.13901 w_k1_KO_s112
                                 + 0.694255 w_k1_MAT_s112
                                 + 1.01208 w_k1_MSFT_s112
                                 + 0.989252 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.25033 w_k1_ADM_s112
                                 + 1.34229 w_k1_AXP_s112 + 1.0494 w_k1_DIS_s112
                                 + 1.11988 w_k1_FDX_s112
                                 + 1.18152 w_k1_IBM_s112
                                 + 0.918429 w_k1_INTC_s112
                                 + 1.42983 w_k1_KO_s112 + 1.47976 w_k1_MAT_s112
                                 + 1.05304 w_k1_MSFT_s112
                                 + 1.06745 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.59351 w_k1_ADM_s112
                                 + 1.39124 w_k1_AXP_s112
                                 + 1.63927 w_k1_DIS_s112 + 1.3035 w_k1_FDX_s112
                                 + 1.24998 w_k1_IBM_s112
                                 + 1.10211 w_k1_INTC_s112
                                 + 1.56501 w_k1_KO_s112 + 1.45884 w_k1_MAT_s112
                                 + 1.36162 w_k1_MSFT_s112
                                 + 1.33254 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.950276 w_k1_ADM_s112
                                 + 0.901286 w_k1_AXP_s112
                                 + 1.05852 w_k1_DIS_s112
                                 + 0.955033 w_k1_FDX_s112
                                 + 0.929786 w_k1_IBM_s112
                                 + 0.98505 w_k1_INTC_s112
                                 + 1.30334 w_k1_KO_s112 + 1.00527 w_k1_MAT_s112
                                 + 1.17723 w_k1_MSFT_s112
                                 + 0.997888 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.865245 w_k1_ADM_s112
                                 + 0.841654 w_k1_AXP_s112
                                 + 0.7714 w_k1_DIS_s112
                                 + 0.828674 w_k1_FDX_s112
                                 + 1.24711 w_k1_IBM_s112
                                 + 0.908099 w_k1_INTC_s112
                                 + 1.21184 w_k1_KO_s112 + 1.16719 w_k1_MAT_s112
                                 + 0.889184 w_k1_MSFT_s112
                                 + 0.916344 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.808509 w_k1_ADM_s112
                                 + 0.787685 w_k1_AXP_s112
                                 + 0.764334 w_k1_DIS_s112
                                 + 0.796481 w_k1_FDX_s112
                                 + 0.879187 w_k1_IBM_s112
                                 + 0.660359 w_k1_INTC_s112
                                 + 0.888227 w_k1_KO_s112
                                 + 0.711273 w_k1_MAT_s112
                                 + 0.81478 w_k1_MSFT_s112
                                 + 0.822876 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.977963 w_k1_ADM_s112
                                 + 0.952024 w_k1_AXP_s112
                                 + 0.939683 w_k1_DIS_s112
                                 + 0.978131 w_k1_FDX_s112
                                 + 1.31608 w_k1_IBM_s112
                                 + 1.0169 w_k1_INTC_s112 + 1.34221 w_k1_KO_s112
                                 + 0.903519 w_k1_MAT_s112
                                 + 1.03454 w_k1_MSFT_s112
                                 + 1.05805 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.919478 w_k1_ADM_s112
                                 + 0.59109 w_k1_AXP_s112
                                 + 0.60892 w_k1_DIS_s112
                                 + 0.761119 w_k1_FDX_s112
                                 + 0.553174 w_k1_IBM_s112
                                 + 0.670558 w_k1_INTC_s112
                                 + 0.738666 w_k1_KO_s112
                                 + 0.679943 w_k1_MAT_s112
                                 + 0.881121 w_k1_MSFT_s112
                                 + 0.789054 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.12631 w_k1_ADM_s113
                                 + 0.942233 w_k1_AXP_s113
                                 + 1.09922 w_k1_DIS_s113
                                 + 1.18417 w_k1_FDX_s113
                                 + 0.578497 w_k1_IBM_s113
                                 + 0.936844 w_k1_INTC_s113
                                 + 0.827076 w_k1_KO_s113
                                 + 0.533315 w_k1_MAT_s113
                                 + 1.05706 w_k1_MSFT_s113
                                 + 0.924385 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.26078 w_k1_ADM_s113
                                 + 1.25169 w_k1_AXP_s113
                                 + 1.31309 w_k1_DIS_s113
                                 + 1.35219 w_k1_FDX_s113
                                 + 0.958117 w_k1_IBM_s113
                                 + 1.18297 w_k1_INTC_s113
                                 + 1.05369 w_k1_KO_s113
                                 + 0.581249 w_k1_MAT_s113
                                 + 1.22192 w_k1_MSFT_s113
                                 + 0.952972 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.31569 w_k1_ADM_s113
                                 + 0.8718 w_k1_AXP_s113 + 1.42765 w_k1_DIS_s113
                                 + 1.19294 w_k1_FDX_s113
                                 + 0.83472 w_k1_IBM_s113
                                 + 1.04582 w_k1_INTC_s113
                                 + 1.00192 w_k1_KO_s113
                                 + 0.703458 w_k1_MAT_s113
                                 + 1.2305 w_k1_MSFT_s113 + 1.16971 w_k1_PG_s113
                                 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.972374 w_k1_ADM_s113
                                 + 0.963307 w_k1_AXP_s113
                                 + 0.734423 w_k1_DIS_s113
                                 + 0.842495 w_k1_FDX_s113
                                 + 1.02682 w_k1_IBM_s113
                                 + 0.878998 w_k1_INTC_s113
                                 + 0.905074 w_k1_KO_s113
                                 + 0.83333 w_k1_MAT_s113
                                 + 0.889541 w_k1_MSFT_s113
                                 + 0.999252 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.00189 w_k1_ADM_s113
                                 + 0.713818 w_k1_AXP_s113
                                 + 0.767427 w_k1_DIS_s113
                                 + 0.803873 w_k1_FDX_s113
                                 + 0.843706 w_k1_IBM_s113
                                 + 0.722981 w_k1_INTC_s113
                                 + 0.929134 w_k1_KO_s113
                                 + 0.430104 w_k1_MAT_s113
                                 + 0.69741 w_k1_MSFT_s113
                                 + 0.908492 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.37496 w_k1_ADM_s113
                                 + 1.45692 w_k1_AXP_s113
                                 + 1.29525 w_k1_DIS_s113
                                 + 1.23475 w_k1_FDX_s113
                                 + 1.19637 w_k1_IBM_s113
                                 + 1.13494 w_k1_INTC_s113
                                 + 1.27361 w_k1_KO_s113
                                 + 0.942009 w_k1_MAT_s113
                                 + 1.14644 w_k1_MSFT_s113
                                 + 1.15713 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.5348 w_k1_ADM_s113
                                 + 1.16767 w_k1_AXP_s113
                                 + 1.52876 w_k1_DIS_s113
                                 + 1.11089 w_k1_FDX_s113
                                 + 1.43745 w_k1_IBM_s113
                                 + 1.02287 w_k1_INTC_s113
                                 + 1.56554 w_k1_KO_s113 + 1.10519 w_k1_MAT_s113
                                 + 1.11269 w_k1_MSFT_s113 + 1.1937 w_k1_PG_s113
                                 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.999087 w_k1_ADM_s113
                                 + 0.823737 w_k1_AXP_s113
                                 + 1.08071 w_k1_DIS_s113
                                 + 0.999486 w_k1_FDX_s113
                                 + 1.08158 w_k1_IBM_s113
                                 + 1.29487 w_k1_INTC_s113
                                 + 1.04146 w_k1_KO_s113
                                 + 0.570208 w_k1_MAT_s113
                                 + 1.10808 w_k1_MSFT_s113
                                 + 0.943336 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.08057 w_k1_ADM_s114
                                 + 1.39938 w_k1_AXP_s114
                                 + 1.16277 w_k1_DIS_s114
                                 + 1.09149 w_k1_FDX_s114
                                 + 1.57654 w_k1_IBM_s114
                                 + 1.27002 w_k1_INTC_s114
                                 + 1.43992 w_k1_KO_s114 + 1.77828 w_k1_MAT_s114
                                 + 1.13919 w_k1_MSFT_s114
                                 + 1.13209 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.752291 w_k1_ADM_s114
                                 + 0.728891 w_k1_AXP_s114
                                 + 0.724167 w_k1_DIS_s114
                                 + 0.803495 w_k1_FDX_s114
                                 + 0.775575 w_k1_IBM_s114
                                 + 0.934729 w_k1_INTC_s114
                                 + 0.779163 w_k1_KO_s114
                                 + 0.94487 w_k1_MAT_s114
                                 + 0.829039 w_k1_MSFT_s114
                                 + 0.783603 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.952813 w_k1_ADM_s114
                                 + 0.872595 w_k1_AXP_s114
                                 + 1.09845 w_k1_DIS_s114
                                 + 0.950607 w_k1_FDX_s114
                                 + 1.23463 w_k1_IBM_s114
                                 + 1.0183 w_k1_INTC_s114 + 1.28611 w_k1_KO_s114
                                 + 1.49601 w_k1_MAT_s114
                                 + 0.996895 w_k1_MSFT_s114
                                 + 1.0575 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.05089 w_k1_ADM_s114
                                 + 0.864377 w_k1_AXP_s114
                                 + 0.752801 w_k1_DIS_s114
                                 + 0.834037 w_k1_FDX_s114
                                 + 1.28377 w_k1_IBM_s114
                                 + 0.843865 w_k1_INTC_s114
                                 + 1.05529 w_k1_KO_s114 + 1.08042 w_k1_MAT_s114
                                 + 0.82604 w_k1_MSFT_s114
                                 + 0.979605 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.02347 w_k1_ADM_s114
                                 + 1.24558 w_k1_AXP_s114
                                 + 0.952048 w_k1_DIS_s114
                                 + 1.03515 w_k1_FDX_s114
                                 + 0.885116 w_k1_IBM_s114
                                 + 1.25513 w_k1_INTC_s114
                                 + 0.981353 w_k1_KO_s114 + 1.4573 w_k1_MAT_s114
                                 + 0.968426 w_k1_MSFT_s114
                                 + 0.945239 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.14087 w_k1_ADM_s114
                                 + 1.55208 w_k1_AXP_s114
                                 + 1.41684 w_k1_DIS_s114
                                 + 1.34589 w_k1_FDX_s114
                                 + 1.10386 w_k1_IBM_s114
                                 + 1.24083 w_k1_INTC_s114
                                 + 1.15604 w_k1_KO_s114 + 1.15257 w_k1_MAT_s114
                                 + 1.18117 w_k1_MSFT_s114
                                 + 1.13914 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.975635 w_k1_ADM_s114
                                 + 0.822528 w_k1_AXP_s114
                                 + 1.15803 w_k1_DIS_s114
                                 + 1.10006 w_k1_FDX_s114
                                 + 0.894632 w_k1_IBM_s114
                                 + 1.16514 w_k1_INTC_s114
                                 + 0.993311 w_k1_KO_s114
                                 + 0.556718 w_k1_MAT_s114
                                 + 1.06406 w_k1_MSFT_s114
                                 + 1.07298 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.855429 w_k1_ADM_s114
                                 + 0.78511 w_k1_AXP_s114
                                 + 0.772825 w_k1_DIS_s114
                                 + 0.864327 w_k1_FDX_s114
                                 + 1.01139 w_k1_IBM_s114
                                 + 0.853114 w_k1_INTC_s114
                                 + 0.951461 w_k1_KO_s114
                                 + 0.944003 w_k1_MAT_s114
                                 + 0.928801 w_k1_MSFT_s114
                                 + 0.932598 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.900867 w_k1_ADM_s115
                                 + 0.651475 w_k1_AXP_s115
                                 + 0.780009 w_k1_DIS_s115
                                 + 0.774686 w_k1_FDX_s115
                                 + 0.840294 w_k1_IBM_s115
                                 + 0.755423 w_k1_INTC_s115
                                 + 0.793341 w_k1_KO_s115
                                 + 0.696136 w_k1_MAT_s115
                                 + 0.922651 w_k1_MSFT_s115
                                 + 0.96717 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.876974 w_k1_ADM_s115
                                 + 0.927017 w_k1_AXP_s115
                                 + 0.531413 w_k1_DIS_s115
                                 + 0.83387 w_k1_FDX_s115
                                 + 0.837988 w_k1_IBM_s115
                                 + 0.77666 w_k1_INTC_s115
                                 + 0.707129 w_k1_KO_s115
                                 + 1.01758 w_k1_MAT_s115
                                 + 0.815208 w_k1_MSFT_s115
                                 + 0.864532 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.10708 w_k1_ADM_s115
                                 + 1.68454 w_k1_AXP_s115
                                 + 1.23937 w_k1_DIS_s115
                                 + 1.32427 w_k1_FDX_s115
                                 + 1.29754 w_k1_IBM_s115
                                 + 1.20457 w_k1_INTC_s115
                                 + 1.28871 w_k1_KO_s115 + 1.785 w_k1_MAT_s115
                                 + 1.29575 w_k1_MSFT_s115
                                 + 1.16385 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.989457 w_k1_ADM_s115
                                 + 1.13178 w_k1_AXP_s115
                                 + 1.09227 w_k1_DIS_s115
                                 + 1.09755 w_k1_FDX_s115
                                 + 1.59101 w_k1_IBM_s115
                                 + 1.15767 w_k1_INTC_s115
                                 + 1.15995 w_k1_KO_s115 + 1.49619 w_k1_MAT_s115
                                 + 1.20747 w_k1_MSFT_s115
                                 + 1.14575 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.736117 w_k1_ADM_s115
                                 + 0.783616 w_k1_AXP_s115
                                 + 0.828421 w_k1_DIS_s115
                                 + 0.806545 w_k1_FDX_s115
                                 + 1.23321 w_k1_IBM_s115
                                 + 0.848465 w_k1_INTC_s115
                                 + 1.05247 w_k1_KO_s115 + 1.09118 w_k1_MAT_s115
                                 + 0.931675 w_k1_MSFT_s115
                                 + 0.918634 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.934337 w_k1_ADM_s115
                                 + 1.10515 w_k1_AXP_s115
                                 + 1.02049 w_k1_DIS_s115
                                 + 0.976238 w_k1_FDX_s115
                                 + 0.991692 w_k1_IBM_s115
                                 + 1.03825 w_k1_INTC_s115
                                 + 0.892788 w_k1_KO_s115
                                 + 1.10847 w_k1_MAT_s115
                                 + 1.07299 w_k1_MSFT_s115
                                 + 0.916467 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.04681 w_k1_ADM_s115
                                 + 0.927108 w_k1_AXP_s115
                                 + 1.07248 w_k1_DIS_s115 + 1.1413 w_k1_FDX_s115
                                 + 0.810696 w_k1_IBM_s115
                                 + 1.14594 w_k1_INTC_s115
                                 + 0.897571 w_k1_KO_s115
                                 + 0.742239 w_k1_MAT_s115
                                 + 1.08043 w_k1_MSFT_s115
                                 + 1.06235 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.797396 w_k1_ADM_s115
                                 + 0.693456 w_k1_AXP_s115
                                 + 0.723356 w_k1_DIS_s115
                                 + 0.829541 w_k1_FDX_s115
                                 + 0.581413 w_k1_IBM_s115
                                 + 0.972193 w_k1_INTC_s115
                                 + 0.651966 w_k1_KO_s115
                                 + 0.477372 w_k1_MAT_s115
                                 + 0.849123 w_k1_MSFT_s115
                                 + 0.905948 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.01181 w_k1_ADM_s116
                                 + 1.09055 w_k1_AXP_s116
                                 + 1.01264 w_k1_DIS_s116
                                 + 1.02132 w_k1_FDX_s116
                                 + 1.15087 w_k1_IBM_s116
                                 + 1.08362 w_k1_INTC_s116
                                 + 1.17087 w_k1_KO_s116 + 1.4472 w_k1_MAT_s116
                                 + 1.1201 w_k1_MSFT_s116
                                 + 0.952821 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.19071 w_k1_ADM_s116
                                 + 1.09317 w_k1_AXP_s116
                                 + 1.00107 w_k1_DIS_s116
                                 + 1.08761 w_k1_FDX_s116
                                 + 0.709816 w_k1_IBM_s116
                                 + 1.05145 w_k1_INTC_s116
                                 + 0.886424 w_k1_KO_s116
                                 + 0.79792 w_k1_MAT_s116
                                 + 0.964449 w_k1_MSFT_s116
                                 + 0.995349 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.803805 w_k1_ADM_s116
                                 + 0.730409 w_k1_AXP_s116
                                 + 0.533049 w_k1_DIS_s116
                                 + 0.681294 w_k1_FDX_s116
                                 + 0.620095 w_k1_IBM_s116
                                 + 0.81563 w_k1_INTC_s116
                                 + 0.64909 w_k1_KO_s116 + 0.64431 w_k1_MAT_s116
                                 + 0.788494 w_k1_MSFT_s116
                                 + 0.75899 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.00115 w_k1_ADM_s116
                                 + 0.690246 w_k1_AXP_s116
                                 + 0.777791 w_k1_DIS_s116
                                 + 0.820092 w_k1_FDX_s116
                                 + 0.832907 w_k1_IBM_s116
                                 + 0.793106 w_k1_INTC_s116
                                 + 0.821309 w_k1_KO_s116
                                 + 0.686113 w_k1_MAT_s116
                                 + 0.872979 w_k1_MSFT_s116
                                 + 0.849615 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.07893 w_k1_ADM_s116
                                 + 1.34147 w_k1_AXP_s116
                                 + 1.53782 w_k1_DIS_s116
                                 + 1.26971 w_k1_FDX_s116
                                 + 0.985146 w_k1_IBM_s116
                                 + 1.30904 w_k1_INTC_s116
                                 + 1.16028 w_k1_KO_s116 + 1.08962 w_k1_MAT_s116
                                 + 1.28782 w_k1_MSFT_s116
                                 + 1.02518 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.26996 w_k1_ADM_s116
                                 + 1.10716 w_k1_AXP_s116
                                 + 0.853475 w_k1_DIS_s116
                                 + 1.00469 w_k1_FDX_s116
                                 + 0.910939 w_k1_IBM_s116
                                 + 0.903763 w_k1_INTC_s116
                                 + 0.950291 w_k1_KO_s116
                                 + 1.15162 w_k1_MAT_s116
                                 + 0.8832 w_k1_MSFT_s116
                                 + 0.921022 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.26098 w_k1_ADM_s116
                                 + 1.27651 w_k1_AXP_s116 + 1.2502 w_k1_DIS_s116
                                 + 1.03074 w_k1_FDX_s116
                                 + 1.58873 w_k1_IBM_s116
                                 + 1.11347 w_k1_INTC_s116
                                 + 1.47651 w_k1_KO_s116 + 1.48638 w_k1_MAT_s116
                                 + 0.993406 w_k1_MSFT_s116
                                 + 1.11271 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.779775 w_k1_ADM_s116
                                 + 0.892771 w_k1_AXP_s116
                                 + 0.757908 w_k1_DIS_s116
                                 + 0.756715 w_k1_FDX_s116
                                 + 1.09794 w_k1_IBM_s116
                                 + 0.826123 w_k1_INTC_s116
                                 + 0.98483 w_k1_KO_s116 + 1.29573 w_k1_MAT_s116
                                 + 0.797018 w_k1_MSFT_s116
                                 + 0.870229 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.19639 w_k1_ADM_s117
                                 + 1.15322 w_k1_AXP_s117
                                 + 1.11589 w_k1_DIS_s117
                                 + 1.03653 w_k1_FDX_s117
                                 + 0.935704 w_k1_IBM_s117
                                 + 0.961584 w_k1_INTC_s117
                                 + 1.05419 w_k1_KO_s117 + 1.22867 w_k1_MAT_s117
                                 + 1.09407 w_k1_MSFT_s117
                                 + 0.966369 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.24477 w_k1_ADM_s117
                                 + 1.34971 w_k1_AXP_s117
                                 + 0.993359 w_k1_DIS_s117
                                 + 1.02347 w_k1_FDX_s117
                                 + 1.24347 w_k1_IBM_s117
                                 + 1.08288 w_k1_INTC_s117
                                 + 1.10248 w_k1_KO_s117 + 1.65359 w_k1_MAT_s117
                                 + 1.14028 w_k1_MSFT_s117
                                 + 0.963397 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.08889 w_k1_ADM_s117
                                 + 0.958638 w_k1_AXP_s117
                                 + 0.743193 w_k1_DIS_s117
                                 + 0.787088 w_k1_FDX_s117
                                 + 0.934228 w_k1_IBM_s117
                                 + 0.935577 w_k1_INTC_s117
                                 + 0.82848 w_k1_KO_s117 + 1.21043 w_k1_MAT_s117
                                 + 0.887171 w_k1_MSFT_s117
                                 + 0.892179 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.07983 w_k1_ADM_s117
                                 + 1.00297 w_k1_AXP_s117
                                 + 0.977922 w_k1_DIS_s117
                                 + 0.859208 w_k1_FDX_s117
                                 + 1.44036 w_k1_IBM_s117
                                 + 0.883694 w_k1_INTC_s117
                                 + 1.1235 w_k1_KO_s117 + 1.47291 w_k1_MAT_s117
                                 + 0.950467 w_k1_MSFT_s117
                                 + 0.913931 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.993619 w_k1_ADM_s117
                                 + 1.0916 w_k1_AXP_s117 + 1.45042 w_k1_DIS_s117
                                 + 1.10599 w_k1_FDX_s117
                                 + 1.52437 w_k1_IBM_s117
                                 + 1.20956 w_k1_INTC_s117
                                 + 1.37801 w_k1_KO_s117 + 1.67327 w_k1_MAT_s117
                                 + 1.24205 w_k1_MSFT_s117
                                 + 0.985164 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.64913 w_k1_ADM_s117
                                 + 1.41395 w_k1_AXP_s117
                                 + 1.77559 w_k1_DIS_s117
                                 + 1.26072 w_k1_FDX_s117
                                 + 1.32331 w_k1_IBM_s117
                                 + 1.22249 w_k1_INTC_s117
                                 + 1.47463 w_k1_KO_s117 + 1.40664 w_k1_MAT_s117
                                 + 1.38958 w_k1_MSFT_s117
                                 + 1.18664 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.13284 w_k1_ADM_s117
                                 + 0.810501 w_k1_AXP_s117
                                 + 0.84691 w_k1_DIS_s117
                                 + 0.861002 w_k1_FDX_s117
                                 + 0.690466 w_k1_IBM_s117
                                 + 0.950629 w_k1_INTC_s117
                                 + 0.874814 w_k1_KO_s117
                                 + 0.532589 w_k1_MAT_s117
                                 + 0.835083 w_k1_MSFT_s117
                                 + 0.90346 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.99485 w_k1_ADM_s117
                                 + 1.16909 w_k1_AXP_s117
                                 + 1.43771 w_k1_DIS_s117
                                 + 1.38125 w_k1_FDX_s117
                                 + 0.810585 w_k1_IBM_s117
                                 + 1.36666 w_k1_INTC_s117
                                 + 1.11865 w_k1_KO_s117
                                 + 0.741396 w_k1_MAT_s117
                                 + 1.37116 w_k1_MSFT_s117
                                 + 1.02074 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0152805369864983 w_k1_ADM_s000
                                 + 0.0187641524011291 w_k1_ADM_s001
                                 + 0.0169328539493372 w_k1_ADM_s002
                                 + 0.0315310523997833 w_k1_ADM_s003
                                 + 0.0364342055596403 w_k1_ADM_s004
                                 + 0.0460108025657182 w_k1_ADM_s005
                                 + 0.0191472566336564 w_k1_ADM_s006
                                 + 0.0241202835532445 w_k1_ADM_s007
                                 + 0.0237937258473767 w_k1_ADM_s010
                                 + 0.0136797742086033 w_k1_ADM_s011
                                 + 0.0134895467308456 w_k1_ADM_s012
                                 + 0.0310250438277648 w_k1_ADM_s013
                                 + 0.0227090941523578 w_k1_ADM_s014
                                 + 0.013365268845096 w_k1_ADM_s015
                                 + 0.0272529702485467 w_k1_ADM_s016
                                 + 0.0135624516030114 w_k1_ADM_s017
                                 + 0.0323024571481376 w_k1_ADM_s100
                                 + 0.053056305040186 w_k1_ADM_s101
                                 + 0.0372707312010498 w_k1_ADM_s102
                                 + 0.0210183772655393 w_k1_ADM_s103
                                 + 0.0397451873755829 w_k1_ADM_s104
                                 + 0.0480814546392851 w_k1_ADM_s105
                                 + 0.0918509378142705 w_k1_ADM_s106
                                 + 0.0426516496349809 w_k1_ADM_s107
                                 + 0.0268434323994661 w_k1_ADM_s110
                                 + 0.0198855126254299 w_k1_ADM_s111
                                 + 0.0392787747877454 w_k1_ADM_s112
                                 + 0.0388150830754999 w_k1_ADM_s113
                                 + 0.0279081551380408 w_k1_ADM_s114
                                 + 0.0387663609511237 w_k1_ADM_s115
                                 + 0.0655402626626617 w_k1_ADM_s116
                                 + 0.0220964089436689 w_k1_ADM_s117
                                 + 0.0169182388254833 w_k1_AXP_s000
                                 + 0.0218738958796595 w_k1_AXP_s001
                                 + 0.0146281451218473 w_k1_AXP_s002
                                 + 0.0339669111445702 w_k1_AXP_s003
                                 + 0.0390793093558031 w_k1_AXP_s004
                                 + 0.0435024867186773 w_k1_AXP_s005
                                 + 0.0190689779803947 w_k1_AXP_s006
                                 + 0.0236733858652534 w_k1_AXP_s007
                                 + 0.0314555596022095 w_k1_AXP_s010
                                 + 0.0140375029050183 w_k1_AXP_s011
                                 + 0.0118551831468638 w_k1_AXP_s012
                                 + 0.033463950734924 w_k1_AXP_s013
                                 + 0.0233182739443957 w_k1_AXP_s014
                                 + 0.0146204604035273 w_k1_AXP_s015
                                 + 0.0320931192701902 w_k1_AXP_s016
                                 + 0.0161376971726986 w_k1_AXP_s017
                                 + 0.0341554496645246 w_k1_AXP_s100
                                 + 0.0587602818252696 w_k1_AXP_s101
                                 + 0.0318545879084977 w_k1_AXP_s102
                                 + 0.0224463140501698 w_k1_AXP_s103
                                 + 0.0426558660613231 w_k1_AXP_s104
                                 + 0.0514231376469334 w_k1_AXP_s105
                                 + 0.0916975256642765 w_k1_AXP_s106
                                 + 0.0407939501278709 w_k1_AXP_s107
                                 + 0.026183251366482 w_k1_AXP_s110
                                 + 0.0192674252769233 w_k1_AXP_s111
                                 + 0.0363378329081386 w_k1_AXP_s112
                                 + 0.0339528552532732 w_k1_AXP_s113
                                 + 0.030020149731666 w_k1_AXP_s114
                                 + 0.040585764786986 w_k1_AXP_s115
                                 + 0.0650080355677799 w_k1_AXP_s116
                                 + 0.0194254288779455 w_k1_AXP_s117
                                 + 0.0171976709134917 w_k1_DIS_s000
                                 + 0.0199575422190995 w_k1_DIS_s001
                                 + 0.015471568558081 w_k1_DIS_s002
                                 + 0.0313103240261576 w_k1_DIS_s003
                                 + 0.0379334096907633 w_k1_DIS_s004
                                 + 0.0438914347623275 w_k1_DIS_s005
                                 + 0.0213692983938975 w_k1_DIS_s006
                                 + 0.0250003552421528 w_k1_DIS_s007
                                 + 0.0294625597172572 w_k1_DIS_s010
                                 + 0.0141273795551971 w_k1_DIS_s011
                                 + 0.0127167536447939 w_k1_DIS_s012
                                 + 0.035209290427103 w_k1_DIS_s013
                                 + 0.0222761851852483 w_k1_DIS_s014
                                 + 0.0145942328118736 w_k1_DIS_s015
                                 + 0.0329405284613298 w_k1_DIS_s016
                                 + 0.0165541751451307 w_k1_DIS_s017
                                 + 0.0301114080099869 w_k1_DIS_s100
                                 + 0.0575871765449071 w_k1_DIS_s101
                                 + 0.0388816064705808 w_k1_DIS_s102
                                 + 0.0223103071810897 w_k1_DIS_s103
                                 + 0.0413629226044107 w_k1_DIS_s104
                                 + 0.0495857591442626 w_k1_DIS_s105
                                 + 0.0876819967592813 w_k1_DIS_s106
                                 + 0.0417188212016218 w_k1_DIS_s107
                                 + 0.0274351907310698 w_k1_DIS_s110
                                 + 0.0199049987121344 w_k1_DIS_s111
                                 + 0.0371591251327249 w_k1_DIS_s112
                                 + 0.0366381038576019 w_k1_DIS_s113
                                 + 0.0295772604046012 w_k1_DIS_s114
                                 + 0.0396657209335123 w_k1_DIS_s115
                                 + 0.0616916316820352 w_k1_DIS_s116
                                 + 0.0194273007545333 w_k1_DIS_s117
                                 + 0.0159690915378861 w_k1_FDX_s000
                                 + 0.0201342312775766 w_k1_FDX_s001
                                 + 0.0144438271569427 w_k1_FDX_s002
                                 + 0.0307505605315743 w_k1_FDX_s003
                                 + 0.0401966135498308 w_k1_FDX_s004
                                 + 0.042158444384799 w_k1_FDX_s005
                                 + 0.01958466793614 w_k1_FDX_s006
                                 + 0.0242936766065559 w_k1_FDX_s007
                                 + 0.0298192111099499 w_k1_FDX_s010
                                 + 0.0141640764263351 w_k1_FDX_s011
                                 + 0.0121255006159211 w_k1_FDX_s012
                                 + 0.036133650122286 w_k1_FDX_s013
                                 + 0.0230896807558693 w_k1_FDX_s014
                                 + 0.0145798102799254 w_k1_FDX_s015
                                 + 0.032433392258169 w_k1_FDX_s016
                                 + 0.0158813368165546 w_k1_FDX_s017
                                 + 0.0309357974978678 w_k1_FDX_s100
                                 + 0.0540315846287287 w_k1_FDX_s101
                                 + 0.0360693710916546 w_k1_FDX_s102
                                 + 0.021500365855141 w_k1_FDX_s103
                                 + 0.042335878475369 w_k1_FDX_s104
                                 + 0.0487596200184983 w_k1_FDX_s105
                                 + 0.0840197766328181 w_k1_FDX_s106
                                 + 0.040126818585278 w_k1_FDX_s107
                                 + 0.0265961810620137 w_k1_FDX_s110
                                 + 0.0177891961813038 w_k1_FDX_s111
                                 + 0.036571673296537 w_k1_FDX_s112
                                 + 0.0349785222116423 w_k1_FDX_s113
                                 + 0.0294267007582879 w_k1_FDX_s114
                                 + 0.0409688819608722 w_k1_FDX_s115
                                 + 0.0604181889232568 w_k1_FDX_s116
                                 + 0.0179189897639287 w_k1_FDX_s117
                                 + 0.016911224557147 w_k1_IBM_s000
                                 + 0.0199041777561817 w_k1_IBM_s001
                                 + 0.0149932499687246 w_k1_IBM_s002
                                 + 0.0334953963757408 w_k1_IBM_s003
                                 + 0.0412884257276881 w_k1_IBM_s004
                                 + 0.0426798459142874 w_k1_IBM_s005
                                 + 0.0192199430057999 w_k1_IBM_s006
                                 + 0.0258403326974421 w_k1_IBM_s007
                                 + 0.0294630575916496 w_k1_IBM_s010
                                 + 0.0150352141641724 w_k1_IBM_s011
                                 + 0.0126642945870307 w_k1_IBM_s012
                                 + 0.0370196542676692 w_k1_IBM_s013
                                 + 0.0223123261366164 w_k1_IBM_s014
                                 + 0.0151193864598451 w_k1_IBM_s015
                                 + 0.0322841348774599 w_k1_IBM_s016
                                 + 0.0172769222535888 w_k1_IBM_s017
                                 + 0.0321960807935664 w_k1_IBM_s100
                                 + 0.0571920529844854 w_k1_IBM_s101
                                 + 0.0358050044836183 w_k1_IBM_s102
                                 + 0.021403346027962 w_k1_IBM_s103
                                 + 0.0422234099402274 w_k1_IBM_s104
                                 + 0.0508782508322017 w_k1_IBM_s105
                                 + 0.0874054434289351 w_k1_IBM_s106
                                 + 0.0431943188486413 w_k1_IBM_s107
                                 + 0.0279458974075968 w_k1_IBM_s110
                                 + 0.0183579076794132 w_k1_IBM_s111
                                 + 0.0353961404696746 w_k1_IBM_s112
                                 + 0.0346627940077738 w_k1_IBM_s113
                                 + 0.0303072977669484 w_k1_IBM_s114
                                 + 0.0434485675015381 w_k1_IBM_s115
                                 + 0.0624852728591095 w_k1_IBM_s116
                                 + 0.019008162642737 w_k1_IBM_s117
                                 + 0.0166567598956869 w_k1_INTC_s000
                                 + 0.021826659121995 w_k1_INTC_s001
                                 + 0.013417255294668 w_k1_INTC_s002
                                 + 0.0279739068448204 w_k1_INTC_s003
                                 + 0.0408424177048233 w_k1_INTC_s004
                                 + 0.0391258040306003 w_k1_INTC_s005
                                 + 0.0208906964169653 w_k1_INTC_s006
                                 + 0.0248198786384393 w_k1_INTC_s007
                                 + 0.0346904958900911 w_k1_INTC_s010
                                 + 0.0134159495110783 w_k1_INTC_s011
                                 + 0.0133727277923643 w_k1_INTC_s012
                                 + 0.0371484246585243 w_k1_INTC_s013
                                 + 0.0242812907986419 w_k1_INTC_s014
                                 + 0.0172234412729328 w_k1_INTC_s015
                                 + 0.0325561513265026 w_k1_INTC_s016
                                 + 0.0193799177370909 w_k1_INTC_s017
                                 + 0.0361609281789002 w_k1_INTC_s100
                                 + 0.0530502005730426 w_k1_INTC_s101
                                 + 0.0408324773840198 w_k1_INTC_s102
                                 + 0.024338419072781 w_k1_INTC_s103
                                 + 0.0360254156659235 w_k1_INTC_s104
                                 + 0.050238464081439 w_k1_INTC_s105
                                 + 0.0858390437789618 w_k1_INTC_s106
                                 + 0.0364878383559988 w_k1_INTC_s107
                                 + 0.0261836990552275 w_k1_INTC_s110
                                 + 0.0153123690375727 w_k1_INTC_s111
                                 + 0.0337157145232463 w_k1_INTC_s112
                                 + 0.0347000018550859 w_k1_INTC_s113
                                 + 0.0313492876169879 w_k1_INTC_s114
                                 + 0.0419139062516458 w_k1_INTC_s115
                                 + 0.0625421955195763 w_k1_INTC_s116
                                 + 0.018685800599287 w_k1_INTC_s117
                                 + 0.0169103146757573 w_k1_KO_s000
                                 + 0.0199311251325433 w_k1_KO_s001
                                 + 0.014325673158258 w_k1_KO_s002
                                 + 0.031973819981573 w_k1_KO_s003
                                 + 0.036794545655109 w_k1_KO_s004
                                 + 0.0416573705861401 w_k1_KO_s005
                                 + 0.0209339803416822 w_k1_KO_s006
                                 + 0.0246095937512975 w_k1_KO_s007
                                 + 0.0280177787547534 w_k1_KO_s010
                                 + 0.0138012513248199 w_k1_KO_s011
                                 + 0.0123165577019051 w_k1_KO_s012
                                 + 0.033184903989748 w_k1_KO_s013
                                 + 0.0241280583983379 w_k1_KO_s014
                                 + 0.0151688513634369 w_k1_KO_s015
                                 + 0.0361592961387701 w_k1_KO_s016
                                 + 0.0169346824506096 w_k1_KO_s017
                                 + 0.0307420431209212 w_k1_KO_s100
                                 + 0.0520201813426699 w_k1_KO_s101
                                 + 0.0374210171660897 w_k1_KO_s102
                                 + 0.0191370017021194 w_k1_KO_s103
                                 + 0.0415990033292856 w_k1_KO_s104
                                 + 0.0513434354084352 w_k1_KO_s105
                                 + 0.0819305015942815 w_k1_KO_s106
                                 + 0.0404655010321393 w_k1_KO_s107
                                 + 0.0279977558198719 w_k1_KO_s110
                                 + 0.0181623564847001 w_k1_KO_s111
                                 + 0.0436946270105515 w_k1_KO_s112
                                 + 0.036054544731233 w_k1_KO_s113
                                 + 0.0304452035628594 w_k1_KO_s114
                                 + 0.0395585092992108 w_k1_KO_s115
                                 + 0.0642183982216618 w_k1_KO_s116
                                 + 0.0191824463192822 w_k1_KO_s117
                                 + 0.0167352130450963 w_k1_MAT_s000
                                 + 0.0191542343810399 w_k1_MAT_s001
                                 + 0.0127951303537386 w_k1_MAT_s002
                                 + 0.035822609897491 w_k1_MAT_s003
                                 + 0.0416654248846688 w_k1_MAT_s004
                                 + 0.0507326098113159 w_k1_MAT_s005
                                 + 0.0171166887451495 w_k1_MAT_s006
                                 + 0.0205028073525043 w_k1_MAT_s007
                                 + 0.0274857246408098 w_k1_MAT_s010
                                 + 0.0151115690627573 w_k1_MAT_s011
                                 + 0.0152031846720269 w_k1_MAT_s012
                                 + 0.0306593127916445 w_k1_MAT_s013
                                 + 0.0219946891622075 w_k1_MAT_s014
                                 + 0.0215968989454947 w_k1_MAT_s015
                                 + 0.0320231177092603 w_k1_MAT_s016
                                 + 0.0145045703211478 w_k1_MAT_s017
                                 + 0.0304872158309173 w_k1_MAT_s100
                                 + 0.053924941099298 w_k1_MAT_s101
                                 + 0.0367732839017798 w_k1_MAT_s102
                                 + 0.0209533242963386 w_k1_MAT_s103
                                 + 0.059493657307547 w_k1_MAT_s104
                                 + 0.0465943882840235 w_k1_MAT_s105
                                 + 0.087678945413468 w_k1_MAT_s106
                                 + 0.0432545206549149 w_k1_MAT_s107
                                 + 0.0269311473816115 w_k1_MAT_s110
                                 + 0.0194679166171635 w_k1_MAT_s111
                                 + 0.0353414297687481 w_k1_MAT_s112
                                 + 0.0239345162452378 w_k1_MAT_s113
                                 + 0.0321593771503645 w_k1_MAT_s114
                                 + 0.0432505584904793 w_k1_MAT_s115
                                 + 0.0671997715931009 w_k1_MAT_s116
                                 + 0.0215878902601783 w_k1_MAT_s117
                                 + 0.0161866461396892 w_k1_MSFT_s000
                                 + 0.0193683238628314 w_k1_MSFT_s001
                                 + 0.0136989290825599 w_k1_MSFT_s002
                                 + 0.0311698041702359 w_k1_MSFT_s003
                                 + 0.0370183669861467 w_k1_MSFT_s004
                                 + 0.0419556493248482 w_k1_MSFT_s005
                                 + 0.0209118695399108 w_k1_MSFT_s006
                                 + 0.0244621748262062 w_k1_MSFT_s007
                                 + 0.0271217042064736 w_k1_MSFT_s010
                                 + 0.0142072557595332 w_k1_MSFT_s011
                                 + 0.011712484965942 w_k1_MSFT_s012
                                 + 0.0352486892467595 w_k1_MSFT_s013
                                 + 0.0232984314784086 w_k1_MSFT_s014
                                 + 0.0161692011410223 w_k1_MSFT_s015
                                 + 0.0316477007455756 w_k1_MSFT_s016
                                 + 0.0175169055805803 w_k1_MSFT_s017
                                 + 0.0302770471053168 w_k1_MSFT_s100
                                 + 0.0512227900118154 w_k1_MSFT_s101
                                 + 0.0364186849814499 w_k1_MSFT_s102
                                 + 0.0226064836636273 w_k1_MSFT_s103
                                 + 0.0399894099831323 w_k1_MSFT_s104
                                 + 0.0495039579525968 w_k1_MSFT_s105
                                 + 0.0906126312517762 w_k1_MSFT_s106
                                 + 0.0401660162827364 w_k1_MSFT_s107
                                 + 0.0261399018048398 w_k1_MSFT_s110
                                 + 0.0169441201062667 w_k1_MSFT_s111
                                 + 0.0376673932576318 w_k1_MSFT_s112
                                 + 0.0344394029396989 w_k1_MSFT_s113
                                 + 0.0288535243991535 w_k1_MSFT_s114
                                 + 0.0433759661734172 w_k1_MSFT_s115
                                 + 0.0603280638704143 w_k1_MSFT_s116
                                 + 0.0191195965717421 w_k1_MSFT_s117
                                 + 0.0167819153334058 w_k1_PG_s000
                                 + 0.0202557265543921 w_k1_PG_s001
                                 + 0.0151360659106971 w_k1_PG_s002
                                 + 0.0333243926439814 w_k1_PG_s003
                                 + 0.0388639537281647 w_k1_PG_s004
                                 + 0.0429405463162382 w_k1_PG_s005
                                 + 0.0193157148470058 w_k1_PG_s006
                                 + 0.0235580808795504 w_k1_PG_s007
                                 + 0.0287748084583881 w_k1_PG_s010
                                 + 0.0136028326846297 w_k1_PG_s011
                                 + 0.0133246304213255 w_k1_PG_s012
                                 + 0.0337598708207802 w_k1_PG_s013
                                 + 0.0240280290177679 w_k1_PG_s014
                                 + 0.0136869556718827 w_k1_PG_s015
                                 + 0.0328142417809155 w_k1_PG_s016
                                 + 0.0155967809053915 w_k1_PG_s017
                                 + 0.0308763612075571 w_k1_PG_s100
                                 + 0.0533491566115128 w_k1_PG_s101
                                 + 0.0353566470703413 w_k1_PG_s102
                                 + 0.0214506804597372 w_k1_PG_s103
                                 + 0.0396920016651337 w_k1_PG_s104
                                 + 0.051961331372635 w_k1_PG_s105
                                 + 0.0847824477956069 w_k1_PG_s106
                                 + 0.0414831375770555 w_k1_PG_s107
                                 + 0.027226041290677 w_k1_PG_s110
                                 + 0.0184181489518539 w_k1_PG_s111
                                 + 0.0362939792032531 w_k1_PG_s112
                                 + 0.0342287648941763 w_k1_PG_s113
                                 + 0.028849369748332 w_k1_PG_s114
                                 + 0.0416349356474964 w_k1_PG_s115
                                 + 0.0593183291300351 w_k1_PG_s116
                                 + 0.0173507870311345 w_k1_PG_s117 >= 1.209
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
