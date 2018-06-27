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
 w_k1_ADM_s0_return_constraint:  1.15886 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.930935 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.0079 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.05152 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.01492 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.00553 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.26904 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.37742 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.653086 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.949152 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.01901 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.790949 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.760519 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.07782 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.23195 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.23144 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.828489 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.22503 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.08784 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.968852 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.52369 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.827181 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.954788 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.26068 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.27699 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.960863 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.941225 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.19131 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.799679 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.231 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.983546 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.22273 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.879707 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.872028 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.965791 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.34518 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.929046 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.31936 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.26933 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.911456 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.885505 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.1933 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.970125 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.1129 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.21539 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.54 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.582709 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.741951 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.19564 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.675334 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.836746 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.96218 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.1155 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.35569 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.647113 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.12507 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.08581 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.02437 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.35719 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.930928 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.07663 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.60916 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.31596 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.959835 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.820877 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.26486 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.714453 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.44275 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.984259 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.29455 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.867449 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.6225 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.01454 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.50849 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.745223 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.48297 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.34243 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.884191 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.828305 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.31912 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.995563 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.23578 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.58261 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.61218 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.542165 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.873586 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.1813 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.802818 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.80709 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.05622 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.82064 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.32822 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.606149 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.09787 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.916132 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.05828 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.45906 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.745506 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.11134 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.61039 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 2.00903 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.849545 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.747492 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.11789 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.674605 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.33193 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.960412 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.2523 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.632648 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.488828 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.973818 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.43507 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.719445 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.78315 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.18989 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.90917 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.974591 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.17726 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.992291 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.04918 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.2972 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.28732 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.680909 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.997469 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.13381 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.84143 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.797788 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.931988 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.3762 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.12181 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.689363 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.06563 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.08715 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.932961 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.40533 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.843774 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.02892 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.50009 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.49488 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.0912 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.812487 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.21691 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.826914 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.13897 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.03359 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.15727 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.804526 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.670251 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.947522 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.28088 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.695693 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.43532 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.15206 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.970366 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.800126 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.34194 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.8891 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.5867 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.981343 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.58815 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.691321 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.520356 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.01041 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.782085 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.27508 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.11098 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.98052 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.64647 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.778591 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.886611 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.647978 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.24875 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.678776 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.05436 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.28373 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.07781 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.42504 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.751752 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.16393 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.02016 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.696795 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.92515 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.758728 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.17505 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.945809 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.684573 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.09571 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.63468 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.30898 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.02336 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.13501 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.964162 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.920707 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.08538 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.922638 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.962442 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.23792 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.33066 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.833576 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.91877 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.991094 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.888122 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.00185 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.982229 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.25312 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.41035 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.877428 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.02947 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.07118 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.13431 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.26968 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.848437 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.01999 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.17159 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.41753 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.957292 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.799729 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.08903 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.783684 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.311 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.0713 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.18653 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.900764 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.758548 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.969417 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.43215 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.848172 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.33463 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.16308 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.976083 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.837531 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.20397 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.934148 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.23449 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.30247 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.62439 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.729726 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.845386 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.10387 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.882207 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.03557 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.10435 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.47453 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.66559 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.79085 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.01073 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.945296 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.19313 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.11145 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.990269 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.06487 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.18037 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.57644 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.829935 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.855012 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.1745 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.680651 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.50504 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.863379 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.12978 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.946412 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.681573 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.01775 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.48747 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.906743 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.25848 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.09508 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.920512 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.921441 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.53843 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.934097 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.52229 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.903121 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.52046 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.549396 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.78568 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.31843 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.639683 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.33426 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.874158 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.971522 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.63151 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.615781 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.03322 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.796563 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.05294 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.689723 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.01178 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.31716 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.45258 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.36876 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.823055 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.09889 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.04448 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.598496 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.79399 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.649827 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.08789 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.11334 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.593615 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.00727 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.61489 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.26151 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.07793 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.17707 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.94062 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.889734 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.12126 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.975358 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.06223 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.44191 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.40961 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.76716 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.0155 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.06639 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.912378 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.857081 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.02365 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.40934 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.05881 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.838443 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.12754 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.01476 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.991117 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.35173 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.872611 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.05212 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.20643 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.5245 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.977232 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.793098 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.17866 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.844879 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.2075 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.02643 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.11306 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.837061 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.664478 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.906572 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.1136 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.796263 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.36692 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.14721 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.943494 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.905601 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.05172 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.00402 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.10938 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.25355 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.24997 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.760024 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.835207 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.06071 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.850466 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.98575 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.04749 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.30617 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.23692 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.869124 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.14344 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.09289 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.01188 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.16675 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.01483 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.996863 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.16646 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.37937 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.931981 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.861164 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.04214 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.833547 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.13082 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.958933 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.10058 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.861831 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.776945 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.00446 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.25407 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.932999 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.23709 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.47300052024799 z_k1_s0000
                                 - 0.195999887367415 z_k1_s0001
                                 - 0.188999913132839 z_k1_s0002
                                 - 0.642998589946354 z_k1_s0003
                                 - 0.415998642740307 z_k1_s0004
                                 - 0.178999775997933 z_k1_s0005
                                 - 0.788997791639241 z_k1_s0006
                                 - 0.158999806127767 z_k1_s0007
                                 - 0.213999591853177 z_k1_s0010
                                 - 0.0900000638887003 z_k1_s0011
                                 - 0.349999906956084 z_k1_s0012
                                 - 0.255999990487874 z_k1_s0013
                                 - 0.448999383023702 z_k1_s0014
                                 - 0.144999932181809 z_k1_s0015
                                 - 0.207000208414006 z_k1_s0016
                                 - 0.338000963917505 z_k1_s0017
                                 - 0.205000509876363 z_k1_s0020
                                 - 0.196000509442767 z_k1_s0021
                                 - 0.157999507611983 z_k1_s0022
                                 - 0.346999516717486 z_k1_s0023
                                 - 0.103999955010433 z_k1_s0024
                                 - 0.153000007371121 z_k1_s0025
                                 - 0.123000005925803 z_k1_s0026
                                 - 0.214000510309958 z_k1_s0027
                                 - 0.0840000081677274 z_k1_s0030
                                 - 0.131000452259309 z_k1_s0031
                                 - 0.0880000358138168 z_k1_s0032
                                 - 0.128000121474661 z_k1_s0033
                                 - 0.101000006413576 z_k1_s0034
                                 - 0.127000329213194 z_k1_s0035
                                 - 0.101000006413576 z_k1_s0036
                                 - 0.193999766734923 z_k1_s0037
                                 - 0.225999989585301 z_k1_s0040
                                 - 0.309999992029396 z_k1_s0041
                                 - 0.214999844846184 z_k1_s0042
                                 - 0.534001187880395 z_k1_s0043
                                 - 0.736998993520143 z_k1_s0044
                                 - 0.260000177469829 z_k1_s0045
                                 - 0.516000626670909 z_k1_s0046
                                 - 0.769999427737495 z_k1_s0047
                                 - 0.284999660263292 z_k1_s0050
                                 - 0.314999571812151 z_k1_s0051
                                 - 1.05299853049211 z_k1_s0052
                                 - 0.223999740004119 z_k1_s0053
                                 - 0.51100059568858 z_k1_s0054
                                 - 0.301999309811501 z_k1_s0055
                                 - 0.26299952490781 z_k1_s0056
                                 - 0.383997933467022 z_k1_s0057
                                 - 0.811998150641314 z_k1_s0060
                                 - 0.287999609890093 z_k1_s0061
                                 - 0.246999721393151 z_k1_s0062
                                 - 0.195999783175125 z_k1_s0063
                                 - 0.391000227137085 z_k1_s0064
                                 - 0.878998028253295 z_k1_s0065
                                 - 0.411999665792417 z_k1_s0066
                                 - 0.278999600181062 z_k1_s0067
                                 - 0.409999649413698 z_k1_s0070
                                 - 0.359001180975846 z_k1_s0071
                                 - 0.17400031026381 z_k1_s0072
                                 - 0.479001394950888 z_k1_s0073
                                 - 0.572999399229282 z_k1_s0074
                                 - 0.305000435686461 z_k1_s0075
                                 - 0.547000434535717 z_k1_s0076
                                 - 0.397999628016194 z_k1_s0077
                                 - 0.0590000057373666 z_k1_s0100
                                 - 0.0759999852040701 z_k1_s0101
                                 - 0.102000338800125 z_k1_s0102
                                 - 0.186999851133722 z_k1_s0103
                                 - 0.0649999984903208 z_k1_s0104
                                 - 0.0420000262706632 z_k1_s0105
                                 - 0.159000346953173 z_k1_s0106
                                 - 0.0799999033727222 z_k1_s0107
                                 - 0.154999474912638 z_k1_s0110
                                 - 0.274999787810483 z_k1_s0111
                                 - 0.341999033264701 z_k1_s0112
                                 - 0.135999505020228 z_k1_s0113
                                 - 0.148999618567245 z_k1_s0114
                                 - 0.27799886638585 z_k1_s0115
                                 - 0.265998304097734 z_k1_s0116
                                 - 0.522999372469369 z_k1_s0117
                                 - 0.516999295080879 z_k1_s0120
                                 - 0.162999727425973 z_k1_s0121
                                 - 0.529998152705739 z_k1_s0122
                                 - 0.631997767132178 z_k1_s0123
                                 - 0.288998907541578 z_k1_s0124
                                 - 0.138999630737722 z_k1_s0125
                                 - 0.247999671614216 z_k1_s0126
                                 - 0.136999888179911 z_k1_s0127
                                 - 0.249999689454856 z_k1_s0130
                                 - 0.319999487230355 z_k1_s0131
                                 - 0.316999537065641 z_k1_s0132
                                 - 0.668000430529047 z_k1_s0133
                                 - 0.179999891679356 z_k1_s0134
                                 - 0.479998430346502 z_k1_s0135
                                 - 0.182999841844071 z_k1_s0136
                                 - 0.803998811728646 z_k1_s0137
                                 - 0.165000590342079 z_k1_s0140
                                 - 0.258000663080278 z_k1_s0141
                                 - 0.0830000802932353 z_k1_s0142
                                 - 0.202999989632802 z_k1_s0143
                                 - 0.249000240879706 z_k1_s0144
                                 - 0.232000396723302 z_k1_s0145
                                 - 0.108000061405645 z_k1_s0146
                                 - 0.132000043273558 z_k1_s0147
                                 - 0.198999932271929 z_k1_s0150
                                 - 0.106000303975941 z_k1_s0151
                                 - 0.158001169490655 z_k1_s0152
                                 - 0.156000146937344 z_k1_s0153
                                 - 0.26700101699104 z_k1_s0154
                                 - 0.321001067853967 z_k1_s0155
                                 - 0.153000450312476 z_k1_s0156
                                 - 0.171000926568059 z_k1_s0157
                                 - 0.111000212083196 z_k1_s0160
                                 - 0.0799998289340212 z_k1_s0161
                                 - 0.13200031699079 z_k1_s0162
                                 - 0.068999922368053 z_k1_s0163
                                 - 0.13699965711204 z_k1_s0164
                                 - 0.0900001071756013 z_k1_s0165
                                 - 0.070999978016369 z_k1_s0166
                                 - 0.109000316234788 z_k1_s0167
                                 - 0.150000327661881 z_k1_s0170
                                 - 0.400001734433568 z_k1_s0171
                                 - 0.175000339238767 z_k1_s0172
                                 - 0.216000523473221 z_k1_s0173
                                 - 0.391000862711988 z_k1_s0174
                                 - 0.181000403985355 z_k1_s0175
                                 - 0.684001700698629 z_k1_s0176
                                 - 0.385000281564268 z_k1_s0177
                                 - 0.407001159345033 z_k1_s1000
                                 - 1.15300114268073 z_k1_s1001
                                 - 0.471001558820321 z_k1_s1002
                                 - 0.474001204049313 z_k1_s1003
                                 - 0.766999723545188 z_k1_s1004
                                 - 0.342001248727838 z_k1_s1005
                                 - 0.351001296218087 z_k1_s1006
                                 - 1.59400346970295 z_k1_s1007
                                 - 0.334000304472665 z_k1_s1010
                                 - 0.373000329378145 z_k1_s1011
                                 - 1.0400021458143 z_k1_s1012
                                 - 0.356000409701489 z_k1_s1013
                                 - 0.395000434606969 z_k1_s1014
                                 - 1.03400138975116 z_k1_s1015
                                 - 0.37700038891979 z_k1_s1016
                                 - 0.536000866573281 z_k1_s1017
                                 - 0.309000200625062 z_k1_s1020
                                 - 0.231000200574085 z_k1_s1021
                                 - 0.559001447865612 z_k1_s1022
                                 - 0.397998886798359 z_k1_s1023
                                 - 0.220000124851497 z_k1_s1024
                                 - 0.336000133835002 z_k1_s1025
                                 - 0.92899988926097 z_k1_s1026
                                 - 0.491998396936621 z_k1_s1027
                                 - 0.356000419053474 z_k1_s1030
                                 - 0.811999753138737 z_k1_s1031
                                 - 0.564000383569829 z_k1_s1032
                                 - 0.432998931739295 z_k1_s1033
                                 - 0.561002500552973 z_k1_s1034
                                 - 0.354000489040887 z_k1_s1035
                                 - 0.518999946284593 z_k1_s1036
                                 - 0.424999211688947 z_k1_s1037
                                 - 1.32900054837563 z_k1_s1040
                                 - 1.16299774503368 z_k1_s1041
                                 - 0.976002005992239 z_k1_s1042
                                 - 1.25999851267782 z_k1_s1043
                                 - 1.13399866141004 z_k1_s1044
                                 - 1.34600652689007 z_k1_s1045
                                 - 1.54900011225559 z_k1_s1046
                                 - 2.3580067437416 z_k1_s1047
                                 - 0.566000398823924 z_k1_s1050
                                 - 0.456000472743014 z_k1_s1051
                                 - 0.936000758914517 z_k1_s1052
                                 - 1.37499740201066 z_k1_s1053
                                 - 0.375000265421505 z_k1_s1054
                                 - 0.944002482324774 z_k1_s1055
                                 - 0.554000492109679 z_k1_s1056
                                 - 1.4900012510242 z_k1_s1057
                                 - 1.58699522486354 z_k1_s1060
                                 - 1.1489994659508 z_k1_s1061
                                 - 1.30999942094981 z_k1_s1062
                                 - 0.792999295245648 z_k1_s1063
                                 - 3.47199203682516 z_k1_s1064
                                 - 1.77900553745294 z_k1_s1065
                                 - 1.41399479011512 z_k1_s1066
                                 - 1.67899934237548 z_k1_s1067
                                 - 0.203000173694778 z_k1_s1070
                                 - 0.424000270458986 z_k1_s1071
                                 - 0.534999448953586 z_k1_s1072
                                 - 0.605998798405009 z_k1_s1073
                                 - 0.3769994300326 z_k1_s1074
                                 - 0.221000096764208 z_k1_s1075
                                 - 0.188000237803586 z_k1_s1076
                                 - 0.91700059311767 z_k1_s1077
                                 - 0.203999864419253 z_k1_s1100
                                 - 0.544999127426754 z_k1_s1101
                                 - 0.488999533421239 z_k1_s1102
                                 - 0.22900029889633 z_k1_s1103
                                 - 0.189999853967944 z_k1_s1104
                                 - 0.182999848742289 z_k1_s1105
                                 - 0.192999888193204 z_k1_s1106
                                 - 0.20599996186938 z_k1_s1107
                                 - 0.195999185454867 z_k1_s1110
                                 - 0.253999951841609 z_k1_s1111
                                 - 0.19800018456879 z_k1_s1112
                                 - 0.109999593228604 z_k1_s1113
                                 - 0.113999652162154 z_k1_s1114
                                 - 0.214998980565658 z_k1_s1115
                                 - 0.28799930683016 z_k1_s1116
                                 - 0.387998135676702 z_k1_s1117
                                 - 0.0900000992166176 z_k1_s1120
                                 - 0.101000030163083 z_k1_s1121
                                 - 0.247999188356769 z_k1_s1122
                                 - 0.326000901904703 z_k1_s1123
                                 - 0.436000211369359 z_k1_s1124
                                 - 0.108000083419937 z_k1_s1125
                                 - 0.126999947945648 z_k1_s1126
                                 - 0.346000290352763 z_k1_s1127
                                 - 0.111999418780541 z_k1_s1130
                                 - 0.112999277590083 z_k1_s1131
                                 - 0.174998283536776 z_k1_s1132
                                 - 0.283998413287025 z_k1_s1133
                                 - 0.247998897770127 z_k1_s1134
                                 - 0.0649995882694947 z_k1_s1135
                                 - 0.240999311306659 z_k1_s1136
                                 - 0.198998918542493 z_k1_s1137
                                 - 0.384998393749626 z_k1_s1140
                                 - 0.132999360022997 z_k1_s1141
                                 - 0.284998182908514 z_k1_s1142
                                 - 0.140999314490579 z_k1_s1143
                                 - 0.184999532060083 z_k1_s1144
                                 - 0.187999397985975 z_k1_s1145
                                 - 0.0999995868440397 z_k1_s1146
                                 - 0.142999381107109 z_k1_s1147
                                 - 0.103000171423073 z_k1_s1150
                                 - 0.096000178748733 z_k1_s1151
                                 - 0.216000109008976 z_k1_s1152
                                 - 0.105000281015998 z_k1_s1153
                                 - 0.156000730230201 z_k1_s1154
                                 - 0.139000450191834 z_k1_s1155
                                 - 0.217000815306935 z_k1_s1156
                                 - 0.271001168309927 z_k1_s1157
                                 - 0.111999615326236 z_k1_s1160
                                 - 0.0429999403090929 z_k1_s1161
                                 - 0.0630001417456539 z_k1_s1162
                                 - 0.0690000188168033 z_k1_s1163
                                 - 0.034999951414378 z_k1_s1164
                                 - 0.0619996847342675 z_k1_s1165
                                 - 0.0439999389209323 z_k1_s1166
                                 - 0.144999798716709 z_k1_s1167
                                 - 0.100999857332235 z_k1_s1170
                                 - 0.100999857332235 z_k1_s1171
                                 - 0.120000066176469 z_k1_s1172
                                 - 0.166000127610714 z_k1_s1173
                                 - 0.0609999074066093 z_k1_s1174
                                 - 0.171999806320149 z_k1_s1175
                                 - 0.231999839408383 z_k1_s1176
                                 - 0.129000125239602 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.817645 w_k1_ADM_s000
                                 + 1.16799 w_k1_AXP_s000
                                 + 1.17778 w_k1_DIS_s000
                                 + 1.05839 w_k1_FDX_s000
                                 + 1.45856 w_k1_IBM_s000
                                 + 1.00299 w_k1_INTC_s000
                                 + 1.33856 w_k1_KO_s000 + 1.73536 w_k1_MAT_s000
                                 + 1.05576 w_k1_MSFT_s000
                                 + 1.08754 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.13292 w_k1_ADM_s000
                                 + 1.21056 w_k1_AXP_s000
                                 + 0.83078 w_k1_DIS_s000
                                 + 1.09344 w_k1_FDX_s000
                                 + 0.642678 w_k1_IBM_s000
                                 + 0.875247 w_k1_INTC_s000
                                 + 0.749103 w_k1_KO_s000
                                 + 0.641219 w_k1_MAT_s000
                                 + 0.856975 w_k1_MSFT_s000
                                 + 0.865936 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.10498 w_k1_ADM_s000
                                 + 1.65536 w_k1_AXP_s000
                                 + 1.74095 w_k1_DIS_s000
                                 + 1.66508 w_k1_FDX_s000
                                 + 1.21676 w_k1_IBM_s000
                                 + 1.38059 w_k1_INTC_s000 + 1.445 w_k1_KO_s000
                                 + 1.77874 w_k1_MAT_s000
                                 + 1.36575 w_k1_MSFT_s000
                                 + 1.17503 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.954263 w_k1_ADM_s000
                                 + 0.96751 w_k1_AXP_s000
                                 + 1.04063 w_k1_DIS_s000
                                 + 1.02568 w_k1_FDX_s000
                                 + 0.743372 w_k1_IBM_s000
                                 + 1.21422 w_k1_INTC_s000
                                 + 0.928806 w_k1_KO_s000
                                 + 0.797288 w_k1_MAT_s000
                                 + 0.99727 w_k1_MSFT_s000
                                 + 0.976617 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.17668 w_k1_ADM_s000
                                 + 1.39987 w_k1_AXP_s000
                                 + 1.39606 w_k1_DIS_s000
                                 + 1.26575 w_k1_FDX_s000
                                 + 1.15656 w_k1_IBM_s000
                                 + 1.15938 w_k1_INTC_s000
                                 + 0.977913 w_k1_KO_s000
                                 + 1.01938 w_k1_MAT_s000
                                 + 1.03662 w_k1_MSFT_s000
                                 + 1.01588 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.595761 w_k1_ADM_s000
                                 + 0.556757 w_k1_AXP_s000
                                 + 0.624304 w_k1_DIS_s000
                                 + 0.738426 w_k1_FDX_s000
                                 + 0.780701 w_k1_IBM_s000
                                 + 0.855047 w_k1_INTC_s000
                                 + 0.756802 w_k1_KO_s000
                                 + 0.862198 w_k1_MAT_s000
                                 + 0.767149 w_k1_MSFT_s000
                                 + 0.761761 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.850316 w_k1_ADM_s000
                                 + 0.848597 w_k1_AXP_s000
                                 + 0.85709 w_k1_DIS_s000
                                 + 0.87297 w_k1_FDX_s000 + 1.0789 w_k1_IBM_s000
                                 + 0.954989 w_k1_INTC_s000
                                 + 0.988088 w_k1_KO_s000
                                 + 1.21925 w_k1_MAT_s000
                                 + 0.868402 w_k1_MSFT_s000
                                 + 0.904215 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.835398 w_k1_ADM_s000
                                 + 0.75386 w_k1_AXP_s000
                                 + 0.636663 w_k1_DIS_s000
                                 + 0.878138 w_k1_FDX_s000
                                 + 0.668759 w_k1_IBM_s000
                                 + 0.739777 w_k1_INTC_s000
                                 + 0.643485 w_k1_KO_s000
                                 + 0.600597 w_k1_MAT_s000
                                 + 0.812351 w_k1_MSFT_s000
                                 + 0.830149 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.536612 w_k1_ADM_s001
                                 + 0.602391 w_k1_AXP_s001
                                 + 0.658097 w_k1_DIS_s001
                                 + 0.666671 w_k1_FDX_s001
                                 + 0.84935 w_k1_IBM_s001
                                 + 0.728731 w_k1_INTC_s001
                                 + 0.905517 w_k1_KO_s001
                                 + 1.05522 w_k1_MAT_s001
                                 + 0.731573 w_k1_MSFT_s001
                                 + 0.807297 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.794946 w_k1_ADM_s001
                                 + 1.26861 w_k1_AXP_s001
                                 + 1.33784 w_k1_DIS_s001
                                 + 0.933079 w_k1_FDX_s001
                                 + 1.97896 w_k1_IBM_s001
                                 + 1.21098 w_k1_INTC_s001
                                 + 1.42481 w_k1_KO_s001 + 1.9473 w_k1_MAT_s001
                                 + 1.04293 w_k1_MSFT_s001
                                 + 1.21033 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.975189 w_k1_ADM_s001
                                 + 0.917119 w_k1_AXP_s001
                                 + 0.967744 w_k1_DIS_s001
                                 + 0.925814 w_k1_FDX_s001
                                 + 0.854308 w_k1_IBM_s001
                                 + 0.962577 w_k1_INTC_s001
                                 + 0.879042 w_k1_KO_s001
                                 + 1.01223 w_k1_MAT_s001
                                 + 0.994634 w_k1_MSFT_s001
                                 + 0.967175 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.944903 w_k1_ADM_s001
                                 + 1.25124 w_k1_AXP_s001 + 1.667 w_k1_DIS_s001
                                 + 1.43081 w_k1_FDX_s001
                                 + 0.939167 w_k1_IBM_s001
                                 + 1.28073 w_k1_INTC_s001
                                 + 1.13147 w_k1_KO_s001
                                 + 0.780724 w_k1_MAT_s001
                                 + 1.18131 w_k1_MSFT_s001
                                 + 1.09769 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.843559 w_k1_ADM_s001
                                 + 1.0443 w_k1_AXP_s001 + 1.09513 w_k1_DIS_s001
                                 + 1.05512 w_k1_FDX_s001
                                 + 1.04565 w_k1_IBM_s001
                                 + 1.12635 w_k1_INTC_s001
                                 + 0.998346 w_k1_KO_s001
                                 + 1.30748 w_k1_MAT_s001
                                 + 1.11024 w_k1_MSFT_s001
                                 + 0.994662 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.821763 w_k1_ADM_s001
                                 + 0.813268 w_k1_AXP_s001
                                 + 0.836115 w_k1_DIS_s001
                                 + 0.785862 w_k1_FDX_s001
                                 + 1.2674 w_k1_IBM_s001
                                 + 0.925042 w_k1_INTC_s001
                                 + 0.973709 w_k1_KO_s001
                                 + 0.956526 w_k1_MAT_s001
                                 + 0.93091 w_k1_MSFT_s001
                                 + 0.970475 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.06609 w_k1_ADM_s001
                                 + 1.43756 w_k1_AXP_s001
                                 + 1.17518 w_k1_DIS_s001
                                 + 1.18282 w_k1_FDX_s001
                                 + 1.32283 w_k1_IBM_s001
                                 + 1.26929 w_k1_INTC_s001
                                 + 1.09699 w_k1_KO_s001 + 1.78288 w_k1_MAT_s001
                                 + 1.10915 w_k1_MSFT_s001
                                 + 1.20534 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.725255 w_k1_ADM_s001
                                 + 0.897914 w_k1_AXP_s001
                                 + 0.808599 w_k1_DIS_s001
                                 + 0.891209 w_k1_FDX_s001
                                 + 0.727546 w_k1_IBM_s001
                                 + 0.874152 w_k1_INTC_s001
                                 + 0.653411 w_k1_KO_s001
                                 + 0.762314 w_k1_MAT_s001
                                 + 0.733892 w_k1_MSFT_s001
                                 + 0.89253 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.20247 w_k1_ADM_s002
                                 + 1.08483 w_k1_AXP_s002 + 1.1022 w_k1_DIS_s002
                                 + 1.07661 w_k1_FDX_s002
                                 + 1.06771 w_k1_IBM_s002
                                 + 1.05645 w_k1_INTC_s002
                                 + 0.982389 w_k1_KO_s002
                                 + 1.40427 w_k1_MAT_s002
                                 + 1.14221 w_k1_MSFT_s002 + 1.1305 w_k1_PG_s002
                                 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.925755 w_k1_ADM_s002
                                 + 0.896697 w_k1_AXP_s002
                                 + 0.773974 w_k1_DIS_s002
                                 + 0.834795 w_k1_FDX_s002
                                 + 0.690591 w_k1_IBM_s002
                                 + 0.848377 w_k1_INTC_s002
                                 + 0.714739 w_k1_KO_s002
                                 + 0.472498 w_k1_MAT_s002
                                 + 0.847057 w_k1_MSFT_s002
                                 + 0.91443 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.837361 w_k1_ADM_s002
                                 + 1.26944 w_k1_AXP_s002
                                 + 1.18186 w_k1_DIS_s002
                                 + 1.07433 w_k1_FDX_s002
                                 + 1.59725 w_k1_IBM_s002
                                 + 0.891196 w_k1_INTC_s002
                                 + 1.14981 w_k1_KO_s002 + 1.89095 w_k1_MAT_s002
                                 + 1.03241 w_k1_MSFT_s002
                                 + 1.01291 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.00395 w_k1_ADM_s002
                                 + 0.876188 w_k1_AXP_s002
                                 + 0.695739 w_k1_DIS_s002
                                 + 0.80596 w_k1_FDX_s002
                                 + 0.915303 w_k1_IBM_s002
                                 + 0.87829 w_k1_INTC_s002
                                 + 0.785934 w_k1_KO_s002
                                 + 0.892919 w_k1_MAT_s002
                                 + 0.814065 w_k1_MSFT_s002
                                 + 0.793027 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.960398 w_k1_ADM_s002
                                 + 1.07965 w_k1_AXP_s002
                                 + 0.884979 w_k1_DIS_s002
                                 + 1.10016 w_k1_FDX_s002
                                 + 0.690409 w_k1_IBM_s002
                                 + 1.03039 w_k1_INTC_s002
                                 + 0.731515 w_k1_KO_s002
                                 + 0.861602 w_k1_MAT_s002
                                 + 1.01142 w_k1_MSFT_s002
                                 + 0.901782 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.915268 w_k1_ADM_s002
                                 + 0.865665 w_k1_AXP_s002
                                 + 0.945221 w_k1_DIS_s002
                                 + 0.792396 w_k1_FDX_s002
                                 + 1.49055 w_k1_IBM_s002
                                 + 0.848319 w_k1_INTC_s002
                                 + 1.00258 w_k1_KO_s002 + 1.26876 w_k1_MAT_s002
                                 + 0.866517 w_k1_MSFT_s002
                                 + 1.01113 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.04174 w_k1_ADM_s002
                                 + 1.855 w_k1_AXP_s002 + 1.21253 w_k1_DIS_s002
                                 + 1.29677 w_k1_FDX_s002
                                 + 1.14829 w_k1_IBM_s002
                                 + 1.19089 w_k1_INTC_s002
                                 + 1.02644 w_k1_KO_s002 + 1.87846 w_k1_MAT_s002
                                 + 1.04219 w_k1_MSFT_s002
                                 + 1.13218 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.22107 w_k1_ADM_s002
                                 + 1.22574 w_k1_AXP_s002
                                 + 1.18735 w_k1_DIS_s002
                                 + 1.15474 w_k1_FDX_s002
                                 + 0.769057 w_k1_IBM_s002
                                 + 1.12924 w_k1_INTC_s002 + 1.0346 w_k1_KO_s002
                                 + 0.807634 w_k1_MAT_s002
                                 + 1.07375 w_k1_MSFT_s002
                                 + 1.07003 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.06751 w_k1_ADM_s003
                                 + 1.57019 w_k1_AXP_s003
                                 + 1.31257 w_k1_DIS_s003
                                 + 1.15217 w_k1_FDX_s003
                                 + 1.53317 w_k1_IBM_s003
                                 + 1.10768 w_k1_INTC_s003
                                 + 1.04576 w_k1_KO_s003 + 1.10282 w_k1_MAT_s003
                                 + 1.1719 w_k1_MSFT_s003 + 1.095 w_k1_PG_s003
                                 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.776336 w_k1_ADM_s003
                                 + 0.647253 w_k1_AXP_s003
                                 + 0.669648 w_k1_DIS_s003
                                 + 0.773337 w_k1_FDX_s003
                                 + 0.778793 w_k1_IBM_s003
                                 + 1.06201 w_k1_INTC_s003
                                 + 0.687071 w_k1_KO_s003
                                 + 0.355377 w_k1_MAT_s003
                                 + 0.869748 w_k1_MSFT_s003
                                 + 0.845958 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.01874 w_k1_ADM_s003
                                 + 0.955954 w_k1_AXP_s003
                                 + 1.46978 w_k1_DIS_s003
                                 + 1.09924 w_k1_FDX_s003
                                 + 1.15335 w_k1_IBM_s003
                                 + 1.16956 w_k1_INTC_s003
                                 + 1.15332 w_k1_KO_s003
                                 + 0.636373 w_k1_MAT_s003
                                 + 1.1957 w_k1_MSFT_s003 + 1.27412 w_k1_PG_s003
                                 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.7765 w_k1_ADM_s003
                                 + 1.06676 w_k1_AXP_s003
                                 + 0.985793 w_k1_DIS_s003
                                 + 0.958428 w_k1_FDX_s003
                                 + 1.16335 w_k1_IBM_s003
                                 + 1.00189 w_k1_INTC_s003
                                 + 0.855551 w_k1_KO_s003
                                 + 0.858428 w_k1_MAT_s003
                                 + 1.05719 w_k1_MSFT_s003
                                 + 1.05039 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.04745 w_k1_ADM_s003
                                 + 0.783723 w_k1_AXP_s003
                                 + 0.830012 w_k1_DIS_s003
                                 + 0.977429 w_k1_FDX_s003
                                 + 0.789324 w_k1_IBM_s003
                                 + 1.06353 w_k1_INTC_s003
                                 + 0.746926 w_k1_KO_s003
                                 + 0.414269 w_k1_MAT_s003
                                 + 0.996383 w_k1_MSFT_s003
                                 + 1.06787 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.961939 w_k1_ADM_s003
                                 + 1.03593 w_k1_AXP_s003
                                 + 0.843137 w_k1_DIS_s003
                                 + 0.826317 w_k1_FDX_s003
                                 + 0.992547 w_k1_IBM_s003
                                 + 0.86827 w_k1_INTC_s003
                                 + 0.899753 w_k1_KO_s003
                                 + 0.635335 w_k1_MAT_s003
                                 + 0.779635 w_k1_MSFT_s003
                                 + 0.907798 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.2477 w_k1_ADM_s003
                                 + 1.34587 w_k1_AXP_s003
                                 + 1.40765 w_k1_DIS_s003
                                 + 1.43404 w_k1_FDX_s003
                                 + 1.06343 w_k1_IBM_s003
                                 + 1.19908 w_k1_INTC_s003
                                 + 1.02545 w_k1_KO_s003
                                 + 0.618434 w_k1_MAT_s003
                                 + 1.36361 w_k1_MSFT_s003
                                 + 1.36348 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.30142 w_k1_ADM_s003
                                 + 1.4345 w_k1_AXP_s003 + 1.05198 w_k1_DIS_s003
                                 + 1.13944 w_k1_FDX_s003
                                 + 0.989944 w_k1_IBM_s003
                                 + 1.19922 w_k1_INTC_s003
                                 + 0.924045 w_k1_KO_s003
                                 + 0.797542 w_k1_MAT_s003
                                 + 1.01281 w_k1_MSFT_s003
                                 + 1.10453 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.18241 w_k1_ADM_s004
                                 + 1.57902 w_k1_AXP_s004
                                 + 1.47198 w_k1_DIS_s004
                                 + 1.34016 w_k1_FDX_s004
                                 + 1.24771 w_k1_IBM_s004
                                 + 1.39576 w_k1_INTC_s004
                                 + 1.22238 w_k1_KO_s004 + 1.03217 w_k1_MAT_s004
                                 + 1.19596 w_k1_MSFT_s004
                                 + 1.33881 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.922324 w_k1_ADM_s004
                                 + 0.805256 w_k1_AXP_s004
                                 + 1.0713 w_k1_DIS_s004
                                 + 0.904855 w_k1_FDX_s004
                                 + 1.0124 w_k1_IBM_s004
                                 + 1.22684 w_k1_INTC_s004
                                 + 1.23478 w_k1_KO_s004
                                 + 0.780303 w_k1_MAT_s004
                                 + 1.11567 w_k1_MSFT_s004
                                 + 1.09449 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.16633 w_k1_ADM_s004
                                 + 1.37205 w_k1_AXP_s004
                                 + 1.27763 w_k1_DIS_s004 + 1.0762 w_k1_FDX_s004
                                 + 1.61721 w_k1_IBM_s004
                                 + 1.18932 w_k1_INTC_s004
                                 + 1.29473 w_k1_KO_s004 + 1.55951 w_k1_MAT_s004
                                 + 1.02772 w_k1_MSFT_s004
                                 + 1.14318 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.809006 w_k1_ADM_s004
                                 + 0.898924 w_k1_AXP_s004
                                 + 0.6683 w_k1_DIS_s004
                                 + 0.776989 w_k1_FDX_s004
                                 + 0.80214 w_k1_IBM_s004
                                 + 0.983711 w_k1_INTC_s004
                                 + 0.714084 w_k1_KO_s004
                                 + 0.674778 w_k1_MAT_s004
                                 + 0.847276 w_k1_MSFT_s004
                                 + 0.875716 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.13396 w_k1_ADM_s004
                                 + 1.16681 w_k1_AXP_s004
                                 + 1.14783 w_k1_DIS_s004
                                 + 1.16197 w_k1_FDX_s004
                                 + 0.896825 w_k1_IBM_s004
                                 + 1.42684 w_k1_INTC_s004
                                 + 0.911262 w_k1_KO_s004
                                 + 0.667274 w_k1_MAT_s004
                                 + 1.13828 w_k1_MSFT_s004
                                 + 1.01771 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.07144 w_k1_ADM_s004
                                 + 0.700856 w_k1_AXP_s004
                                 + 0.627655 w_k1_DIS_s004
                                 + 0.874462 w_k1_FDX_s004
                                 + 0.529173 w_k1_IBM_s004
                                 + 0.951338 w_k1_INTC_s004
                                 + 0.64773 w_k1_KO_s004 + 0.43481 w_k1_MAT_s004
                                 + 0.91726 w_k1_MSFT_s004
                                 + 0.779663 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.09179 w_k1_ADM_s004
                                 + 1.11339 w_k1_AXP_s004
                                 + 0.871368 w_k1_DIS_s004
                                 + 0.89082 w_k1_FDX_s004 + 1.3303 w_k1_IBM_s004
                                 + 1.04688 w_k1_INTC_s004
                                 + 1.07705 w_k1_KO_s004 + 1.50832 w_k1_MAT_s004
                                 + 0.99474 w_k1_MSFT_s004
                                 + 0.970805 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.06133 w_k1_ADM_s004
                                 + 0.969086 w_k1_AXP_s004
                                 + 0.973316 w_k1_DIS_s004
                                 + 1.00844 w_k1_FDX_s004
                                 + 0.881975 w_k1_IBM_s004
                                 + 1.07272 w_k1_INTC_s004
                                 + 0.990178 w_k1_KO_s004
                                 + 0.704232 w_k1_MAT_s004
                                 + 1.15296 w_k1_MSFT_s004
                                 + 0.993007 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.40358 w_k1_ADM_s005
                                 + 1.72901 w_k1_AXP_s005
                                 + 1.52124 w_k1_DIS_s005
                                 + 1.32815 w_k1_FDX_s005
                                 + 0.888919 w_k1_IBM_s005
                                 + 1.21995 w_k1_INTC_s005
                                 + 1.09621 w_k1_KO_s005 + 1.10411 w_k1_MAT_s005
                                 + 1.23688 w_k1_MSFT_s005
                                 + 1.26379 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.759343 w_k1_ADM_s005
                                 + 0.727627 w_k1_AXP_s005
                                 + 0.744973 w_k1_DIS_s005
                                 + 0.682294 w_k1_FDX_s005
                                 + 1.06005 w_k1_IBM_s005
                                 + 0.897372 w_k1_INTC_s005
                                 + 0.909097 w_k1_KO_s005
                                 + 1.04765 w_k1_MAT_s005
                                 + 0.866009 w_k1_MSFT_s005
                                 + 0.978306 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.918044 w_k1_ADM_s005
                                 + 1.07302 w_k1_AXP_s005
                                 + 0.898747 w_k1_DIS_s005
                                 + 0.873599 w_k1_FDX_s005
                                 + 1.05616 w_k1_IBM_s005
                                 + 0.869146 w_k1_INTC_s005
                                 + 0.891483 w_k1_KO_s005
                                 + 1.00731 w_k1_MAT_s005
                                 + 0.92162 w_k1_MSFT_s005
                                 + 0.91709 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.944379 w_k1_ADM_s005
                                 + 1.04433 w_k1_AXP_s005
                                 + 1.46483 w_k1_DIS_s005
                                 + 1.11144 w_k1_FDX_s005 + 1.0882 w_k1_IBM_s005
                                 + 1.14564 w_k1_INTC_s005
                                 + 1.31236 w_k1_KO_s005 + 1.23731 w_k1_MAT_s005
                                 + 1.25679 w_k1_MSFT_s005
                                 + 1.13357 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.31325 w_k1_ADM_s005
                                 + 1.23567 w_k1_AXP_s005 + 1.0885 w_k1_DIS_s005
                                 + 1.00458 w_k1_FDX_s005
                                 + 0.97836 w_k1_IBM_s005
                                 + 1.05885 w_k1_INTC_s005
                                 + 0.956397 w_k1_KO_s005
                                 + 0.986614 w_k1_MAT_s005
                                 + 1.03326 w_k1_MSFT_s005 + 1.0723 w_k1_PG_s005
                                 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.863576 w_k1_ADM_s005
                                 + 0.655547 w_k1_AXP_s005
                                 + 0.633582 w_k1_DIS_s005
                                 + 0.72541 w_k1_FDX_s005
                                 + 0.708732 w_k1_IBM_s005
                                 + 0.77628 w_k1_INTC_s005
                                 + 0.579211 w_k1_KO_s005
                                 + 0.569158 w_k1_MAT_s005
                                 + 0.721914 w_k1_MSFT_s005
                                 + 0.783523 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.37054 w_k1_ADM_s005
                                 + 1.40248 w_k1_AXP_s005
                                 + 1.64754 w_k1_DIS_s005
                                 + 1.24046 w_k1_FDX_s005
                                 + 1.60648 w_k1_IBM_s005
                                 + 1.46987 w_k1_INTC_s005
                                 + 1.52209 w_k1_KO_s005 + 1.72474 w_k1_MAT_s005
                                 + 1.33839 w_k1_MSFT_s005
                                 + 1.36113 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.15849 w_k1_ADM_s005
                                 + 1.39361 w_k1_AXP_s005
                                 + 1.09056 w_k1_DIS_s005
                                 + 1.15148 w_k1_FDX_s005
                                 + 0.870839 w_k1_IBM_s005
                                 + 1.06754 w_k1_INTC_s005
                                 + 1.03961 w_k1_KO_s005
                                 + 0.954047 w_k1_MAT_s005
                                 + 1.08678 w_k1_MSFT_s005
                                 + 1.12836 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.00507 w_k1_ADM_s006
                                 + 1.46992 w_k1_AXP_s006
                                 + 1.19655 w_k1_DIS_s006
                                 + 1.22289 w_k1_FDX_s006
                                 + 1.11511 w_k1_IBM_s006
                                 + 1.03889 w_k1_INTC_s006
                                 + 1.12501 w_k1_KO_s006 + 1.25473 w_k1_MAT_s006
                                 + 1.16095 w_k1_MSFT_s006
                                 + 1.15289 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.19393 w_k1_ADM_s006
                                 + 0.771629 w_k1_AXP_s006
                                 + 0.884551 w_k1_DIS_s006
                                 + 1.04502 w_k1_FDX_s006
                                 + 0.649084 w_k1_IBM_s006
                                 + 0.801407 w_k1_INTC_s006
                                 + 0.784159 w_k1_KO_s006
                                 + 0.401897 w_k1_MAT_s006
                                 + 1.00209 w_k1_MSFT_s006
                                 + 0.850543 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.696842 w_k1_ADM_s006
                                 + 0.709323 w_k1_AXP_s006
                                 + 0.650662 w_k1_DIS_s006
                                 + 0.743316 w_k1_FDX_s006
                                 + 0.621967 w_k1_IBM_s006
                                 + 0.684154 w_k1_INTC_s006
                                 + 0.628702 w_k1_KO_s006
                                 + 0.40662 w_k1_MAT_s006
                                 + 0.701119 w_k1_MSFT_s006
                                 + 0.738675 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.867034 w_k1_ADM_s006
                                 + 0.726596 w_k1_AXP_s006
                                 + 0.991781 w_k1_DIS_s006
                                 + 0.939538 w_k1_FDX_s006
                                 + 0.909605 w_k1_IBM_s006
                                 + 0.927338 w_k1_INTC_s006
                                 + 1.01054 w_k1_KO_s006
                                 + 0.663525 w_k1_MAT_s006
                                 + 1.06698 w_k1_MSFT_s006
                                 + 0.935506 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.98949 w_k1_ADM_s006
                                 + 0.706032 w_k1_AXP_s006
                                 + 0.630021 w_k1_DIS_s006
                                 + 0.689455 w_k1_FDX_s006
                                 + 0.890532 w_k1_IBM_s006
                                 + 0.669598 w_k1_INTC_s006
                                 + 0.750545 w_k1_KO_s006
                                 + 0.639566 w_k1_MAT_s006
                                 + 0.776112 w_k1_MSFT_s006
                                 + 0.842065 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.958093 w_k1_ADM_s006
                                 + 1.15362 w_k1_AXP_s006
                                 + 1.04756 w_k1_DIS_s006
                                 + 0.959991 w_k1_FDX_s006
                                 + 1.39122 w_k1_IBM_s006
                                 + 0.859422 w_k1_INTC_s006
                                 + 1.21872 w_k1_KO_s006 + 1.32858 w_k1_MAT_s006
                                 + 1.04644 w_k1_MSFT_s006
                                 + 1.02127 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.10233 w_k1_ADM_s006
                                 + 1.00417 w_k1_AXP_s006
                                 + 1.02956 w_k1_DIS_s006
                                 + 0.915265 w_k1_FDX_s006
                                 + 0.883168 w_k1_IBM_s006
                                 + 0.979762 w_k1_INTC_s006
                                 + 0.935626 w_k1_KO_s006
                                 + 0.696244 w_k1_MAT_s006
                                 + 1.00401 w_k1_MSFT_s006
                                 + 0.897283 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.42592 w_k1_ADM_s006
                                 + 1.23208 w_k1_AXP_s006
                                 + 1.41033 w_k1_DIS_s006
                                 + 1.35192 w_k1_FDX_s006
                                 + 0.816681 w_k1_IBM_s006
                                 + 0.833906 w_k1_INTC_s006
                                 + 1.02843 w_k1_KO_s006
                                 + 0.970029 w_k1_MAT_s006
                                 + 1.18692 w_k1_MSFT_s006
                                 + 1.07304 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.702047 w_k1_ADM_s007
                                 + 0.79354 w_k1_AXP_s007
                                 + 0.745642 w_k1_DIS_s007
                                 + 0.797401 w_k1_FDX_s007
                                 + 0.65104 w_k1_IBM_s007
                                 + 0.968229 w_k1_INTC_s007
                                 + 0.818241 w_k1_KO_s007
                                 + 0.732468 w_k1_MAT_s007
                                 + 0.891325 w_k1_MSFT_s007
                                 + 0.845239 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.80269 w_k1_ADM_s007
                                 + 0.931204 w_k1_AXP_s007
                                 + 1.04881 w_k1_DIS_s007
                                 + 1.08945 w_k1_FDX_s007
                                 + 0.71045 w_k1_IBM_s007
                                 + 1.06926 w_k1_INTC_s007
                                 + 0.849254 w_k1_KO_s007
                                 + 0.634026 w_k1_MAT_s007
                                 + 0.986104 w_k1_MSFT_s007
                                 + 0.861623 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.875445 w_k1_ADM_s007
                                 + 0.919431 w_k1_AXP_s007
                                 + 0.82109 w_k1_DIS_s007
                                 + 0.903744 w_k1_FDX_s007
                                 + 0.918039 w_k1_IBM_s007
                                 + 0.897257 w_k1_INTC_s007
                                 + 0.8903 w_k1_KO_s007 + 1.17745 w_k1_MAT_s007
                                 + 0.832885 w_k1_MSFT_s007
                                 + 0.966705 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.16119 w_k1_ADM_s007
                                 + 1.21696 w_k1_AXP_s007
                                 + 1.42495 w_k1_DIS_s007
                                 + 1.24174 w_k1_FDX_s007
                                 + 1.21733 w_k1_IBM_s007
                                 + 1.17249 w_k1_INTC_s007
                                 + 1.30486 w_k1_KO_s007 + 1.20167 w_k1_MAT_s007
                                 + 1.21089 w_k1_MSFT_s007
                                 + 1.20848 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.775043 w_k1_ADM_s007
                                 + 0.832998 w_k1_AXP_s007
                                 + 1.06291 w_k1_DIS_s007
                                 + 0.894066 w_k1_FDX_s007
                                 + 1.06227 w_k1_IBM_s007
                                 + 0.999282 w_k1_INTC_s007
                                 + 1.06149 w_k1_KO_s007 + 1.09146 w_k1_MAT_s007
                                 + 1.01069 w_k1_MSFT_s007
                                 + 0.932213 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.786301 w_k1_ADM_s007
                                 + 0.840238 w_k1_AXP_s007
                                 + 0.808361 w_k1_DIS_s007
                                 + 0.822169 w_k1_FDX_s007
                                 + 1.20059 w_k1_IBM_s007
                                 + 0.832748 w_k1_INTC_s007
                                 + 0.96963 w_k1_KO_s007 + 1.40916 w_k1_MAT_s007
                                 + 0.788424 w_k1_MSFT_s007
                                 + 0.917117 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.934071 w_k1_ADM_s007
                                 + 1.14248 w_k1_AXP_s007
                                 + 1.17259 w_k1_DIS_s007 + 1.0717 w_k1_FDX_s007
                                 + 0.972425 w_k1_IBM_s007
                                 + 1.03161 w_k1_INTC_s007
                                 + 1.00453 w_k1_KO_s007 + 1.12697 w_k1_MAT_s007
                                 + 1.09819 w_k1_MSFT_s007
                                 + 0.99078 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.873759 w_k1_ADM_s007
                                 + 1.2058 w_k1_AXP_s007 + 1.4626 w_k1_DIS_s007
                                 + 1.03253 w_k1_FDX_s007
                                 + 1.77275 w_k1_IBM_s007
                                 + 1.1917 w_k1_INTC_s007 + 1.58137 w_k1_KO_s007
                                 + 1.95505 w_k1_MAT_s007
                                 + 1.21875 w_k1_MSFT_s007 + 1.1555 w_k1_PG_s007
                                 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.973226 w_k1_ADM_s010
                                 + 0.767947 w_k1_AXP_s010
                                 + 0.643211 w_k1_DIS_s010
                                 + 0.697952 w_k1_FDX_s010
                                 + 0.83943 w_k1_IBM_s010
                                 + 0.876191 w_k1_INTC_s010
                                 + 0.709603 w_k1_KO_s010
                                 + 0.683623 w_k1_MAT_s010
                                 + 0.878762 w_k1_MSFT_s010
                                 + 0.804875 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.864236 w_k1_ADM_s010
                                 + 0.755947 w_k1_AXP_s010
                                 + 0.541396 w_k1_DIS_s010
                                 + 0.745994 w_k1_FDX_s010
                                 + 0.614708 w_k1_IBM_s010
                                 + 0.74621 w_k1_INTC_s010
                                 + 0.595852 w_k1_KO_s010
                                 + 0.51294 w_k1_MAT_s010
                                 + 0.753469 w_k1_MSFT_s010
                                 + 0.763783 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.937693 w_k1_ADM_s010
                                 + 1.03632 w_k1_AXP_s010
                                 + 0.870058 w_k1_DIS_s010
                                 + 0.976615 w_k1_FDX_s010
                                 + 0.871189 w_k1_IBM_s010
                                 + 1.19716 w_k1_INTC_s010
                                 + 0.815193 w_k1_KO_s010
                                 + 0.602418 w_k1_MAT_s010
                                 + 1.0889 w_k1_MSFT_s010
                                 + 0.894785 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.905103 w_k1_ADM_s010
                                 + 0.863437 w_k1_AXP_s010
                                 + 0.864868 w_k1_DIS_s010
                                 + 0.865599 w_k1_FDX_s010
                                 + 1.16568 w_k1_IBM_s010
                                 + 0.906939 w_k1_INTC_s010
                                 + 0.860338 w_k1_KO_s010
                                 + 0.938545 w_k1_MAT_s010
                                 + 0.884613 w_k1_MSFT_s010
                                 + 0.877807 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.55115 w_k1_ADM_s010
                                 + 1.89639 w_k1_AXP_s010
                                 + 1.96977 w_k1_DIS_s010
                                 + 1.52542 w_k1_FDX_s010
                                 + 1.53973 w_k1_IBM_s010
                                 + 1.27134 w_k1_INTC_s010
                                 + 1.65926 w_k1_KO_s010 + 2.45034 w_k1_MAT_s010
                                 + 1.35616 w_k1_MSFT_s010
                                 + 1.30192 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.23963 w_k1_ADM_s010
                                 + 0.954093 w_k1_AXP_s010
                                 + 0.933879 w_k1_DIS_s010
                                 + 0.93738 w_k1_FDX_s010
                                 + 0.686073 w_k1_IBM_s010
                                 + 0.80292 w_k1_INTC_s010
                                 + 0.908896 w_k1_KO_s010
                                 + 0.741762 w_k1_MAT_s010
                                 + 0.973593 w_k1_MSFT_s010
                                 + 0.993465 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.29559 w_k1_ADM_s010
                                 + 1.17187 w_k1_AXP_s010
                                 + 1.17336 w_k1_DIS_s010
                                 + 1.10107 w_k1_FDX_s010
                                 + 0.965704 w_k1_IBM_s010
                                 + 1.02129 w_k1_INTC_s010
                                 + 1.05946 w_k1_KO_s010
                                 + 0.939743 w_k1_MAT_s010
                                 + 1.21695 w_k1_MSFT_s010
                                 + 1.05559 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.31289 w_k1_ADM_s010
                                 + 1.46481 w_k1_AXP_s010
                                 + 1.48715 w_k1_DIS_s010
                                 + 1.21925 w_k1_FDX_s010
                                 + 1.18513 w_k1_IBM_s010
                                 + 1.31851 w_k1_INTC_s010
                                 + 1.39978 w_k1_KO_s010 + 1.40727 w_k1_MAT_s010
                                 + 1.32305 w_k1_MSFT_s010
                                 + 1.26806 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.804847 w_k1_ADM_s011
                                 + 0.690392 w_k1_AXP_s011
                                 + 0.77879 w_k1_DIS_s011
                                 + 0.830198 w_k1_FDX_s011
                                 + 0.61412 w_k1_IBM_s011
                                 + 0.729864 w_k1_INTC_s011
                                 + 0.615835 w_k1_KO_s011
                                 + 0.612145 w_k1_MAT_s011
                                 + 0.941448 w_k1_MSFT_s011
                                 + 0.732294 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.21178 w_k1_ADM_s011
                                 + 0.939504 w_k1_AXP_s011
                                 + 1.23385 w_k1_DIS_s011
                                 + 1.16237 w_k1_FDX_s011
                                 + 0.769769 w_k1_IBM_s011
                                 + 0.822819 w_k1_INTC_s011
                                 + 0.946952 w_k1_KO_s011
                                 + 0.801579 w_k1_MAT_s011
                                 + 1.28499 w_k1_MSFT_s011
                                 + 1.14035 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.938703 w_k1_ADM_s011
                                 + 1.15541 w_k1_AXP_s011 + 1.4718 w_k1_DIS_s011
                                 + 1.14009 w_k1_FDX_s011
                                 + 1.37189 w_k1_IBM_s011
                                 + 0.82224 w_k1_INTC_s011 + 1.1338 w_k1_KO_s011
                                 + 1.54473 w_k1_MAT_s011
                                 + 1.33106 w_k1_MSFT_s011
                                 + 1.04164 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.954393 w_k1_ADM_s011
                                 + 0.877811 w_k1_AXP_s011
                                 + 1.06937 w_k1_DIS_s011
                                 + 0.83448 w_k1_FDX_s011
                                 + 1.73841 w_k1_IBM_s011
                                 + 0.833059 w_k1_INTC_s011
                                 + 1.27469 w_k1_KO_s011 + 1.48014 w_k1_MAT_s011
                                 + 1.0691 w_k1_MSFT_s011 + 1.01153 w_k1_PG_s011
                                 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.803904 w_k1_ADM_s011
                                 + 0.865808 w_k1_AXP_s011
                                 + 0.600967 w_k1_DIS_s011
                                 + 0.775861 w_k1_FDX_s011
                                 + 0.845749 w_k1_IBM_s011
                                 + 0.631742 w_k1_INTC_s011
                                 + 0.655432 w_k1_KO_s011
                                 + 0.98471 w_k1_MAT_s011
                                 + 0.83772 w_k1_MSFT_s011
                                 + 0.889298 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.27989 w_k1_ADM_s011
                                 + 1.45914 w_k1_AXP_s011
                                 + 1.85901 w_k1_DIS_s011
                                 + 1.43721 w_k1_FDX_s011 + 1.0836 w_k1_IBM_s011
                                 + 1.20793 w_k1_INTC_s011
                                 + 1.01551 w_k1_KO_s011
                                 + 0.974967 w_k1_MAT_s011
                                 + 1.58275 w_k1_MSFT_s011 + 1.1299 w_k1_PG_s011
                                 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.766289 w_k1_ADM_s011
                                 + 0.714514 w_k1_AXP_s011
                                 + 0.632417 w_k1_DIS_s011
                                 + 0.694647 w_k1_FDX_s011
                                 + 1.03981 w_k1_IBM_s011
                                 + 0.703909 w_k1_INTC_s011
                                 + 0.657089 w_k1_KO_s011
                                 + 0.861954 w_k1_MAT_s011
                                 + 0.833143 w_k1_MSFT_s011
                                 + 0.740326 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.07185 w_k1_ADM_s011
                                 + 1.01127 w_k1_AXP_s011
                                 + 0.974284 w_k1_DIS_s011
                                 + 0.941567 w_k1_FDX_s011
                                 + 1.01381 w_k1_IBM_s011
                                 + 0.876653 w_k1_INTC_s011
                                 + 0.853292 w_k1_KO_s011
                                 + 1.32656 w_k1_MAT_s011
                                 + 1.11309 w_k1_MSFT_s011
                                 + 0.876384 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.29377 w_k1_ADM_s012
                                 + 1.36716 w_k1_AXP_s012
                                 + 1.24088 w_k1_DIS_s012
                                 + 1.25668 w_k1_FDX_s012
                                 + 0.883562 w_k1_IBM_s012
                                 + 1.1737 w_k1_INTC_s012 + 1.13467 w_k1_KO_s012
                                 + 1.01214 w_k1_MAT_s012
                                 + 1.06035 w_k1_MSFT_s012 + 1.1215 w_k1_PG_s012
                                 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.970998 w_k1_ADM_s012
                                 + 1.12085 w_k1_AXP_s012
                                 + 0.763588 w_k1_DIS_s012
                                 + 0.795256 w_k1_FDX_s012
                                 + 1.32625 w_k1_IBM_s012
                                 + 0.888818 w_k1_INTC_s012
                                 + 0.985029 w_k1_KO_s012 + 1.5182 w_k1_MAT_s012
                                 + 0.822401 w_k1_MSFT_s012
                                 + 0.867273 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.876494 w_k1_ADM_s012
                                 + 0.99244 w_k1_AXP_s012
                                 + 0.873788 w_k1_DIS_s012
                                 + 1.00789 w_k1_FDX_s012
                                 + 0.934167 w_k1_IBM_s012
                                 + 0.937835 w_k1_INTC_s012
                                 + 0.93724 w_k1_KO_s012 + 1.17526 w_k1_MAT_s012
                                 + 0.914263 w_k1_MSFT_s012
                                 + 0.986036 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.82887 w_k1_ADM_s012
                                 + 0.688472 w_k1_AXP_s012
                                 + 0.703636 w_k1_DIS_s012
                                 + 0.769637 w_k1_FDX_s012
                                 + 0.796143 w_k1_IBM_s012
                                 + 0.628696 w_k1_INTC_s012
                                 + 0.752226 w_k1_KO_s012
                                 + 0.897685 w_k1_MAT_s012
                                 + 0.666367 w_k1_MSFT_s012
                                 + 0.777739 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.979264 w_k1_ADM_s012
                                 + 0.901118 w_k1_AXP_s012
                                 + 1.15647 w_k1_DIS_s012
                                 + 1.04785 w_k1_FDX_s012
                                 + 0.957696 w_k1_IBM_s012
                                 + 0.8382 w_k1_INTC_s012
                                 + 0.997105 w_k1_KO_s012
                                 + 1.17729 w_k1_MAT_s012
                                 + 0.94474 w_k1_MSFT_s012
                                 + 1.05444 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.02647 w_k1_ADM_s012
                                 + 0.885714 w_k1_AXP_s012
                                 + 1.19835 w_k1_DIS_s012
                                 + 0.961549 w_k1_FDX_s012
                                 + 1.35458 w_k1_IBM_s012
                                 + 1.06487 w_k1_INTC_s012
                                 + 1.12522 w_k1_KO_s012 + 1.13092 w_k1_MAT_s012
                                 + 0.952039 w_k1_MSFT_s012
                                 + 1.01226 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.09958 w_k1_ADM_s012
                                 + 1.4199 w_k1_AXP_s012 + 1.29568 w_k1_DIS_s012
                                 + 1.27932 w_k1_FDX_s012
                                 + 1.51881 w_k1_IBM_s012
                                 + 1.06982 w_k1_INTC_s012
                                 + 1.16661 w_k1_KO_s012 + 1.76542 w_k1_MAT_s012
                                 + 1.0607 w_k1_MSFT_s012 + 1.24592 w_k1_PG_s012
                                 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.826531 w_k1_ADM_s012
                                 + 1.00781 w_k1_AXP_s012
                                 + 0.807716 w_k1_DIS_s012
                                 + 0.81082 w_k1_FDX_s012
                                 + 1.93622 w_k1_IBM_s012
                                 + 0.968083 w_k1_INTC_s012
                                 + 1.44217 w_k1_KO_s012 + 1.8025 w_k1_MAT_s012
                                 + 0.832739 w_k1_MSFT_s012
                                 + 0.989433 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.82293 w_k1_ADM_s013
                                 + 0.72882 w_k1_AXP_s013
                                 + 0.591453 w_k1_DIS_s013
                                 + 0.842278 w_k1_FDX_s013
                                 + 0.643525 w_k1_IBM_s013
                                 + 0.908223 w_k1_INTC_s013
                                 + 0.722248 w_k1_KO_s013
                                 + 0.569364 w_k1_MAT_s013
                                 + 0.760857 w_k1_MSFT_s013
                                 + 0.760804 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.704504 w_k1_ADM_s013
                                 + 0.736088 w_k1_AXP_s013
                                 + 0.648103 w_k1_DIS_s013
                                 + 0.765583 w_k1_FDX_s013
                                 + 0.849426 w_k1_IBM_s013
                                 + 0.83696 w_k1_INTC_s013
                                 + 0.913824 w_k1_KO_s013
                                 + 0.673023 w_k1_MAT_s013
                                 + 0.808952 w_k1_MSFT_s013
                                 + 0.835981 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.730264 w_k1_ADM_s013
                                 + 0.696399 w_k1_AXP_s013
                                 + 0.867236 w_k1_DIS_s013
                                 + 0.810909 w_k1_FDX_s013
                                 + 1.24394 w_k1_IBM_s013
                                 + 0.973707 w_k1_INTC_s013
                                 + 1.29859 w_k1_KO_s013 + 1.07907 w_k1_MAT_s013
                                 + 0.989766 w_k1_MSFT_s013
                                 + 0.967605 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.810322 w_k1_ADM_s013
                                 + 0.911794 w_k1_AXP_s013
                                 + 0.936258 w_k1_DIS_s013
                                 + 0.959236 w_k1_FDX_s013
                                 + 0.888615 w_k1_IBM_s013
                                 + 1.16572 w_k1_INTC_s013
                                 + 0.912692 w_k1_KO_s013
                                 + 0.776206 w_k1_MAT_s013
                                 + 1.05036 w_k1_MSFT_s013
                                 + 0.894684 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.645277 w_k1_ADM_s013
                                 + 0.504793 w_k1_AXP_s013
                                 + 0.534636 w_k1_DIS_s013
                                 + 0.650846 w_k1_FDX_s013
                                 + 0.735561 w_k1_IBM_s013
                                 + 0.717466 w_k1_INTC_s013
                                 + 0.793723 w_k1_KO_s013
                                 + 0.59245 w_k1_MAT_s013
                                 + 0.603201 w_k1_MSFT_s013
                                 + 0.734263 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.20751 w_k1_ADM_s013
                                 + 1.3251 w_k1_AXP_s013 + 1.16338 w_k1_DIS_s013
                                 + 1.22179 w_k1_FDX_s013
                                 + 0.99716 w_k1_IBM_s013
                                 + 1.05568 w_k1_INTC_s013
                                 + 1.16681 w_k1_KO_s013
                                 + 0.998291 w_k1_MAT_s013
                                 + 1.03098 w_k1_MSFT_s013
                                 + 1.11666 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.687699 w_k1_ADM_s013
                                 + 1.12266 w_k1_AXP_s013
                                 + 0.869088 w_k1_DIS_s013
                                 + 0.940084 w_k1_FDX_s013
                                 + 1.11645 w_k1_IBM_s013
                                 + 1.16742 w_k1_INTC_s013
                                 + 0.99853 w_k1_KO_s013 + 1.33053 w_k1_MAT_s013
                                 + 0.933172 w_k1_MSFT_s013
                                 + 0.953222 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.19798 w_k1_ADM_s013
                                 + 1.40287 w_k1_AXP_s013
                                 + 1.30058 w_k1_DIS_s013
                                 + 1.15281 w_k1_FDX_s013
                                 + 1.39217 w_k1_IBM_s013
                                 + 1.24689 w_k1_INTC_s013
                                 + 1.24295 w_k1_KO_s013 + 1.30006 w_k1_MAT_s013
                                 + 1.06797 w_k1_MSFT_s013
                                 + 1.10866 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.889618 w_k1_ADM_s014
                                 + 0.913697 w_k1_AXP_s014
                                 + 0.993983 w_k1_DIS_s014
                                 + 0.907387 w_k1_FDX_s014
                                 + 1.24242 w_k1_IBM_s014
                                 + 0.864932 w_k1_INTC_s014
                                 + 0.973548 w_k1_KO_s014
                                 + 1.48535 w_k1_MAT_s014
                                 + 1.06025 w_k1_MSFT_s014
                                 + 1.07842 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.85313 w_k1_ADM_s014
                                 + 0.737886 w_k1_AXP_s014
                                 + 0.810933 w_k1_DIS_s014
                                 + 0.781961 w_k1_FDX_s014
                                 + 0.92649 w_k1_IBM_s014
                                 + 0.700062 w_k1_INTC_s014
                                 + 0.685066 w_k1_KO_s014
                                 + 0.99953 w_k1_MAT_s014
                                 + 0.813194 w_k1_MSFT_s014
                                 + 0.769846 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.17351 w_k1_ADM_s014
                                 + 1.44718 w_k1_AXP_s014
                                 + 0.89205 w_k1_DIS_s014
                                 + 1.09354 w_k1_FDX_s014
                                 + 0.915829 w_k1_IBM_s014
                                 + 0.923192 w_k1_INTC_s014
                                 + 0.777379 w_k1_KO_s014
                                 + 0.982549 w_k1_MAT_s014
                                 + 0.93138 w_k1_MSFT_s014
                                 + 0.906283 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.62046 w_k1_ADM_s014
                                 + 1.53071 w_k1_AXP_s014
                                 + 1.63214 w_k1_DIS_s014
                                 + 1.42423 w_k1_FDX_s014
                                 + 1.38138 w_k1_IBM_s014
                                 + 1.32931 w_k1_INTC_s014
                                 + 1.33353 w_k1_KO_s014 + 1.30787 w_k1_MAT_s014
                                 + 1.45925 w_k1_MSFT_s014
                                 + 1.31063 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.970356 w_k1_ADM_s014
                                 + 0.683979 w_k1_AXP_s014
                                 + 0.678236 w_k1_DIS_s014
                                 + 0.810633 w_k1_FDX_s014
                                 + 0.672727 w_k1_IBM_s014
                                 + 0.72864 w_k1_INTC_s014
                                 + 0.661735 w_k1_KO_s014
                                 + 0.569581 w_k1_MAT_s014
                                 + 0.811741 w_k1_MSFT_s014
                                 + 0.775001 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.22121 w_k1_ADM_s014
                                 + 0.91873 w_k1_AXP_s014
                                 + 1.31541 w_k1_DIS_s014
                                 + 1.04595 w_k1_FDX_s014
                                 + 0.863593 w_k1_IBM_s014
                                 + 1.01552 w_k1_INTC_s014
                                 + 0.981434 w_k1_KO_s014
                                 + 0.863652 w_k1_MAT_s014
                                 + 1.23934 w_k1_MSFT_s014
                                 + 0.992654 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.24695 w_k1_ADM_s014
                                 + 1.1412 w_k1_AXP_s014 + 1.27018 w_k1_DIS_s014
                                 + 1.05784 w_k1_FDX_s014
                                 + 1.47956 w_k1_IBM_s014
                                 + 1.13865 w_k1_INTC_s014
                                 + 1.32105 w_k1_KO_s014 + 1.33705 w_k1_MAT_s014
                                 + 1.20475 w_k1_MSFT_s014
                                 + 1.00469 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.12148 w_k1_ADM_s014
                                 + 1.06802 w_k1_AXP_s014 + 1.2576 w_k1_DIS_s014
                                 + 1.17254 w_k1_FDX_s014 + 1.0221 w_k1_IBM_s014
                                 + 1.14282 w_k1_INTC_s014
                                 + 1.01315 w_k1_KO_s014
                                 + 0.762725 w_k1_MAT_s014
                                 + 1.13131 w_k1_MSFT_s014
                                 + 0.953941 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.860843 w_k1_ADM_s015
                                 + 0.898552 w_k1_AXP_s015
                                 + 0.898907 w_k1_DIS_s015
                                 + 0.920702 w_k1_FDX_s015
                                 + 0.704527 w_k1_IBM_s015
                                 + 1.01499 w_k1_INTC_s015
                                 + 0.796698 w_k1_KO_s015
                                 + 0.822887 w_k1_MAT_s015
                                 + 1.00315 w_k1_MSFT_s015
                                 + 1.00711 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.781177 w_k1_ADM_s015
                                 + 1.15012 w_k1_AXP_s015
                                 + 1.49137 w_k1_DIS_s015
                                 + 1.12114 w_k1_FDX_s015 + 1.746 w_k1_IBM_s015
                                 + 1.1083 w_k1_INTC_s015 + 1.52797 w_k1_KO_s015
                                 + 2.00041 w_k1_MAT_s015
                                 + 1.36679 w_k1_MSFT_s015
                                 + 1.18783 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.931898 w_k1_ADM_s015
                                 + 0.631678 w_k1_AXP_s015
                                 + 0.643864 w_k1_DIS_s015
                                 + 0.705982 w_k1_FDX_s015
                                 + 0.757124 w_k1_IBM_s015
                                 + 0.694733 w_k1_INTC_s015
                                 + 0.769556 w_k1_KO_s015
                                 + 0.618409 w_k1_MAT_s015
                                 + 0.776728 w_k1_MSFT_s015
                                 + 0.749774 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.948991 w_k1_ADM_s015
                                 + 0.894008 w_k1_AXP_s015
                                 + 0.935881 w_k1_DIS_s015
                                 + 0.978786 w_k1_FDX_s015
                                 + 0.968266 w_k1_IBM_s015
                                 + 0.943432 w_k1_INTC_s015
                                 + 0.964687 w_k1_KO_s015
                                 + 1.23603 w_k1_MAT_s015
                                 + 1.03315 w_k1_MSFT_s015 + 1.0293 w_k1_PG_s015
                                 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.785235 w_k1_ADM_s015
                                 + 1.04393 w_k1_AXP_s015
                                 + 0.933978 w_k1_DIS_s015
                                 + 0.890678 w_k1_FDX_s015
                                 + 1.28636 w_k1_IBM_s015
                                 + 0.810037 w_k1_INTC_s015
                                 + 0.942021 w_k1_KO_s015 + 1.4627 w_k1_MAT_s015
                                 + 0.892786 w_k1_MSFT_s015
                                 + 1.1561 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.31074 w_k1_ADM_s015
                                 + 1.33883 w_k1_AXP_s015
                                 + 1.45419 w_k1_DIS_s015
                                 + 1.26459 w_k1_FDX_s015
                                 + 1.14771 w_k1_IBM_s015
                                 + 1.04748 w_k1_INTC_s015
                                 + 1.13432 w_k1_KO_s015 + 1.0385 w_k1_MAT_s015
                                 + 1.15157 w_k1_MSFT_s015
                                 + 1.07801 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.01259 w_k1_ADM_s015
                                 + 0.718775 w_k1_AXP_s015
                                 + 0.851868 w_k1_DIS_s015
                                 + 0.796058 w_k1_FDX_s015
                                 + 1.07427 w_k1_IBM_s015
                                 + 0.655536 w_k1_INTC_s015
                                 + 1.00414 w_k1_KO_s015 + 1.04546 w_k1_MAT_s015
                                 + 0.830653 w_k1_MSFT_s015
                                 + 1.07892 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.26543 w_k1_ADM_s015
                                 + 1.24297 w_k1_AXP_s015 + 1.071 w_k1_DIS_s015
                                 + 1.2255 w_k1_FDX_s015
                                 + 0.858088 w_k1_IBM_s015
                                 + 1.16588 w_k1_INTC_s015
                                 + 0.958646 w_k1_KO_s015
                                 + 0.919649 w_k1_MAT_s015
                                 + 1.11168 w_k1_MSFT_s015 + 1.173 w_k1_PG_s015
                                 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.08967 w_k1_ADM_s016
                                 + 0.971967 w_k1_AXP_s016
                                 + 1.07083 w_k1_DIS_s016
                                 + 0.993231 w_k1_FDX_s016
                                 + 1.39865 w_k1_IBM_s016
                                 + 1.09855 w_k1_INTC_s016
                                 + 1.12679 w_k1_KO_s016 + 1.5502 w_k1_MAT_s016
                                 + 1.02926 w_k1_MSFT_s016
                                 + 1.21305 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.23536 w_k1_ADM_s016
                                 + 1.89247 w_k1_AXP_s016
                                 + 1.66546 w_k1_DIS_s016
                                 + 1.46715 w_k1_FDX_s016
                                 + 1.49594 w_k1_IBM_s016
                                 + 1.63302 w_k1_INTC_s016
                                 + 1.35955 w_k1_KO_s016 + 1.77442 w_k1_MAT_s016
                                 + 1.30961 w_k1_MSFT_s016
                                 + 1.27455 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.851924 w_k1_ADM_s016
                                 + 0.807478 w_k1_AXP_s016
                                 + 0.863838 w_k1_DIS_s016
                                 + 0.946419 w_k1_FDX_s016
                                 + 0.766051 w_k1_IBM_s016
                                 + 0.827646 w_k1_INTC_s016
                                 + 0.797269 w_k1_KO_s016
                                 + 0.942832 w_k1_MAT_s016
                                 + 0.923679 w_k1_MSFT_s016
                                 + 0.983707 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.705136 w_k1_ADM_s016
                                 + 0.930836 w_k1_AXP_s016
                                 + 0.691642 w_k1_DIS_s016
                                 + 0.820985 w_k1_FDX_s016
                                 + 1.09199 w_k1_IBM_s016
                                 + 0.825601 w_k1_INTC_s016
                                 + 0.806485 w_k1_KO_s016 + 1.2238 w_k1_MAT_s016
                                 + 0.776621 w_k1_MSFT_s016
                                 + 0.918344 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.825329 w_k1_ADM_s016
                                 + 0.630566 w_k1_AXP_s016
                                 + 0.62691 w_k1_DIS_s016
                                 + 0.770708 w_k1_FDX_s016
                                 + 0.738847 w_k1_IBM_s016
                                 + 0.673652 w_k1_INTC_s016
                                 + 0.755582 w_k1_KO_s016
                                 + 0.544704 w_k1_MAT_s016
                                 + 0.698109 w_k1_MSFT_s016
                                 + 0.838923 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.03395 w_k1_ADM_s016
                                 + 1.33007 w_k1_AXP_s016
                                 + 1.29813 w_k1_DIS_s016 + 1.3631 w_k1_FDX_s016
                                 + 1.06208 w_k1_IBM_s016
                                 + 1.54911 w_k1_INTC_s016
                                 + 1.12186 w_k1_KO_s016 + 1.03608 w_k1_MAT_s016
                                 + 1.24792 w_k1_MSFT_s016
                                 + 1.29701 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.06358 w_k1_ADM_s016
                                 + 0.755048 w_k1_AXP_s016
                                 + 0.850402 w_k1_DIS_s016
                                 + 0.986002 w_k1_FDX_s016
                                 + 0.963217 w_k1_IBM_s016
                                 + 1.08806 w_k1_INTC_s016
                                 + 0.999983 w_k1_KO_s016 + 1.0558 w_k1_MAT_s016
                                 + 1.04875 w_k1_MSFT_s016
                                 + 1.06594 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.993609 w_k1_ADM_s016
                                 + 1.27417 w_k1_AXP_s016
                                 + 0.970412 w_k1_DIS_s016
                                 + 1.17282 w_k1_FDX_s016 + 1.0167 w_k1_IBM_s016
                                 + 1.0056 w_k1_INTC_s016
                                 + 0.931387 w_k1_KO_s016
                                 + 1.43427 w_k1_MAT_s016
                                 + 0.959635 w_k1_MSFT_s016
                                 + 1.05629 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.19219 w_k1_ADM_s017
                                 + 0.701178 w_k1_AXP_s017
                                 + 0.83701 w_k1_DIS_s017
                                 + 0.815443 w_k1_FDX_s017
                                 + 0.928787 w_k1_IBM_s017
                                 + 1.00314 w_k1_INTC_s017
                                 + 0.953262 w_k1_KO_s017
                                 + 0.687209 w_k1_MAT_s017
                                 + 0.770968 w_k1_MSFT_s017
                                 + 0.822868 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.05627 w_k1_ADM_s017
                                 + 1.45145 w_k1_AXP_s017 + 1.2591 w_k1_DIS_s017
                                 + 1.25135 w_k1_FDX_s017
                                 + 0.960113 w_k1_IBM_s017
                                 + 1.22992 w_k1_INTC_s017 + 1.1389 w_k1_KO_s017
                                 + 1.25451 w_k1_MAT_s017
                                 + 1.18456 w_k1_MSFT_s017
                                 + 1.05399 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.28895 w_k1_ADM_s017
                                 + 0.758255 w_k1_AXP_s017
                                 + 0.835786 w_k1_DIS_s017
                                 + 0.969396 w_k1_FDX_s017
                                 + 0.617551 w_k1_IBM_s017
                                 + 0.988706 w_k1_INTC_s017
                                 + 0.818179 w_k1_KO_s017
                                 + 0.538869 w_k1_MAT_s017
                                 + 1.0907 w_k1_MSFT_s017
                                 + 0.947609 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.00936 w_k1_ADM_s017
                                 + 0.787628 w_k1_AXP_s017
                                 + 0.723525 w_k1_DIS_s017
                                 + 0.750698 w_k1_FDX_s017
                                 + 1.16493 w_k1_IBM_s017
                                 + 0.739609 w_k1_INTC_s017
                                 + 0.918764 w_k1_KO_s017
                                 + 1.18164 w_k1_MAT_s017
                                 + 0.856631 w_k1_MSFT_s017
                                 + 0.870163 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.18128 w_k1_ADM_s017
                                 + 1.06982 w_k1_AXP_s017
                                 + 1.48945 w_k1_DIS_s017
                                 + 1.21463 w_k1_FDX_s017
                                 + 0.990701 w_k1_IBM_s017
                                 + 1.59069 w_k1_INTC_s017
                                 + 1.17187 w_k1_KO_s017
                                 + 0.943756 w_k1_MAT_s017
                                 + 1.32167 w_k1_MSFT_s017
                                 + 1.10153 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.07635 w_k1_ADM_s017
                                 + 0.784914 w_k1_AXP_s017
                                 + 0.652396 w_k1_DIS_s017
                                 + 0.722471 w_k1_FDX_s017
                                 + 0.726659 w_k1_IBM_s017
                                 + 0.791162 w_k1_INTC_s017
                                 + 0.66176 w_k1_KO_s017
                                 + 0.571529 w_k1_MAT_s017
                                 + 0.725746 w_k1_MSFT_s017
                                 + 0.739011 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.05665 w_k1_ADM_s017
                                 + 0.968808 w_k1_AXP_s017
                                 + 1.20684 w_k1_DIS_s017
                                 + 1.01356 w_k1_FDX_s017
                                 + 1.40627 w_k1_IBM_s017
                                 + 1.16262 w_k1_INTC_s017
                                 + 1.13585 w_k1_KO_s017 + 1.4587 w_k1_MAT_s017
                                 + 0.948462 w_k1_MSFT_s017
                                 + 1.06199 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.03172 w_k1_ADM_s017
                                 + 0.981095 w_k1_AXP_s017
                                 + 0.867348 w_k1_DIS_s017
                                 + 0.945502 w_k1_FDX_s017
                                 + 0.847587 w_k1_IBM_s017
                                 + 1.17424 w_k1_INTC_s017
                                 + 0.854647 w_k1_KO_s017
                                 + 0.709725 w_k1_MAT_s017
                                 + 0.855736 w_k1_MSFT_s017
                                 + 0.889338 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.07278 w_k1_ADM_s100
                                 + 0.8489 w_k1_AXP_s100
                                 + 0.948956 w_k1_DIS_s100
                                 + 0.957475 w_k1_FDX_s100
                                 + 0.791543 w_k1_IBM_s100
                                 + 1.09933 w_k1_INTC_s100
                                 + 1.10971 w_k1_KO_s100
                                 + 0.956196 w_k1_MAT_s100
                                 + 1.06573 w_k1_MSFT_s100
                                 + 1.13804 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.866428 w_k1_ADM_s100
                                 + 0.888159 w_k1_AXP_s100
                                 + 0.755015 w_k1_DIS_s100
                                 + 0.860282 w_k1_FDX_s100
                                 + 1.03057 w_k1_IBM_s100
                                 + 1.17157 w_k1_INTC_s100
                                 + 0.937029 w_k1_KO_s100
                                 + 0.952309 w_k1_MAT_s100
                                 + 0.957157 w_k1_MSFT_s100
                                 + 1.02626 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.894115 w_k1_ADM_s100
                                 + 0.69583 w_k1_AXP_s100
                                 + 0.829773 w_k1_DIS_s100
                                 + 0.850848 w_k1_FDX_s100
                                 + 0.734249 w_k1_IBM_s100
                                 + 0.763493 w_k1_INTC_s100
                                 + 0.827738 w_k1_KO_s100
                                 + 0.647794 w_k1_MAT_s100
                                 + 0.814806 w_k1_MSFT_s100
                                 + 0.924697 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.22485 w_k1_ADM_s100
                                 + 1.0565 w_k1_AXP_s100 + 1.17986 w_k1_DIS_s100
                                 + 1.19719 w_k1_FDX_s100
                                 + 0.771255 w_k1_IBM_s100
                                 + 1.05821 w_k1_INTC_s100
                                 + 0.944312 w_k1_KO_s100
                                 + 1.01222 w_k1_MAT_s100
                                 + 1.02718 w_k1_MSFT_s100
                                 + 1.09172 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.30436 w_k1_ADM_s100
                                 + 1.42049 w_k1_AXP_s100
                                 + 1.27887 w_k1_DIS_s100
                                 + 1.16593 w_k1_FDX_s100
                                 + 1.48456 w_k1_IBM_s100
                                 + 0.994497 w_k1_INTC_s100
                                 + 1.49018 w_k1_KO_s100 + 1.90186 w_k1_MAT_s100
                                 + 1.09648 w_k1_MSFT_s100
                                 + 1.08283 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.846754 w_k1_ADM_s100
                                 + 0.743448 w_k1_AXP_s100
                                 + 0.697679 w_k1_DIS_s100
                                 + 0.873403 w_k1_FDX_s100
                                 + 0.56802 w_k1_IBM_s100
                                 + 1.06016 w_k1_INTC_s100
                                 + 0.661516 w_k1_KO_s100
                                 + 0.499632 w_k1_MAT_s100
                                 + 0.84448 w_k1_MSFT_s100
                                 + 0.885739 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.08903 w_k1_ADM_s100
                                 + 1.32589 w_k1_AXP_s100
                                 + 0.817667 w_k1_DIS_s100
                                 + 1.02499 w_k1_FDX_s100
                                 + 0.952587 w_k1_IBM_s100
                                 + 0.833148 w_k1_INTC_s100
                                 + 0.896349 w_k1_KO_s100
                                 + 1.51708 w_k1_MAT_s100
                                 + 0.893929 w_k1_MSFT_s100
                                 + 1.06426 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.16763 w_k1_ADM_s100
                                 + 1.15439 w_k1_AXP_s100
                                 + 1.27593 w_k1_DIS_s100
                                 + 1.12959 w_k1_FDX_s100 + 1.1257 w_k1_IBM_s100
                                 + 1.23539 w_k1_INTC_s100
                                 + 1.17743 w_k1_KO_s100 + 1.05136 w_k1_MAT_s100
                                 + 1.07572 w_k1_MSFT_s100 + 1.059 w_k1_PG_s100
                                 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.873692 w_k1_ADM_s101
                                 + 0.681107 w_k1_AXP_s101
                                 + 0.705162 w_k1_DIS_s101
                                 + 0.780131 w_k1_FDX_s101
                                 + 0.843147 w_k1_IBM_s101
                                 + 0.909967 w_k1_INTC_s101
                                 + 0.7938 w_k1_KO_s101 + 0.516492 w_k1_MAT_s101
                                 + 0.879869 w_k1_MSFT_s101
                                 + 0.902317 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.06784 w_k1_ADM_s101
                                 + 1.46103 w_k1_AXP_s101
                                 + 1.47851 w_k1_DIS_s101
                                 + 1.21361 w_k1_FDX_s101
                                 + 1.69293 w_k1_IBM_s101
                                 + 1.55961 w_k1_INTC_s101
                                 + 1.60427 w_k1_KO_s101 + 1.44335 w_k1_MAT_s101
                                 + 1.32142 w_k1_MSFT_s101
                                 + 1.11713 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.01183 w_k1_ADM_s101
                                 + 0.92504 w_k1_AXP_s101
                                 + 0.895247 w_k1_DIS_s101
                                 + 0.919927 w_k1_FDX_s101
                                 + 0.984985 w_k1_IBM_s101
                                 + 1.25577 w_k1_INTC_s101
                                 + 1.05019 w_k1_KO_s101
                                 + 0.824983 w_k1_MAT_s101
                                 + 0.956401 w_k1_MSFT_s101
                                 + 1.00402 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.76233 w_k1_ADM_s101
                                 + 1.53394 w_k1_AXP_s101
                                 + 1.46691 w_k1_DIS_s101
                                 + 1.39924 w_k1_FDX_s101
                                 + 1.33667 w_k1_IBM_s101
                                 + 1.25629 w_k1_INTC_s101
                                 + 1.26691 w_k1_KO_s101 + 1.01155 w_k1_MAT_s101
                                 + 1.30325 w_k1_MSFT_s101
                                 + 1.22477 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.866132 w_k1_ADM_s101
                                 + 0.657314 w_k1_AXP_s101
                                 + 0.635141 w_k1_DIS_s101
                                 + 0.798405 w_k1_FDX_s101
                                 + 0.602177 w_k1_IBM_s101
                                 + 0.989923 w_k1_INTC_s101
                                 + 0.813835 w_k1_KO_s101
                                 + 0.488369 w_k1_MAT_s101
                                 + 0.833466 w_k1_MSFT_s101
                                 + 0.835841 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.22761 w_k1_ADM_s101
                                 + 0.9301 w_k1_AXP_s101 + 1.08595 w_k1_DIS_s101
                                 + 1.1579 w_k1_FDX_s101
                                 + 0.852245 w_k1_IBM_s101
                                 + 1.15407 w_k1_INTC_s101
                                 + 1.02528 w_k1_KO_s101
                                 + 0.765168 w_k1_MAT_s101
                                 + 1.21593 w_k1_MSFT_s101
                                 + 1.11466 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.35253 w_k1_ADM_s101
                                 + 1.70649 w_k1_AXP_s101
                                 + 1.28034 w_k1_DIS_s101
                                 + 1.19545 w_k1_FDX_s101
                                 + 1.28221 w_k1_IBM_s101
                                 + 1.52678 w_k1_INTC_s101
                                 + 1.30342 w_k1_KO_s101 + 1.19287 w_k1_MAT_s101
                                 + 1.08786 w_k1_MSFT_s101 + 1.1142 w_k1_PG_s101
                                 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.10959 w_k1_ADM_s101
                                 + 1.06387 w_k1_AXP_s101
                                 + 0.918206 w_k1_DIS_s101
                                 + 0.894602 w_k1_FDX_s101
                                 + 1.45247 w_k1_IBM_s101
                                 + 1.06305 w_k1_INTC_s101
                                 + 1.16903 w_k1_KO_s101 + 1.09341 w_k1_MAT_s101
                                 + 1.14668 w_k1_MSFT_s101
                                 + 1.08964 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.79243 w_k1_ADM_s102
                                 + 1.59086 w_k1_AXP_s102 + 1.688 w_k1_DIS_s102
                                 + 1.47015 w_k1_FDX_s102
                                 + 1.11242 w_k1_IBM_s102
                                 + 1.31494 w_k1_INTC_s102
                                 + 1.23855 w_k1_KO_s102 + 1.12056 w_k1_MAT_s102
                                 + 1.34923 w_k1_MSFT_s102
                                 + 1.15672 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.840486 w_k1_ADM_s102
                                 + 0.759636 w_k1_AXP_s102
                                 + 0.740393 w_k1_DIS_s102
                                 + 0.941414 w_k1_FDX_s102
                                 + 0.675798 w_k1_IBM_s102
                                 + 0.98385 w_k1_INTC_s102
                                 + 0.725831 w_k1_KO_s102
                                 + 0.652729 w_k1_MAT_s102
                                 + 0.895106 w_k1_MSFT_s102
                                 + 0.804631 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.08186 w_k1_ADM_s102
                                 + 1.2752 w_k1_AXP_s102 + 1.20678 w_k1_DIS_s102
                                 + 1.03367 w_k1_FDX_s102
                                 + 1.47175 w_k1_IBM_s102
                                 + 0.964171 w_k1_INTC_s102
                                 + 1.16228 w_k1_KO_s102 + 1.75004 w_k1_MAT_s102
                                 + 1.04082 w_k1_MSFT_s102
                                 + 0.921404 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.41919 w_k1_ADM_s102
                                 + 1.65967 w_k1_AXP_s102
                                 + 1.25815 w_k1_DIS_s102
                                 + 1.27414 w_k1_FDX_s102
                                 + 1.12965 w_k1_IBM_s102
                                 + 1.39588 w_k1_INTC_s102
                                 + 1.27096 w_k1_KO_s102 + 1.54402 w_k1_MAT_s102
                                 + 1.29205 w_k1_MSFT_s102
                                 + 1.04986 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.882347 w_k1_ADM_s102
                                 + 1.13009 w_k1_AXP_s102
                                 + 0.880397 w_k1_DIS_s102
                                 + 0.843234 w_k1_FDX_s102
                                 + 1.33967 w_k1_IBM_s102
                                 + 0.776325 w_k1_INTC_s102
                                 + 0.926929 w_k1_KO_s102 + 1.3115 w_k1_MAT_s102
                                 + 0.821874 w_k1_MSFT_s102
                                 + 0.854543 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.733917 w_k1_ADM_s102
                                 + 0.66924 w_k1_AXP_s102
                                 + 0.664547 w_k1_DIS_s102
                                 + 0.712834 w_k1_FDX_s102
                                 + 0.623523 w_k1_IBM_s102
                                 + 0.87139 w_k1_INTC_s102
                                 + 0.743075 w_k1_KO_s102
                                 + 0.622418 w_k1_MAT_s102
                                 + 0.847573 w_k1_MSFT_s102
                                 + 0.750348 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.04113 w_k1_ADM_s102
                                 + 1.14877 w_k1_AXP_s102
                                 + 1.05582 w_k1_DIS_s102
                                 + 1.03964 w_k1_FDX_s102
                                 + 0.818515 w_k1_IBM_s102
                                 + 1.13768 w_k1_INTC_s102
                                 + 0.955168 w_k1_KO_s102
                                 + 0.687457 w_k1_MAT_s102
                                 + 0.916881 w_k1_MSFT_s102
                                 + 0.933245 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.644045 w_k1_ADM_s102
                                 + 0.944609 w_k1_AXP_s102
                                 + 0.705423 w_k1_DIS_s102
                                 + 0.778321 w_k1_FDX_s102
                                 + 0.997915 w_k1_IBM_s102
                                 + 0.963277 w_k1_INTC_s102
                                 + 0.843483 w_k1_KO_s102
                                 + 1.30573 w_k1_MAT_s102
                                 + 0.8412 w_k1_MSFT_s102
                                 + 0.829314 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.787327 w_k1_ADM_s103
                                 + 0.667248 w_k1_AXP_s103
                                 + 0.710428 w_k1_DIS_s103
                                 + 0.766791 w_k1_FDX_s103
                                 + 0.610083 w_k1_IBM_s103
                                 + 0.762421 w_k1_INTC_s103
                                 + 0.702963 w_k1_KO_s103
                                 + 0.546157 w_k1_MAT_s103
                                 + 0.794847 w_k1_MSFT_s103
                                 + 0.747993 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.02344 w_k1_ADM_s103
                                 + 1.15529 w_k1_AXP_s103
                                 + 1.40891 w_k1_DIS_s103
                                 + 1.09971 w_k1_FDX_s103
                                 + 1.22384 w_k1_IBM_s103
                                 + 1.13234 w_k1_INTC_s103
                                 + 1.24764 w_k1_KO_s103 + 1.20221 w_k1_MAT_s103
                                 + 1.17009 w_k1_MSFT_s103
                                 + 1.02428 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.950992 w_k1_ADM_s103
                                 + 1.0341 w_k1_AXP_s103
                                 + 0.853994 w_k1_DIS_s103
                                 + 0.894627 w_k1_FDX_s103
                                 + 1.05087 w_k1_IBM_s103
                                 + 1.13219 w_k1_INTC_s103
                                 + 0.951329 w_k1_KO_s103
                                 + 0.999443 w_k1_MAT_s103
                                 + 0.921562 w_k1_MSFT_s103
                                 + 0.914108 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.0612 w_k1_ADM_s103
                                 + 1.26153 w_k1_AXP_s103
                                 + 1.27295 w_k1_DIS_s103
                                 + 1.20619 w_k1_FDX_s103
                                 + 0.90365 w_k1_IBM_s103
                                 + 1.16844 w_k1_INTC_s103
                                 + 1.00434 w_k1_KO_s103
                                 + 0.982902 w_k1_MAT_s103
                                 + 1.12183 w_k1_MSFT_s103 + 1.0471 w_k1_PG_s103
                                 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.898389 w_k1_ADM_s103
                                 + 0.8228 w_k1_AXP_s103
                                 + 0.953533 w_k1_DIS_s103
                                 + 0.859041 w_k1_FDX_s103
                                 + 1.24456 w_k1_IBM_s103
                                 + 1.02755 w_k1_INTC_s103
                                 + 1.04986 w_k1_KO_s103
                                 + 0.973908 w_k1_MAT_s103
                                 + 1.03715 w_k1_MSFT_s103
                                 + 1.01637 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.36248 w_k1_ADM_s103
                                 + 1.63092 w_k1_AXP_s103
                                 + 1.78342 w_k1_DIS_s103
                                 + 1.47076 w_k1_FDX_s103
                                 + 1.03817 w_k1_IBM_s103
                                 + 1.5556 w_k1_INTC_s103 + 1.37979 w_k1_KO_s103
                                 + 1.2292 w_k1_MAT_s103 + 1.4584 w_k1_MSFT_s103
                                 + 1.35196 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.827246 w_k1_ADM_s103
                                 + 0.768946 w_k1_AXP_s103
                                 + 0.622639 w_k1_DIS_s103
                                 + 0.786662 w_k1_FDX_s103
                                 + 0.910926 w_k1_IBM_s103
                                 + 0.890226 w_k1_INTC_s103
                                 + 0.79017 w_k1_KO_s103
                                 + 0.742881 w_k1_MAT_s103
                                 + 0.79092 w_k1_MSFT_s103
                                 + 0.832259 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.19081 w_k1_ADM_s103
                                 + 1.10088 w_k1_AXP_s103
                                 + 0.958286 w_k1_DIS_s103
                                 + 1.1247 w_k1_FDX_s103
                                 + 0.665599 w_k1_IBM_s103
                                 + 1.14028 w_k1_INTC_s103
                                 + 0.710944 w_k1_KO_s103
                                 + 0.668379 w_k1_MAT_s103
                                 + 1.00006 w_k1_MSFT_s103
                                 + 0.914671 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.893478 w_k1_ADM_s104
                                 + 0.961974 w_k1_AXP_s104
                                 + 1.31704 w_k1_DIS_s104
                                 + 1.19688 w_k1_FDX_s104
                                 + 0.818067 w_k1_IBM_s104
                                 + 1.30463 w_k1_INTC_s104
                                 + 1.17459 w_k1_KO_s104
                                 + 0.731722 w_k1_MAT_s104
                                 + 1.2538 w_k1_MSFT_s104 + 1.0636 w_k1_PG_s104
                                 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.842007 w_k1_ADM_s104
                                 + 0.718266 w_k1_AXP_s104
                                 + 0.906047 w_k1_DIS_s104
                                 + 0.939345 w_k1_FDX_s104
                                 + 0.913021 w_k1_IBM_s104
                                 + 1.09273 w_k1_INTC_s104
                                 + 0.994334 w_k1_KO_s104
                                 + 0.712181 w_k1_MAT_s104
                                 + 1.01204 w_k1_MSFT_s104
                                 + 1.01399 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.91722 w_k1_ADM_s104
                                 + 0.834767 w_k1_AXP_s104
                                 + 0.747904 w_k1_DIS_s104
                                 + 0.930474 w_k1_FDX_s104
                                 + 0.604858 w_k1_IBM_s104
                                 + 0.938986 w_k1_INTC_s104
                                 + 0.793467 w_k1_KO_s104
                                 + 0.623329 w_k1_MAT_s104
                                 + 0.961789 w_k1_MSFT_s104
                                 + 0.862517 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.757253 w_k1_ADM_s104
                                 + 0.667106 w_k1_AXP_s104
                                 + 0.620241 w_k1_DIS_s104
                                 + 0.670014 w_k1_FDX_s104
                                 + 0.821039 w_k1_IBM_s104
                                 + 0.802045 w_k1_INTC_s104
                                 + 0.788792 w_k1_KO_s104
                                 + 0.800556 w_k1_MAT_s104
                                 + 0.76293 w_k1_MSFT_s104
                                 + 0.86925 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.594694 w_k1_ADM_s104
                                 + 0.858728 w_k1_AXP_s104
                                 + 0.827107 w_k1_DIS_s104
                                 + 0.810002 w_k1_FDX_s104
                                 + 1.14315 w_k1_IBM_s104
                                 + 1.10405 w_k1_INTC_s104
                                 + 1.08192 w_k1_KO_s104 + 1.18746 w_k1_MAT_s104
                                 + 0.952046 w_k1_MSFT_s104
                                 + 0.903724 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.938068 w_k1_ADM_s104
                                 + 1.08214 w_k1_AXP_s104
                                 + 0.911063 w_k1_DIS_s104
                                 + 0.922372 w_k1_FDX_s104
                                 + 0.948347 w_k1_IBM_s104
                                 + 1.10548 w_k1_INTC_s104
                                 + 0.978147 w_k1_KO_s104
                                 + 1.00216 w_k1_MAT_s104
                                 + 1.0112 w_k1_MSFT_s104 + 1.06321 w_k1_PG_s104
                                 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.12352 w_k1_ADM_s104
                                 + 1.37631 w_k1_AXP_s104
                                 + 1.81227 w_k1_DIS_s104
                                 + 1.38129 w_k1_FDX_s104
                                 + 1.27575 w_k1_IBM_s104
                                 + 1.28515 w_k1_INTC_s104
                                 + 1.53349 w_k1_KO_s104 + 1.20917 w_k1_MAT_s104
                                 + 1.35314 w_k1_MSFT_s104
                                 + 1.18687 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.03503 w_k1_ADM_s104
                                 + 1.18254 w_k1_AXP_s104
                                 + 1.23723 w_k1_DIS_s104
                                 + 1.03338 w_k1_FDX_s104
                                 + 1.26536 w_k1_IBM_s104
                                 + 1.29946 w_k1_INTC_s104
                                 + 1.30102 w_k1_KO_s104 + 1.41004 w_k1_MAT_s104
                                 + 1.07929 w_k1_MSFT_s104
                                 + 1.14678 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.04186 w_k1_ADM_s105
                                 + 1.04224 w_k1_AXP_s105
                                 + 1.05977 w_k1_DIS_s105
                                 + 0.946584 w_k1_FDX_s105
                                 + 1.49653 w_k1_IBM_s105
                                 + 1.21124 w_k1_INTC_s105
                                 + 0.954162 w_k1_KO_s105
                                 + 1.26097 w_k1_MAT_s105
                                 + 0.939376 w_k1_MSFT_s105
                                 + 1.17174 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.751338 w_k1_ADM_s105
                                 + 0.789844 w_k1_AXP_s105
                                 + 0.587347 w_k1_DIS_s105
                                 + 0.688963 w_k1_FDX_s105
                                 + 0.901051 w_k1_IBM_s105
                                 + 0.735559 w_k1_INTC_s105
                                 + 0.651686 w_k1_KO_s105
                                 + 0.906856 w_k1_MAT_s105
                                 + 0.653559 w_k1_MSFT_s105
                                 + 0.78172 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.885425 w_k1_ADM_s105
                                 + 0.746121 w_k1_AXP_s105
                                 + 0.824965 w_k1_DIS_s105
                                 + 0.815832 w_k1_FDX_s105
                                 + 0.756715 w_k1_IBM_s105
                                 + 0.985705 w_k1_INTC_s105
                                 + 0.721204 w_k1_KO_s105
                                 + 0.641999 w_k1_MAT_s105
                                 + 0.793024 w_k1_MSFT_s105
                                 + 0.922472 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.21383 w_k1_ADM_s105
                                 + 1.3426 w_k1_AXP_s105 + 1.23596 w_k1_DIS_s105
                                 + 1.22187 w_k1_FDX_s105
                                 + 0.97763 w_k1_IBM_s105
                                 + 1.37516 w_k1_INTC_s105
                                 + 0.867405 w_k1_KO_s105
                                 + 1.03752 w_k1_MAT_s105
                                 + 1.05539 w_k1_MSFT_s105
                                 + 1.23838 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.978463 w_k1_ADM_s105
                                 + 0.861753 w_k1_AXP_s105
                                 + 1.34343 w_k1_DIS_s105
                                 + 1.09412 w_k1_FDX_s105
                                 + 0.82661 w_k1_IBM_s105
                                 + 1.28358 w_k1_INTC_s105
                                 + 1.03461 w_k1_KO_s105
                                 + 0.833912 w_k1_MAT_s105
                                 + 1.24911 w_k1_MSFT_s105
                                 + 1.17896 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.997464 w_k1_ADM_s105
                                 + 0.885978 w_k1_AXP_s105
                                 + 0.938475 w_k1_DIS_s105
                                 + 0.963995 w_k1_FDX_s105
                                 + 0.78611 w_k1_IBM_s105
                                 + 1.07575 w_k1_INTC_s105
                                 + 0.822244 w_k1_KO_s105
                                 + 0.848874 w_k1_MAT_s105
                                 + 0.968265 w_k1_MSFT_s105
                                 + 1.0418 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.902981 w_k1_ADM_s105
                                 + 0.928749 w_k1_AXP_s105
                                 + 0.856672 w_k1_DIS_s105
                                 + 0.882676 w_k1_FDX_s105
                                 + 1.19311 w_k1_IBM_s105
                                 + 0.802704 w_k1_INTC_s105
                                 + 0.814114 w_k1_KO_s105
                                 + 1.24592 w_k1_MAT_s105
                                 + 0.856811 w_k1_MSFT_s105
                                 + 0.985852 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.971042 w_k1_ADM_s105
                                 + 1.11468 w_k1_AXP_s105
                                 + 1.31114 w_k1_DIS_s105
                                 + 1.04966 w_k1_FDX_s105
                                 + 1.21129 w_k1_IBM_s105
                                 + 1.18329 w_k1_INTC_s105
                                 + 1.15274 w_k1_KO_s105 + 1.3075 w_k1_MAT_s105
                                 + 1.12229 w_k1_MSFT_s105
                                 + 1.11803 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.21279 w_k1_ADM_s106
                                 + 1.0215 w_k1_AXP_s106 + 1.13533 w_k1_DIS_s106
                                 + 1.12609 w_k1_FDX_s106
                                 + 0.780304 w_k1_IBM_s106
                                 + 1.23896 w_k1_INTC_s106
                                 + 1.00304 w_k1_KO_s106
                                 + 0.794531 w_k1_MAT_s106
                                 + 1.13326 w_k1_MSFT_s106
                                 + 0.988708 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.760163 w_k1_ADM_s106
                                 + 0.630495 w_k1_AXP_s106
                                 + 0.524128 w_k1_DIS_s106
                                 + 0.636576 w_k1_FDX_s106
                                 + 0.699756 w_k1_IBM_s106
                                 + 0.831844 w_k1_INTC_s106
                                 + 0.727852 w_k1_KO_s106
                                 + 0.880879 w_k1_MAT_s106
                                 + 0.742902 w_k1_MSFT_s106
                                 + 0.778488 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.832825 w_k1_ADM_s106
                                 + 0.772803 w_k1_AXP_s106
                                 + 0.657121 w_k1_DIS_s106
                                 + 0.894257 w_k1_FDX_s106
                                 + 0.623771 w_k1_IBM_s106
                                 + 0.890974 w_k1_INTC_s106
                                 + 0.667457 w_k1_KO_s106
                                 + 0.479705 w_k1_MAT_s106
                                 + 0.808839 w_k1_MSFT_s106
                                 + 0.809669 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.76865 w_k1_ADM_s106
                                 + 1.44508 w_k1_AXP_s106
                                 + 1.91083 w_k1_DIS_s106
                                 + 1.55952 w_k1_FDX_s106
                                 + 0.882828 w_k1_IBM_s106
                                 + 1.61177 w_k1_INTC_s106
                                 + 1.43126 w_k1_KO_s106 + 1.352 w_k1_MAT_s106
                                 + 1.59051 w_k1_MSFT_s106
                                 + 1.29733 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.20008 w_k1_ADM_s106
                                 + 1.4641 w_k1_AXP_s106 + 1.33735 w_k1_DIS_s106
                                 + 1.19818 w_k1_FDX_s106
                                 + 1.35863 w_k1_IBM_s106
                                 + 1.30876 w_k1_INTC_s106
                                 + 1.33215 w_k1_KO_s106 + 1.45963 w_k1_MAT_s106
                                 + 1.21006 w_k1_MSFT_s106
                                 + 1.20559 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.884036 w_k1_ADM_s106
                                 + 0.81416 w_k1_AXP_s106
                                 + 0.565852 w_k1_DIS_s106
                                 + 0.748388 w_k1_FDX_s106
                                 + 1.08304 w_k1_IBM_s106
                                 + 0.9911 w_k1_INTC_s106
                                 + 0.818544 w_k1_KO_s106
                                 + 1.09491 w_k1_MAT_s106
                                 + 0.742591 w_k1_MSFT_s106
                                 + 0.872295 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.968715 w_k1_ADM_s106
                                 + 0.948182 w_k1_AXP_s106
                                 + 1.03003 w_k1_DIS_s106
                                 + 0.954996 w_k1_FDX_s106
                                 + 0.863331 w_k1_IBM_s106
                                 + 1.00632 w_k1_INTC_s106
                                 + 0.933007 w_k1_KO_s106
                                 + 0.964538 w_k1_MAT_s106
                                 + 0.936059 w_k1_MSFT_s106
                                 + 0.860362 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.888838 w_k1_ADM_s106
                                 + 0.836413 w_k1_AXP_s106
                                 + 0.928599 w_k1_DIS_s106
                                 + 0.842408 w_k1_FDX_s106
                                 + 1.19878 w_k1_IBM_s106
                                 + 1.08453 w_k1_INTC_s106
                                 + 1.14171 w_k1_KO_s106
                                 + 0.996275 w_k1_MAT_s106
                                 + 0.948421 w_k1_MSFT_s106
                                 + 1.01192 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.64004 w_k1_ADM_s107
                                 + 0.558321 w_k1_AXP_s107
                                 + 0.53156 w_k1_DIS_s107
                                 + 0.644343 w_k1_FDX_s107
                                 + 0.741065 w_k1_IBM_s107
                                 + 0.600157 w_k1_INTC_s107
                                 + 0.669148 w_k1_KO_s107
                                 + 0.612991 w_k1_MAT_s107
                                 + 0.653868 w_k1_MSFT_s107
                                 + 0.851096 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.778746 w_k1_ADM_s107
                                 + 0.836308 w_k1_AXP_s107
                                 + 0.79031 w_k1_DIS_s107
                                 + 0.917562 w_k1_FDX_s107
                                 + 0.687997 w_k1_IBM_s107
                                 + 0.822616 w_k1_INTC_s107
                                 + 0.759429 w_k1_KO_s107
                                 + 0.889998 w_k1_MAT_s107
                                 + 0.825802 w_k1_MSFT_s107
                                 + 0.975296 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.15984 w_k1_ADM_s107
                                 + 0.936435 w_k1_AXP_s107
                                 + 1.17728 w_k1_DIS_s107
                                 + 1.08036 w_k1_FDX_s107
                                 + 0.82067 w_k1_IBM_s107
                                 + 0.907086 w_k1_INTC_s107
                                 + 0.928598 w_k1_KO_s107
                                 + 1.04685 w_k1_MAT_s107
                                 + 1.06461 w_k1_MSFT_s107
                                 + 0.991434 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.956329 w_k1_ADM_s107
                                 + 1.17679 w_k1_AXP_s107
                                 + 1.28382 w_k1_DIS_s107
                                 + 1.08191 w_k1_FDX_s107
                                 + 1.13421 w_k1_IBM_s107
                                 + 1.00467 w_k1_INTC_s107
                                 + 1.04187 w_k1_KO_s107 + 1.4409 w_k1_MAT_s107
                                 + 1.18993 w_k1_MSFT_s107
                                 + 1.18698 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.25476 w_k1_ADM_s107
                                 + 1.16313 w_k1_AXP_s107
                                 + 1.43938 w_k1_DIS_s107
                                 + 1.16547 w_k1_FDX_s107
                                 + 1.48837 w_k1_IBM_s107
                                 + 1.09668 w_k1_INTC_s107
                                 + 1.26379 w_k1_KO_s107 + 1.70935 w_k1_MAT_s107
                                 + 1.10899 w_k1_MSFT_s107
                                 + 1.43151 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.11164 w_k1_ADM_s107
                                 + 1.27786 w_k1_AXP_s107
                                 + 1.18987 w_k1_DIS_s107 + 1.2389 w_k1_FDX_s107
                                 + 0.793597 w_k1_IBM_s107
                                 + 1.01054 w_k1_INTC_s107
                                 + 0.870341 w_k1_KO_s107
                                 + 1.29823 w_k1_MAT_s107
                                 + 1.0946 w_k1_MSFT_s107 + 1.13475 w_k1_PG_s107
                                 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.28392 w_k1_ADM_s107
                                 + 1.63824 w_k1_AXP_s107
                                 + 1.98634 w_k1_DIS_s107
                                 + 1.56123 w_k1_FDX_s107
                                 + 1.23711 w_k1_IBM_s107
                                 + 1.48304 w_k1_INTC_s107
                                 + 1.40525 w_k1_KO_s107 + 1.6233 w_k1_MAT_s107
                                 + 1.32573 w_k1_MSFT_s107
                                 + 1.23605 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.820633 w_k1_ADM_s107
                                 + 0.930159 w_k1_AXP_s107
                                 + 0.788935 w_k1_DIS_s107
                                 + 0.836775 w_k1_FDX_s107
                                 + 1.03404 w_k1_IBM_s107
                                 + 0.708025 w_k1_INTC_s107
                                 + 0.887051 w_k1_KO_s107 + 1.1968 w_k1_MAT_s107
                                 + 0.837064 w_k1_MSFT_s107
                                 + 1.06184 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.17819 w_k1_ADM_s110
                                 + 1.3089 w_k1_AXP_s110 + 1.18281 w_k1_DIS_s110
                                 + 1.08074 w_k1_FDX_s110
                                 + 1.02712 w_k1_IBM_s110
                                 + 0.936289 w_k1_INTC_s110
                                 + 1.00652 w_k1_KO_s110 + 1.1427 w_k1_MAT_s110
                                 + 0.988979 w_k1_MSFT_s110
                                 + 1.05893 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.948125 w_k1_ADM_s110
                                 + 0.837471 w_k1_AXP_s110
                                 + 0.915298 w_k1_DIS_s110
                                 + 0.931677 w_k1_FDX_s110
                                 + 1.02587 w_k1_IBM_s110
                                 + 0.957877 w_k1_INTC_s110
                                 + 0.976353 w_k1_KO_s110 + 1.322 w_k1_MAT_s110
                                 + 0.922833 w_k1_MSFT_s110
                                 + 0.893785 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.960653 w_k1_ADM_s110
                                 + 0.766761 w_k1_AXP_s110
                                 + 0.586363 w_k1_DIS_s110
                                 + 0.76487 w_k1_FDX_s110
                                 + 0.808604 w_k1_IBM_s110
                                 + 0.910702 w_k1_INTC_s110
                                 + 0.766479 w_k1_KO_s110
                                 + 1.03698 w_k1_MAT_s110
                                 + 0.828267 w_k1_MSFT_s110
                                 + 0.948348 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.04951 w_k1_ADM_s110
                                 + 0.803207 w_k1_AXP_s110
                                 + 1.25782 w_k1_DIS_s110
                                 + 0.972691 w_k1_FDX_s110
                                 + 1.10312 w_k1_IBM_s110
                                 + 0.909274 w_k1_INTC_s110
                                 + 1.18754 w_k1_KO_s110 + 1.30518 w_k1_MAT_s110
                                 + 1.08069 w_k1_MSFT_s110
                                 + 1.04785 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.08406 w_k1_ADM_s110
                                 + 1.255 w_k1_AXP_s110 + 1.4409 w_k1_DIS_s110
                                 + 1.05919 w_k1_FDX_s110
                                 + 1.39578 w_k1_IBM_s110
                                 + 1.12991 w_k1_INTC_s110
                                 + 1.56003 w_k1_KO_s110 + 2.05607 w_k1_MAT_s110
                                 + 1.21088 w_k1_MSFT_s110
                                 + 1.18832 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.16216 w_k1_ADM_s110
                                 + 0.975491 w_k1_AXP_s110
                                 + 0.854098 w_k1_DIS_s110
                                 + 1.03311 w_k1_FDX_s110
                                 + 0.737354 w_k1_IBM_s110
                                 + 0.963819 w_k1_INTC_s110
                                 + 0.934628 w_k1_KO_s110
                                 + 1.12517 w_k1_MAT_s110
                                 + 0.984881 w_k1_MSFT_s110
                                 + 0.976278 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.44095 w_k1_ADM_s110
                                 + 1.20569 w_k1_AXP_s110
                                 + 1.38142 w_k1_DIS_s110
                                 + 1.34733 w_k1_FDX_s110
                                 + 0.956628 w_k1_IBM_s110
                                 + 1.09531 w_k1_INTC_s110
                                 + 1.22961 w_k1_KO_s110 + 1.67656 w_k1_MAT_s110
                                 + 1.33195 w_k1_MSFT_s110 + 1.1963 w_k1_PG_s110
                                 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.41891 w_k1_ADM_s110
                                 + 1.26942 w_k1_AXP_s110
                                 + 1.43391 w_k1_DIS_s110
                                 + 1.14499 w_k1_FDX_s110 + 1.5285 w_k1_IBM_s110
                                 + 1.37125 w_k1_INTC_s110
                                 + 1.58939 w_k1_KO_s110 + 2.00621 w_k1_MAT_s110
                                 + 1.23439 w_k1_MSFT_s110
                                 + 1.27928 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.803839 w_k1_ADM_s111
                                 + 0.801703 w_k1_AXP_s111
                                 + 0.595567 w_k1_DIS_s111
                                 + 0.762297 w_k1_FDX_s111
                                 + 0.936505 w_k1_IBM_s111
                                 + 0.868508 w_k1_INTC_s111
                                 + 0.863232 w_k1_KO_s111
                                 + 0.875357 w_k1_MAT_s111
                                 + 0.736738 w_k1_MSFT_s111
                                 + 0.814869 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.89665 w_k1_ADM_s111
                                 + 1.07078 w_k1_AXP_s111
                                 + 0.890431 w_k1_DIS_s111
                                 + 0.930889 w_k1_FDX_s111
                                 + 1.07946 w_k1_IBM_s111
                                 + 0.919738 w_k1_INTC_s111
                                 + 1.02714 w_k1_KO_s111 + 1.28683 w_k1_MAT_s111
                                 + 0.894302 w_k1_MSFT_s111
                                 + 1.07938 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.04231 w_k1_ADM_s111
                                 + 0.803526 w_k1_AXP_s111
                                 + 1.0454 w_k1_DIS_s111 + 1.00238 w_k1_FDX_s111
                                 + 0.929316 w_k1_IBM_s111
                                 + 1.1079 w_k1_INTC_s111 + 1.18251 w_k1_KO_s111
                                 + 1.07886 w_k1_MAT_s111
                                 + 1.11216 w_k1_MSFT_s111
                                 + 0.986676 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.27727 w_k1_ADM_s111
                                 + 1.12461 w_k1_AXP_s111
                                 + 0.708867 w_k1_DIS_s111
                                 + 0.845334 w_k1_FDX_s111
                                 + 0.777637 w_k1_IBM_s111
                                 + 0.82946 w_k1_INTC_s111
                                 + 0.873705 w_k1_KO_s111
                                 + 1.13222 w_k1_MAT_s111
                                 + 0.839046 w_k1_MSFT_s111
                                 + 0.939582 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.25632 w_k1_ADM_s111
                                 + 1.10933 w_k1_AXP_s111
                                 + 1.04616 w_k1_DIS_s111
                                 + 1.14319 w_k1_FDX_s111
                                 + 0.821881 w_k1_IBM_s111
                                 + 0.859757 w_k1_INTC_s111
                                 + 0.89838 w_k1_KO_s111 + 1.35321 w_k1_MAT_s111
                                 + 0.986085 w_k1_MSFT_s111
                                 + 1.00659 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.08624 w_k1_ADM_s111
                                 + 0.921622 w_k1_AXP_s111
                                 + 0.791074 w_k1_DIS_s111
                                 + 0.999262 w_k1_FDX_s111
                                 + 0.673633 w_k1_IBM_s111
                                 + 1.08945 w_k1_INTC_s111
                                 + 0.828509 w_k1_KO_s111
                                 + 0.94437 w_k1_MAT_s111
                                 + 0.985288 w_k1_MSFT_s111
                                 + 0.896784 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.17763 w_k1_ADM_s111
                                 + 1.51877 w_k1_AXP_s111
                                 + 1.52454 w_k1_DIS_s111
                                 + 1.32781 w_k1_FDX_s111
                                 + 1.13168 w_k1_IBM_s111
                                 + 1.27795 w_k1_INTC_s111
                                 + 1.04508 w_k1_KO_s111 + 1.15435 w_k1_MAT_s111
                                 + 1.34012 w_k1_MSFT_s111
                                 + 1.11114 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.940585 w_k1_ADM_s111
                                 + 1.25097 w_k1_AXP_s111 + 1.0174 w_k1_DIS_s111
                                 + 1.06351 w_k1_FDX_s111
                                 + 1.33864 w_k1_IBM_s111
                                 + 0.978324 w_k1_INTC_s111
                                 + 1.02903 w_k1_KO_s111 + 1.76536 w_k1_MAT_s111
                                 + 0.938205 w_k1_MSFT_s111
                                 + 1.19372 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.770269 w_k1_ADM_s112
                                 + 1.1969 w_k1_AXP_s112 + 1.47775 w_k1_DIS_s112
                                 + 1.08983 w_k1_FDX_s112
                                 + 1.69613 w_k1_IBM_s112
                                 + 1.37193 w_k1_INTC_s112
                                 + 1.23282 w_k1_KO_s112 + 1.92965 w_k1_MAT_s112
                                 + 1.14395 w_k1_MSFT_s112
                                 + 1.21091 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.12405 w_k1_ADM_s112
                                 + 0.960354 w_k1_AXP_s112
                                 + 1.39335 w_k1_DIS_s112
                                 + 1.19895 w_k1_FDX_s112
                                 + 0.752137 w_k1_IBM_s112
                                 + 1.05423 w_k1_INTC_s112
                                 + 0.871267 w_k1_KO_s112
                                 + 0.613318 w_k1_MAT_s112
                                 + 1.19582 w_k1_MSFT_s112
                                 + 1.18465 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.37791 w_k1_ADM_s112
                                 + 1.65087 w_k1_AXP_s112
                                 + 1.51593 w_k1_DIS_s112
                                 + 1.30389 w_k1_FDX_s112
                                 + 1.31926 w_k1_IBM_s112
                                 + 1.21144 w_k1_INTC_s112
                                 + 1.12861 w_k1_KO_s112 + 1.46604 w_k1_MAT_s112
                                 + 1.0785 w_k1_MSFT_s112 + 1.06867 w_k1_PG_s112
                                 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.26509 w_k1_ADM_s112
                                 + 0.960286 w_k1_AXP_s112
                                 + 1.20921 w_k1_DIS_s112
                                 + 1.06114 w_k1_FDX_s112
                                 + 0.85726 w_k1_IBM_s112
                                 + 0.914704 w_k1_INTC_s112
                                 + 0.876235 w_k1_KO_s112
                                 + 0.70191 w_k1_MAT_s112
                                 + 1.00114 w_k1_MSFT_s112
                                 + 1.02569 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.05096 w_k1_ADM_s112
                                 + 1.07605 w_k1_AXP_s112
                                 + 0.955886 w_k1_DIS_s112
                                 + 0.960199 w_k1_FDX_s112
                                 + 1.00554 w_k1_IBM_s112
                                 + 0.855972 w_k1_INTC_s112
                                 + 0.913712 w_k1_KO_s112
                                 + 0.86857 w_k1_MAT_s112
                                 + 0.786738 w_k1_MSFT_s112
                                 + 1.09185 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.967362 w_k1_ADM_s112
                                 + 0.952789 w_k1_AXP_s112
                                 + 0.772558 w_k1_DIS_s112
                                 + 0.856123 w_k1_FDX_s112
                                 + 0.668503 w_k1_IBM_s112
                                 + 0.827298 w_k1_INTC_s112
                                 + 0.71773 w_k1_KO_s112
                                 + 0.593789 w_k1_MAT_s112
                                 + 0.790705 w_k1_MSFT_s112
                                 + 0.876658 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.72298 w_k1_ADM_s112
                                 + 0.685094 w_k1_AXP_s112
                                 + 0.637265 w_k1_DIS_s112
                                 + 0.753468 w_k1_FDX_s112
                                 + 0.818635 w_k1_IBM_s112
                                 + 0.828061 w_k1_INTC_s112
                                 + 0.63079 w_k1_KO_s112
                                 + 0.633592 w_k1_MAT_s112
                                 + 0.697841 w_k1_MSFT_s112
                                 + 0.902279 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.887671 w_k1_ADM_s112
                                 + 1.02531 w_k1_AXP_s112
                                 + 1.03403 w_k1_DIS_s112
                                 + 0.977298 w_k1_FDX_s112
                                 + 1.1826 w_k1_IBM_s112
                                 + 1.03155 w_k1_INTC_s112
                                 + 0.87185 w_k1_KO_s112
                                 + 0.970061 w_k1_MAT_s112
                                 + 0.967996 w_k1_MSFT_s112
                                 + 1.09492 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.484021 w_k1_ADM_s113
                                 + 1.03603 w_k1_AXP_s113
                                 + 0.871629 w_k1_DIS_s113
                                 + 0.766226 w_k1_FDX_s113
                                 + 1.02567 w_k1_IBM_s113
                                 + 0.969455 w_k1_INTC_s113
                                 + 0.856837 w_k1_KO_s113
                                 + 1.42694 w_k1_MAT_s113
                                 + 0.917255 w_k1_MSFT_s113
                                 + 0.921687 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.0077 w_k1_ADM_s113
                                 + 1.4292 w_k1_AXP_s113 + 1.22913 w_k1_DIS_s113
                                 + 1.12387 w_k1_FDX_s113
                                 + 1.34506 w_k1_IBM_s113
                                 + 1.18604 w_k1_INTC_s113 + 1.1523 w_k1_KO_s113
                                 + 1.36196 w_k1_MAT_s113
                                 + 1.06446 w_k1_MSFT_s113
                                 + 1.13811 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.93094 w_k1_ADM_s113
                                 + 1.06143 w_k1_AXP_s113 + 1.2674 w_k1_DIS_s113
                                 + 1.26192 w_k1_FDX_s113
                                 + 0.698674 w_k1_IBM_s113
                                 + 1.32892 w_k1_INTC_s113
                                 + 0.972311 w_k1_KO_s113
                                 + 0.838357 w_k1_MAT_s113
                                 + 1.26076 w_k1_MSFT_s113
                                 + 1.01621 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.723544 w_k1_ADM_s113
                                 + 0.725423 w_k1_AXP_s113
                                 + 0.820398 w_k1_DIS_s113
                                 + 0.767988 w_k1_FDX_s113
                                 + 0.850125 w_k1_IBM_s113
                                 + 0.839537 w_k1_INTC_s113
                                 + 0.846557 w_k1_KO_s113
                                 + 0.957486 w_k1_MAT_s113
                                 + 0.916872 w_k1_MSFT_s113
                                 + 0.941249 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.869817 w_k1_ADM_s113
                                 + 1.3043 w_k1_AXP_s113 + 1.49269 w_k1_DIS_s113
                                 + 1.23308 w_k1_FDX_s113
                                 + 1.04869 w_k1_IBM_s113
                                 + 1.37349 w_k1_INTC_s113
                                 + 1.20322 w_k1_KO_s113 + 1.10497 w_k1_MAT_s113
                                 + 1.18723 w_k1_MSFT_s113 + 1.0936 w_k1_PG_s113
                                 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.644418 w_k1_ADM_s113
                                 + 0.871456 w_k1_AXP_s113
                                 + 1.20911 w_k1_DIS_s113
                                 + 1.00414 w_k1_FDX_s113 + 1.7207 w_k1_IBM_s113
                                 + 1.13799 w_k1_INTC_s113
                                 + 1.27323 w_k1_KO_s113 + 2.01933 w_k1_MAT_s113
                                 + 1.05416 w_k1_MSFT_s113
                                 + 1.01214 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.858812 w_k1_ADM_s113
                                 + 1.04783 w_k1_AXP_s113
                                 + 0.878249 w_k1_DIS_s113
                                 + 0.951291 w_k1_FDX_s113
                                 + 0.924626 w_k1_IBM_s113
                                 + 0.957444 w_k1_INTC_s113
                                 + 0.865653 w_k1_KO_s113
                                 + 1.01295 w_k1_MAT_s113
                                 + 0.932031 w_k1_MSFT_s113
                                 + 0.921701 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.01762 w_k1_ADM_s113
                                 + 1.174 w_k1_AXP_s113 + 1.45849 w_k1_DIS_s113
                                 + 1.13633 w_k1_FDX_s113
                                 + 1.40979 w_k1_IBM_s113
                                 + 1.13138 w_k1_INTC_s113
                                 + 1.42929 w_k1_KO_s113 + 1.65282 w_k1_MAT_s113
                                 + 1.30112 w_k1_MSFT_s113 + 1.3109 w_k1_PG_s113
                                 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.995082 w_k1_ADM_s114
                                 + 0.901429 w_k1_AXP_s114
                                 + 0.96375 w_k1_DIS_s114
                                 + 1.01041 w_k1_FDX_s114
                                 + 0.86462 w_k1_IBM_s114
                                 + 1.09827 w_k1_INTC_s114 + 1.0635 w_k1_KO_s114
                                 + 1.06106 w_k1_MAT_s114
                                 + 1.17591 w_k1_MSFT_s114
                                 + 1.07722 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.22263 w_k1_ADM_s114
                                 + 1.22122 w_k1_AXP_s114
                                 + 1.00902 w_k1_DIS_s114
                                 + 1.13815 w_k1_FDX_s114
                                 + 1.09886 w_k1_IBM_s114
                                 + 1.19298 w_k1_INTC_s114
                                 + 1.11333 w_k1_KO_s114 + 1.16335 w_k1_MAT_s114
                                 + 1.09416 w_k1_MSFT_s114
                                 + 0.908252 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.727207 w_k1_ADM_s114
                                 + 0.656585 w_k1_AXP_s114
                                 + 0.735545 w_k1_DIS_s114
                                 + 0.826216 w_k1_FDX_s114
                                 + 0.7994 w_k1_IBM_s114
                                 + 0.975442 w_k1_INTC_s114
                                 + 0.74391 w_k1_KO_s114
                                 + 0.997861 w_k1_MAT_s114
                                 + 0.94937 w_k1_MSFT_s114
                                 + 0.903853 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.964086 w_k1_ADM_s114
                                 + 1.31271 w_k1_AXP_s114
                                 + 1.48812 w_k1_DIS_s114 + 1.1563 w_k1_FDX_s114
                                 + 1.20078 w_k1_IBM_s114
                                 + 1.2793 w_k1_INTC_s114 + 1.27555 w_k1_KO_s114
                                 + 1.80374 w_k1_MAT_s114
                                 + 1.34139 w_k1_MSFT_s114
                                 + 1.10936 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.02572 w_k1_ADM_s114
                                 + 1.11126 w_k1_AXP_s114
                                 + 1.39069 w_k1_DIS_s114
                                 + 1.01709 w_k1_FDX_s114
                                 + 1.59193 w_k1_IBM_s114
                                 + 1.11851 w_k1_INTC_s114
                                 + 1.57386 w_k1_KO_s114 + 2.04383 w_k1_MAT_s114
                                 + 1.06284 w_k1_MSFT_s114
                                 + 1.33981 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.649322 w_k1_ADM_s114
                                 + 0.889144 w_k1_AXP_s114
                                 + 0.786628 w_k1_DIS_s114
                                 + 0.812436 w_k1_FDX_s114
                                 + 1.17092 w_k1_IBM_s114
                                 + 0.943803 w_k1_INTC_s114
                                 + 0.902497 w_k1_KO_s114
                                 + 1.26251 w_k1_MAT_s114
                                 + 0.921739 w_k1_MSFT_s114
                                 + 0.926215 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.05084 w_k1_ADM_s114
                                 + 0.882988 w_k1_AXP_s114
                                 + 0.86726 w_k1_DIS_s114
                                 + 1.05613 w_k1_FDX_s114
                                 + 0.825551 w_k1_IBM_s114
                                 + 1.16807 w_k1_INTC_s114
                                 + 0.789014 w_k1_KO_s114
                                 + 0.596867 w_k1_MAT_s114
                                 + 1.01039 w_k1_MSFT_s114 + 0.9041 w_k1_PG_s114
                                 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.64429 w_k1_ADM_s114
                                 + 1.25462 w_k1_AXP_s114
                                 + 1.42116 w_k1_DIS_s114
                                 + 1.28084 w_k1_FDX_s114
                                 + 0.91896 w_k1_IBM_s114
                                 + 1.2703 w_k1_INTC_s114 + 1.16174 w_k1_KO_s114
                                 + 1.02024 w_k1_MAT_s114
                                 + 1.1913 w_k1_MSFT_s114 + 1.08499 w_k1_PG_s114
                                 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.10106 w_k1_ADM_s115
                                 + 0.886067 w_k1_AXP_s115
                                 + 1.10681 w_k1_DIS_s115
                                 + 0.938637 w_k1_FDX_s115
                                 + 0.751853 w_k1_IBM_s115
                                 + 1.02058 w_k1_INTC_s115
                                 + 0.931189 w_k1_KO_s115
                                 + 0.649948 w_k1_MAT_s115
                                 + 1.14033 w_k1_MSFT_s115
                                 + 0.974323 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.856441 w_k1_ADM_s115
                                 + 0.554789 w_k1_AXP_s115
                                 + 0.672112 w_k1_DIS_s115
                                 + 0.76416 w_k1_FDX_s115
                                 + 0.605593 w_k1_IBM_s115
                                 + 0.708964 w_k1_INTC_s115
                                 + 0.722489 w_k1_KO_s115
                                 + 0.549901 w_k1_MAT_s115
                                 + 0.932067 w_k1_MSFT_s115
                                 + 0.830493 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.09989 w_k1_ADM_s115
                                 + 1.42199 w_k1_AXP_s115
                                 + 1.63741 w_k1_DIS_s115
                                 + 1.17715 w_k1_FDX_s115
                                 + 1.13674 w_k1_IBM_s115
                                 + 1.07556 w_k1_INTC_s115
                                 + 1.15695 w_k1_KO_s115 + 1.47642 w_k1_MAT_s115
                                 + 1.20382 w_k1_MSFT_s115
                                 + 1.10489 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.1792 w_k1_ADM_s115
                                 + 1.48433 w_k1_AXP_s115
                                 + 1.89205 w_k1_DIS_s115 + 1.3394 w_k1_FDX_s115
                                 + 1.62495 w_k1_IBM_s115
                                 + 0.951101 w_k1_INTC_s115
                                 + 1.63625 w_k1_KO_s115 + 2.42102 w_k1_MAT_s115
                                 + 1.34049 w_k1_MSFT_s115
                                 + 1.19617 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.766736 w_k1_ADM_s115
                                 + 0.595897 w_k1_AXP_s115
                                 + 0.592984 w_k1_DIS_s115
                                 + 0.637876 w_k1_FDX_s115
                                 + 0.874978 w_k1_IBM_s115
                                 + 0.659144 w_k1_INTC_s115
                                 + 0.709487 w_k1_KO_s115
                                 + 0.81944 w_k1_MAT_s115
                                 + 0.850901 w_k1_MSFT_s115
                                 + 0.818842 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.886501 w_k1_ADM_s115
                                 + 0.87946 w_k1_AXP_s115
                                 + 0.674673 w_k1_DIS_s115
                                 + 0.771716 w_k1_FDX_s115
                                 + 0.971647 w_k1_IBM_s115
                                 + 0.892917 w_k1_INTC_s115
                                 + 0.761274 w_k1_KO_s115
                                 + 1.12506 w_k1_MAT_s115
                                 + 0.890251 w_k1_MSFT_s115
                                 + 0.932424 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.01016 w_k1_ADM_s115
                                 + 1.18976 w_k1_AXP_s115 + 1.1203 w_k1_DIS_s115
                                 + 1.05694 w_k1_FDX_s115
                                 + 0.847494 w_k1_IBM_s115
                                 + 0.870627 w_k1_INTC_s115
                                 + 0.947777 w_k1_KO_s115
                                 + 1.15273 w_k1_MAT_s115
                                 + 1.21775 w_k1_MSFT_s115
                                 + 1.12139 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.80602 w_k1_ADM_s115
                                 + 0.82867 w_k1_AXP_s115
                                 + 0.811048 w_k1_DIS_s115
                                 + 0.796453 w_k1_FDX_s115
                                 + 1.31787 w_k1_IBM_s115
                                 + 0.821931 w_k1_INTC_s115
                                 + 1.11963 w_k1_KO_s115 + 1.56806 w_k1_MAT_s115
                                 + 0.992321 w_k1_MSFT_s115
                                 + 1.0065 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.716474 w_k1_ADM_s116
                                 + 0.8665 w_k1_AXP_s116
                                 + 0.595377 w_k1_DIS_s116
                                 + 0.781435 w_k1_FDX_s116
                                 + 0.975588 w_k1_IBM_s116
                                 + 0.937453 w_k1_INTC_s116
                                 + 0.783958 w_k1_KO_s116
                                 + 0.765521 w_k1_MAT_s116
                                 + 0.74039 w_k1_MSFT_s116
                                 + 0.800414 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.04991 w_k1_ADM_s116
                                 + 2.2238 w_k1_AXP_s116 + 1.64786 w_k1_DIS_s116
                                 + 1.41357 w_k1_FDX_s116
                                 + 1.32767 w_k1_IBM_s116
                                 + 1.13959 w_k1_INTC_s116
                                 + 1.23954 w_k1_KO_s116 + 1.96819 w_k1_MAT_s116
                                 + 1.13588 w_k1_MSFT_s116
                                 + 1.24537 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.524929 w_k1_ADM_s116
                                 + 0.781633 w_k1_AXP_s116
                                 + 0.727776 w_k1_DIS_s116
                                 + 0.744781 w_k1_FDX_s116
                                 + 0.642317 w_k1_IBM_s116
                                 + 0.757419 w_k1_INTC_s116
                                 + 0.800327 w_k1_KO_s116
                                 + 0.551086 w_k1_MAT_s116
                                 + 0.670213 w_k1_MSFT_s116
                                 + 0.724816 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.874765 w_k1_ADM_s116
                                 + 1.026 w_k1_AXP_s116 + 0.9282 w_k1_DIS_s116
                                 + 0.992028 w_k1_FDX_s116
                                 + 0.865101 w_k1_IBM_s116
                                 + 1.05921 w_k1_INTC_s116
                                 + 1.00567 w_k1_KO_s116 + 1.08326 w_k1_MAT_s116
                                 + 0.913135 w_k1_MSFT_s116
                                 + 0.986621 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.954749 w_k1_ADM_s116
                                 + 1.5877 w_k1_AXP_s116
                                 + 0.861054 w_k1_DIS_s116
                                 + 1.20286 w_k1_FDX_s116
                                 + 0.787097 w_k1_IBM_s116
                                 + 1.04791 w_k1_INTC_s116
                                 + 0.887561 w_k1_KO_s116
                                 + 0.96884 w_k1_MAT_s116
                                 + 0.994268 w_k1_MSFT_s116
                                 + 0.934116 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.08097 w_k1_ADM_s116
                                 + 0.889285 w_k1_AXP_s116
                                 + 1.15455 w_k1_DIS_s116
                                 + 1.07428 w_k1_FDX_s116
                                 + 0.885063 w_k1_IBM_s116
                                 + 1.27638 w_k1_INTC_s116
                                 + 1.13261 w_k1_KO_s116
                                 + 0.553903 w_k1_MAT_s116
                                 + 1.1281 w_k1_MSFT_s116 + 1.09244 w_k1_PG_s116
                                 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.878793 w_k1_ADM_s116
                                 + 1.26222 w_k1_AXP_s116
                                 + 1.27786 w_k1_DIS_s116
                                 + 1.33462 w_k1_FDX_s116
                                 + 0.839949 w_k1_IBM_s116
                                 + 1.37776 w_k1_INTC_s116
                                 + 1.10499 w_k1_KO_s116
                                 + 0.784373 w_k1_MAT_s116
                                 + 1.25243 w_k1_MSFT_s116
                                 + 0.948236 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.779863 w_k1_ADM_s116
                                 + 1.25529 w_k1_AXP_s116
                                 + 1.12764 w_k1_DIS_s116
                                 + 1.02873 w_k1_FDX_s116
                                 + 1.33888 w_k1_IBM_s116
                                 + 1.16395 w_k1_INTC_s116 + 1.2845 w_k1_KO_s116
                                 + 1.18209 w_k1_MAT_s116
                                 + 1.04449 w_k1_MSFT_s116
                                 + 1.11244 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.935187 w_k1_ADM_s117
                                 + 0.933186 w_k1_AXP_s117
                                 + 0.676571 w_k1_DIS_s117
                                 + 0.955351 w_k1_FDX_s117
                                 + 0.947085 w_k1_IBM_s117
                                 + 0.983911 w_k1_INTC_s117
                                 + 0.762289 w_k1_KO_s117
                                 + 0.949796 w_k1_MAT_s117
                                 + 0.749482 w_k1_MSFT_s117
                                 + 0.860026 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.91374 w_k1_ADM_s117
                                 + 0.74244 w_k1_AXP_s117
                                 + 0.880825 w_k1_DIS_s117
                                 + 0.909596 w_k1_FDX_s117
                                 + 0.756691 w_k1_IBM_s117
                                 + 0.835802 w_k1_INTC_s117
                                 + 0.906419 w_k1_KO_s117
                                 + 0.907623 w_k1_MAT_s117
                                 + 0.993449 w_k1_MSFT_s117
                                 + 0.871859 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.26649 w_k1_ADM_s117
                                 + 1.36811 w_k1_AXP_s117
                                 + 1.00222 w_k1_DIS_s117
                                 + 1.25795 w_k1_FDX_s117
                                 + 0.841486 w_k1_IBM_s117
                                 + 1.19648 w_k1_INTC_s117
                                 + 1.05865 w_k1_KO_s117 + 1.2709 w_k1_MAT_s117
                                 + 1.09766 w_k1_MSFT_s117 + 1.0559 w_k1_PG_s117
                                 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.02628 w_k1_ADM_s117
                                 + 1.01967 w_k1_AXP_s117
                                 + 1.22128 w_k1_DIS_s117
                                 + 1.13843 w_k1_FDX_s117
                                 + 0.923963 w_k1_IBM_s117
                                 + 1.29041 w_k1_INTC_s117
                                 + 1.06454 w_k1_KO_s117
                                 + 0.847448 w_k1_MAT_s117
                                 + 1.24382 w_k1_MSFT_s117
                                 + 1.13583 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.36297 w_k1_ADM_s117
                                 + 1.18827 w_k1_AXP_s117
                                 + 0.899309 w_k1_DIS_s117
                                 + 1.05036 w_k1_FDX_s117
                                 + 0.957822 w_k1_IBM_s117
                                 + 1.26233 w_k1_INTC_s117
                                 + 0.997369 w_k1_KO_s117
                                 + 1.20543 w_k1_MAT_s117
                                 + 1.04818 w_k1_MSFT_s117
                                 + 1.03719 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.29434 w_k1_ADM_s117
                                 + 1.64119 w_k1_AXP_s117
                                 + 1.46007 w_k1_DIS_s117
                                 + 1.21925 w_k1_FDX_s117
                                 + 1.47525 w_k1_IBM_s117
                                 + 1.26239 w_k1_INTC_s117
                                 + 1.38747 w_k1_KO_s117 + 2.10146 w_k1_MAT_s117
                                 + 1.29285 w_k1_MSFT_s117
                                 + 1.23494 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.785859 w_k1_ADM_s117
                                 + 0.915816 w_k1_AXP_s117
                                 + 0.632971 w_k1_DIS_s117
                                 + 0.798305 w_k1_FDX_s117
                                 + 0.753724 w_k1_IBM_s117
                                 + 0.995175 w_k1_INTC_s117
                                 + 0.753516 w_k1_KO_s117
                                 + 0.966172 w_k1_MAT_s117
                                 + 0.871507 w_k1_MSFT_s117
                                 + 0.965137 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.03684 w_k1_ADM_s117
                                 + 1.1678 w_k1_AXP_s117
                                 + 0.951281 w_k1_DIS_s117
                                 + 1.02619 w_k1_FDX_s117
                                 + 1.42774 w_k1_IBM_s117
                                 + 0.98796 w_k1_INTC_s117
                                 + 1.19536 w_k1_KO_s117 + 1.6871 w_k1_MAT_s117
                                 + 0.890737 w_k1_MSFT_s117
                                 + 1.03128 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0283109352494897 w_k1_ADM_s000
                                 + 0.0173332194913583 w_k1_ADM_s001
                                 + 0.0153798919406821 w_k1_ADM_s002
                                 + 0.0098686437746165 w_k1_ADM_s003
                                 + 0.0373080580004994 w_k1_ADM_s004
                                 + 0.0355463219947053 w_k1_ADM_s005
                                 + 0.0358306798547795 w_k1_ADM_s006
                                 + 0.0282715933407194 w_k1_ADM_s007
                                 + 0.00851920965620009 w_k1_ADM_s010
                                 + 0.0214882040966876 w_k1_ADM_s011
                                 + 0.026271500880774 w_k1_ADM_s012
                                 + 0.0298873795743958 w_k1_ADM_s013
                                 + 0.0160089655561964 w_k1_ADM_s014
                                 + 0.015511203191888 w_k1_ADM_s015
                                 + 0.00770834009734826 w_k1_ADM_s016
                                 + 0.0282159186977285 w_k1_ADM_s017
                                 + 0.0597081700903551 w_k1_ADM_s100
                                 + 0.0508593213903315 w_k1_ADM_s101
                                 + 0.0364240243337679 w_k1_ADM_s102
                                 + 0.0402893109103118 w_k1_ADM_s103
                                 + 0.101339993846327 w_k1_ADM_s104
                                 + 0.0668571331705042 w_k1_ADM_s105
                                 + 0.138931468145322 w_k1_ADM_s106
                                 + 0.0337277929699204 w_k1_ADM_s107
                                 + 0.0245621981112092 w_k1_ADM_s110
                                 + 0.018130399846785 w_k1_ADM_s111
                                 + 0.0189864121788909 w_k1_ADM_s112
                                 + 0.0120355663468845 w_k1_ADM_s113
                                 + 0.0154094800122592 w_k1_ADM_s114
                                 + 0.012374938683677 w_k1_ADM_s115
                                 + 0.00471003066758734 w_k1_ADM_s116
                                 + 0.0113092178275535 w_k1_ADM_s117
                                 + 0.0319610811439584 w_k1_AXP_s000
                                 + 0.0207228027961152 w_k1_AXP_s001
                                 + 0.0163795577874922 w_k1_AXP_s002
                                 + 0.0106230082745313 w_k1_AXP_s003
                                 + 0.0374437096123959 w_k1_AXP_s004
                                 + 0.0381918536907314 w_k1_AXP_s005
                                 + 0.037809723239137 w_k1_AXP_s006
                                 + 0.0324098659194059 w_k1_AXP_s007
                                 + 0.00836778093034876 w_k1_AXP_s010
                                 + 0.0213350217795857 w_k1_AXP_s011
                                 + 0.0272436376168939 w_k1_AXP_s012
                                 + 0.0330785052221876 w_k1_AXP_s013
                                 + 0.0141967080853244 w_k1_AXP_s014
                                 + 0.0157101425056833 w_k1_AXP_s015
                                 + 0.00828687875776858 w_k1_AXP_s016
                                 + 0.025893408546566 w_k1_AXP_s017
                                 + 0.0584733726849812 w_k1_AXP_s100
                                 + 0.0471552811392122 w_k1_AXP_s101
                                 + 0.0404587557347994 w_k1_AXP_s102
                                 + 0.0421100510132431 w_k1_AXP_s103
                                 + 0.111197980909589 w_k1_AXP_s104
                                 + 0.06829443859796 w_k1_AXP_s105
                                 + 0.137806729190883 w_k1_AXP_s106
                                 + 0.0356391200809835 w_k1_AXP_s107
                                 + 0.0219347983007593 w_k1_AXP_s110
                                 + 0.0195930590187818 w_k1_AXP_s111
                                 + 0.0194100920285367 w_k1_AXP_s112
                                 + 0.0153556101671921 w_k1_AXP_s113
                                 + 0.0152213637300931 w_k1_AXP_s114
                                 + 0.0130548508942097 w_k1_AXP_s115
                                 + 0.00660967155774064 w_k1_AXP_s116
                                 + 0.0122056086089074 w_k1_AXP_s117
                                 + 0.0318806711273272 w_k1_DIS_s000
                                 + 0.021562195279469 w_k1_DIS_s001
                                 + 0.0144569655436325 w_k1_DIS_s002
                                 + 0.00990668825495604 w_k1_DIS_s003
                                 + 0.0350455286495294 w_k1_DIS_s004
                                 + 0.035423603673294 w_k1_DIS_s005
                                 + 0.0356625033437083 w_k1_DIS_s006
                                 + 0.0358677868379265 w_k1_DIS_s007
                                 + 0.00802366431667579 w_k1_DIS_s010
                                 + 0.0239292679356063 w_k1_DIS_s011
                                 + 0.0260657511817336 w_k1_DIS_s012
                                 + 0.0311495258870354 w_k1_DIS_s013
                                 + 0.0155583299985267 w_k1_DIS_s014
                                 + 0.0161434474263916 w_k1_DIS_s015
                                 + 0.0078272103868776 w_k1_DIS_s016
                                 + 0.0279160858156869 w_k1_DIS_s017
                                 + 0.0574717783661942 w_k1_DIS_s100
                                 + 0.0458889107159978 w_k1_DIS_s101
                                 + 0.0361285631593965 w_k1_DIS_s102
                                 + 0.0432647252327822 w_k1_DIS_s103
                                 + 0.122043741341664 w_k1_DIS_s104
                                 + 0.0715717669341224 w_k1_DIS_s105
                                 + 0.134455976093822 w_k1_DIS_s106
                                 + 0.0375332903473525 w_k1_DIS_s107
                                 + 0.0230697198560793 w_k1_DIS_s110
                                 + 0.0175102229873375 w_k1_DIS_s111
                                 + 0.0198278912739575 w_k1_DIS_s112
                                 + 0.0164196976202074 w_k1_DIS_s113
                                 + 0.0161980733637386 w_k1_DIS_s114
                                 + 0.0138005776875154 w_k1_DIS_s115
                                 + 0.00568887604386189 w_k1_DIS_s116
                                 + 0.010558499374136 w_k1_DIS_s117
                                 + 0.0317626745112403 w_k1_FDX_s000
                                 + 0.0205073830505059 w_k1_FDX_s001
                                 + 0.0147600803702952 w_k1_FDX_s002
                                 + 0.00987053855661205 w_k1_FDX_s003
                                 + 0.0354956996767366 w_k1_FDX_s004
                                 + 0.032631259797412 w_k1_FDX_s005
                                 + 0.0352937913128695 w_k1_FDX_s006
                                 + 0.0323032300130836 w_k1_FDX_s007
                                 + 0.00770489326515136 w_k1_FDX_s010
                                 + 0.0214408992011888 w_k1_FDX_s011
                                 + 0.0266475353943621 w_k1_FDX_s012
                                 + 0.031558858296303 w_k1_FDX_s013
                                 + 0.0144487917312534 w_k1_FDX_s014
                                 + 0.0154140309809478 w_k1_FDX_s015
                                 + 0.00835305858974744 w_k1_FDX_s016
                                 + 0.0261763889812329 w_k1_FDX_s017
                                 + 0.0570313590213451 w_k1_FDX_s100
                                 + 0.0461093008188599 w_k1_FDX_s101
                                 + 0.0353045482986725 w_k1_FDX_s102
                                 + 0.0408163917594717 w_k1_FDX_s103
                                 + 0.111718667170107 w_k1_FDX_s104
                                 + 0.0666692843683942 w_k1_FDX_s105
                                 + 0.131829190113446 w_k1_FDX_s106
                                 + 0.0352748976990917 w_k1_FDX_s107
                                 + 0.0221120892313634 w_k1_FDX_s110
                                 + 0.0181752500394388 w_k1_FDX_s111
                                 + 0.018334191639612 w_k1_FDX_s112
                                 + 0.0147821465653084 w_k1_FDX_s113
                                 + 0.0156855755859008 w_k1_FDX_s114
                                 + 0.0121691570166043 w_k1_FDX_s115
                                 + 0.0058026900855257 w_k1_FDX_s116
                                 + 0.0111966088637272 w_k1_FDX_s117
                                 + 0.0310227409965041 w_k1_IBM_s000
                                 + 0.0207230799724862 w_k1_IBM_s001
                                 + 0.015298866275878 w_k1_IBM_s002
                                 + 0.00986443177085296 w_k1_IBM_s003
                                 + 0.0353597012608115 w_k1_IBM_s004
                                 + 0.0341403132493368 w_k1_IBM_s005
                                 + 0.035871076994917 w_k1_IBM_s006
                                 + 0.0347715168274763 w_k1_IBM_s007
                                 + 0.00780342821586717 w_k1_IBM_s010
                                 + 0.0224654808287054 w_k1_IBM_s011
                                 + 0.0277823222459326 w_k1_IBM_s012
                                 + 0.0335526987558753 w_k1_IBM_s013
                                 + 0.0146304199274088 w_k1_IBM_s014
                                 + 0.01618926672387 w_k1_IBM_s015
                                 + 0.00827409697611747 w_k1_IBM_s016
                                 + 0.0269016516742811 w_k1_IBM_s017
                                 + 0.0568346516496898 w_k1_IBM_s100
                                 + 0.0479431792202728 w_k1_IBM_s101
                                 + 0.0352776160344134 w_k1_IBM_s102
                                 + 0.0401628170445293 w_k1_IBM_s103
                                 + 0.113065795239858 w_k1_IBM_s104
                                 + 0.068283166865017 w_k1_IBM_s105
                                 + 0.134369626889388 w_k1_IBM_s106
                                 + 0.0348582705567177 w_k1_IBM_s107
                                 + 0.0231628608943739 w_k1_IBM_s110
                                 + 0.0181111893468148 w_k1_IBM_s111
                                 + 0.0185902124843392 w_k1_IBM_s112
                                 + 0.0150586582424949 w_k1_IBM_s113
                                 + 0.0160476572257276 w_k1_IBM_s114
                                 + 0.0136434197069699 w_k1_IBM_s115
                                 + 0.00580030277722037 w_k1_IBM_s116
                                 + 0.0109764973787388 w_k1_IBM_s117
                                 + 0.0319409858268444 w_k1_INTC_s000
                                 + 0.021277741376766 w_k1_INTC_s001
                                 + 0.0145352058779764 w_k1_INTC_s002
                                 + 0.0103477488356301 w_k1_INTC_s003
                                 + 0.0414188080860828 w_k1_INTC_s004
                                 + 0.0337420692475079 w_k1_INTC_s005
                                 + 0.0307868982707575 w_k1_INTC_s006
                                 + 0.0336375025873071 w_k1_INTC_s007
                                 + 0.00784341302681183 w_k1_INTC_s010
                                 + 0.0180956454074718 w_k1_INTC_s011
                                 + 0.0243426214263567 w_k1_INTC_s012
                                 + 0.0343425266785573 w_k1_INTC_s013
                                 + 0.0136066481491638 w_k1_INTC_s014
                                 + 0.0142859400008162 w_k1_INTC_s015
                                 + 0.00837369656018805 w_k1_INTC_s016
                                 + 0.0298769987029365 w_k1_INTC_s017
                                 + 0.0604645628281432 w_k1_INTC_s100
                                 + 0.0536862992221787 w_k1_INTC_s101
                                 + 0.0372253202960702 w_k1_INTC_s102
                                 + 0.0440915748952071 w_k1_INTC_s103
                                 + 0.127265316930212 w_k1_INTC_s104
                                 + 0.0753909513261013 w_k1_INTC_s105
                                 + 0.149183528939908 w_k1_INTC_s106
                                 + 0.0312958909500172 w_k1_INTC_s107
                                 + 0.02251535450566 w_k1_INTC_s110
                                 + 0.0179432417592371 w_k1_INTC_s111
                                 + 0.0175321409298446 w_k1_INTC_s112
                                 + 0.0158406558320867 w_k1_INTC_s113
                                 + 0.0172269423663124 w_k1_INTC_s114
                                 + 0.0114398137560684 w_k1_INTC_s115
                                 + 0.00620005519906845 w_k1_INTC_s116
                                 + 0.0119403690976661 w_k1_INTC_s117
                                 + 0.0307448061220466 w_k1_KO_s000
                                 + 0.0195670937036456 w_k1_KO_s001
                                 + 0.0137299691456348 w_k1_KO_s002
                                 + 0.00861396722538697 w_k1_KO_s003
                                 + 0.0347693217089127 w_k1_KO_s004
                                 + 0.0329464102849202 w_k1_KO_s005
                                 + 0.0352982326586427 w_k1_KO_s006
                                 + 0.0350313767153055 w_k1_KO_s007
                                 + 0.00757646690065672 w_k1_KO_s010
                                 + 0.0191800690064396 w_k1_KO_s011
                                 + 0.0265078711551802 w_k1_KO_s012
                                 + 0.0337931424255737 w_k1_KO_s013
                                 + 0.0134148714929221 w_k1_KO_s014
                                 + 0.0152579191698696 w_k1_KO_s015
                                 + 0.00771727103228165 w_k1_KO_s016
                                 + 0.0262413037090464 w_k1_KO_s017
                                 + 0.0593017378983693 w_k1_KO_s100
                                 + 0.0490634007622794 w_k1_KO_s101
                                 + 0.0346187721830438 w_k1_KO_s102
                                 + 0.040244358097042 w_k1_KO_s103
                                 + 0.124724192783239 w_k1_KO_s104
                                 + 0.0603773530986929 w_k1_KO_s105
                                 + 0.137550835812905 w_k1_KO_s106
                                 + 0.0333241383021341 w_k1_KO_s107
                                 + 0.0241636749165703 w_k1_KO_s110
                                 + 0.0174111830728577 w_k1_KO_s111
                                 + 0.0162216385516053 w_k1_KO_s112
                                 + 0.0151095351503915 w_k1_KO_s113
                                 + 0.0165524407432683 w_k1_KO_s114
                                 + 0.0131256707431381 w_k1_KO_s115
                                 + 0.00597072892573056 w_k1_KO_s116
                                 + 0.0110079228445689 w_k1_KO_s117
                                 + 0.03431216109182 w_k1_MAT_s000
                                 + 0.0230769283294853 w_k1_MAT_s001
                                 + 0.0167670908296565 w_k1_MAT_s002
                                 + 0.00644784897543575 w_k1_MAT_s003
                                 + 0.0309617424040492 w_k1_MAT_s004
                                 + 0.0347853688626511 w_k1_MAT_s005
                                 + 0.0334045005173427 w_k1_MAT_s006
                                 + 0.0375816856826341 w_k1_MAT_s007
                                 + 0.00768698855567864 w_k1_MAT_s010
                                 + 0.0238574016741555 w_k1_MAT_s011
                                 + 0.0314316126219726 w_k1_MAT_s012
                                 + 0.0309283099432087 w_k1_MAT_s013
                                 + 0.0143728689990595 w_k1_MAT_s014
                                 + 0.0170744741060669 w_k1_MAT_s015
                                 + 0.00922091285717365 w_k1_MAT_s016
                                 + 0.0269787886311985 w_k1_MAT_s017
                                 + 0.0611006112480294 w_k1_MAT_s100
                                 + 0.0394884671945183 w_k1_MAT_s101
                                 + 0.0386855416849787 w_k1_MAT_s102
                                 + 0.0381102606709341 w_k1_MAT_s103
                                 + 0.113111738061704 w_k1_MAT_s104
                                 + 0.0690720845673388 w_k1_MAT_s105
                                 + 0.140258661893891 w_k1_MAT_s106
                                 + 0.0426902454115834 w_k1_MAT_s107
                                 + 0.0309298285341912 w_k1_MAT_s110
                                 + 0.0221129468079649 w_k1_MAT_s111
                                 + 0.0168694735490899 w_k1_MAT_s112
                                 + 0.0171066909305966 w_k1_MAT_s113
                                 + 0.0192298473201243 w_k1_MAT_s114
                                 + 0.0165215653768006 w_k1_MAT_s115
                                 + 0.00553999906996579 w_k1_MAT_s116
                                 + 0.013575533057606 w_k1_MAT_s117
                                 + 0.0294959479184798 w_k1_MSFT_s000
                                 + 0.0201208669271018 w_k1_MSFT_s001
                                 + 0.0144151483514824 w_k1_MSFT_s002
                                 + 0.00986777195716688 w_k1_MSFT_s003
                                 + 0.0376801465036867 w_k1_MSFT_s004
                                 + 0.0339261919640295 w_k1_MSFT_s005
                                 + 0.0358167548324579 w_k1_MSFT_s006
                                 + 0.0334976256578505 w_k1_MSFT_s007
                                 + 0.00813981938035429 w_k1_MSFT_s010
                                 + 0.0246849688984177 w_k1_MSFT_s011
                                 + 0.0237045179747462 w_k1_MSFT_s012
                                 + 0.03097336370682 w_k1_MSFT_s013
                                 + 0.0152737593591953 w_k1_MSFT_s014
                                 + 0.0155362098447886 w_k1_MSFT_s015
                                 + 0.00781544736554212 w_k1_MSFT_s016
                                 + 0.0259171977299258 w_k1_MSFT_s017
                                 + 0.0556630063473947 w_k1_MSFT_s100
                                 + 0.0485662109418812 w_k1_MSFT_s101
                                 + 0.0345098477789543 w_k1_MSFT_s102
                                 + 0.0417222137513278 w_k1_MSFT_s103
                                 + 0.118249875839982 w_k1_MSFT_s104
                                 + 0.0655248834230232 w_k1_MSFT_s105
                                 + 0.134112971101843 w_k1_MSFT_s106
                                 + 0.0345035854602633 w_k1_MSFT_s107
                                 + 0.0227884586244933 w_k1_MSFT_s110
                                 + 0.017582800013133 w_k1_MSFT_s111
                                 + 0.0166953994513164 w_k1_MSFT_s112
                                 + 0.0155052826641648 w_k1_MSFT_s113
                                 + 0.0169925486134844 w_k1_MSFT_s114
                                 + 0.0139736922232682 w_k1_MSFT_s115
                                 + 0.00548294872877703 w_k1_MSFT_s116
                                 + 0.0111763283484924 w_k1_MSFT_s117
                                 + 0.0293855118387923 w_k1_PG_s000
                                 + 0.0203971386043588 w_k1_PG_s001
                                 + 0.0146293438690257 w_k1_PG_s002
                                 + 0.0102450911556736 w_k1_PG_s003
                                 + 0.0357359441838671 w_k1_PG_s004
                                 + 0.0346379329487381 w_k1_PG_s005
                                 + 0.0344291204212467 w_k1_PG_s006
                                 + 0.0321866617816176 w_k1_PG_s007
                                 + 0.0075658748563901 w_k1_PG_s010
                                 + 0.0202279837027796 w_k1_PG_s011
                                 + 0.0262528227744789 w_k1_PG_s012
                                 + 0.0309605699486844 w_k1_PG_s013
                                 + 0.0137553854575228 w_k1_PG_s014
                                 + 0.0162574285257449 w_k1_PG_s015
                                 + 0.00852308752722247 w_k1_PG_s016
                                 + 0.0253207467405435 w_k1_PG_s017
                                 + 0.0579453334361161 w_k1_PG_s100
                                 + 0.0468508495186932 w_k1_PG_s101
                                 + 0.0319132817114269 w_k1_PG_s102
                                 + 0.0393640881554283 w_k1_PG_s103
                                 + 0.115283512386463 w_k1_PG_s104
                                 + 0.0722347446248716 w_k1_PG_s105
                                 + 0.132061817830137 w_k1_PG_s106
                                 + 0.0383256741863547 w_k1_PG_s107
                                 + 0.0230569006451949 w_k1_PG_s110
                                 + 0.0182331903615449 w_k1_PG_s111
                                 + 0.0189219485543084 w_k1_PG_s112
                                 + 0.0149697951075705 w_k1_PG_s113
                                 + 0.0161709497702236 w_k1_PG_s114
                                 + 0.0131778998495066 w_k1_PG_s115
                                 + 0.00560388126442582 w_k1_PG_s116
                                 + 0.011228009456799 w_k1_PG_s117 >= 1.041
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
