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
 w_k1_ADM_s0_return_constraint:  1.13424 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.960224 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.11142 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.852923 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.12791 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.929314 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.934824 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.552351 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.931321 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.07298 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.862382 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.789178 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.05149 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.964985 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.10729 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.961644 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.07894 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.48718 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.02045 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.950505 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.16135 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.792443 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.08275 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.921163 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.86948 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.24861 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.0186 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.98232 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.17025 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.08031 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.954617 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.880097 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.10188 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.21661 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.691182 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.01779 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.948478 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.856867 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.22743 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.908005 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.15427 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.789447 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.31498 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.909274 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.778988 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.551115 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.06875 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.36529 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.732732 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.831778 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.16979 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.959695 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.06068 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.844993 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.756286 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.48906 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.12518 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.647257 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.24367 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.659259 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.965104 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.901028 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.991431 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.63353 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.747982 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.960633 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.23623 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.00549 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.06565 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.819071 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.950744 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.40813 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.541331 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.21789 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.921072 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.78728 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.38609 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.886473 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.11592 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.719803 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.483 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.855986 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.73557 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.505726 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.969174 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.64875 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.743543 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.745688 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.34101 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.897466 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.16941 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.820736 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.94697 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.98887 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.929306 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.657452 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.32955 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.668639 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.826991 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.856978 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.945576 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.52387 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.644157 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.699006 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.31509 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.21772 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.05954 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.761673 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.09413 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.60766 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.545719 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.34655 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.981058 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.691184 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.21876 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.912278 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.05757 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.86279 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.30308 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.927453 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.958221 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.623637 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.963597 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.2439 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.767388 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.738421 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.28175 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.98359 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.14314 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.839793 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.903922 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.58453 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.831029 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.863885 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.17182 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.698892 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.829997 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.96417 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.828876 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.18728 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.840647 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.791643 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.16892 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.2075 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.02165 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.772231 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.1489 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.44435 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.699369 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.09455 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.985308 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.887176 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.18662 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.94156 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.15862 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.677317 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.24387 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.95853 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.603598 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.70402 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.69194 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.35111 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.841789 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.25703 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.847736 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.943913 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.813026 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.816883 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.14106 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.01952 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.45134 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.581763 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.29415 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.1084 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.11797 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.765396 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.71939 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.67931 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.543302 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.622105 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.02421 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.586174 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.01544 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.14019 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.733381 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.925566 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.56251 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.57954 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.652273 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.611237 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.16431 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.00022 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.01971 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.938521 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.16217 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.82501 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.955424 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.752873 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.01436 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.34087 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.774829 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.694913 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.18163 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.12304 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.90952 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.818829 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.876788 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.39265 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.732954 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.771212 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.0726 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.614616 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.819595 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.830262 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.788123 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.17498 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.775567 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.786056 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.18373 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.13013 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.01303 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.868545 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.12102 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.34273 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.744664 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.24497 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.961626 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.936372 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.26055 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.933385 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.12036 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.714112 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.37299 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.906062 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.68727 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.591076 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.29752 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.3532 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.823788 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.897101 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.03818 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.900787 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.96496 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.778262 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.05408 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.35921 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.09026 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.618189 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.2762 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.822546 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.856537 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.824764 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.01573 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.36984 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.596374 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.645603 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.05256 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.686365 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.996424 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.962513 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.980709 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.24332 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.681983 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.23682 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.837829 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.734851 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.19332 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.964473 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.18145 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.660918 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.19813 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.771407 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.632495 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.569649 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.91366 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.65472 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.876764 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.37063 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.928584 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.981605 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.96331 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.760324 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.01944 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.02739 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.35924 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.439164 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.35882 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.982126 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.07921 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.877639 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.58117 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 2.08632 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.693966 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.730029 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.2126 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.511525 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.62493 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.26067 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.889537 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.1821 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.654967 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.87326 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.737121 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.921373 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.15487 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.929717 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.05449 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.902798 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.25375 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.873379 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.952324 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.674044 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.992478 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.32628 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.799319 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.72836 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.3385 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.06643 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.961613 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.888481 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.898347 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.48446 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.82142 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.806854 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.11866 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.684216 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.843291 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.898352 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.890184 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.18383 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.757023 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.791405 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.13639 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.11538 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.05787 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.833528 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.07873 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.28899 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.79447 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.21165 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.0289 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.893406 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.14901 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.928785 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.03565 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.818553 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.18126 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.933899 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.83309 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.749912 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.08146 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.30599 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.824622 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.919926 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.983829 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.980965 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.05457 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.836734 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.964488 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.36312 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.991787 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.847046 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.17898 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.915984 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.914654 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.928728 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.04388 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.39254 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.868222 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.827431 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.17546 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.981658 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.0141 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.941047 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.03552 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.14294 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.719731 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.15069 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.954264 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.900839 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.266000719886738 z_k1_s0000
                                 - 0.498002420576923 z_k1_s0001
                                 - 0.16000048314631 z_k1_s0002
                                 - 0.435002355354068 z_k1_s0003
                                 - 0.522002826424882 z_k1_s0004
                                 - 0.168000440628423 z_k1_s0005
                                 - 0.370002034059875 z_k1_s0006
                                 - 0.248000815551983 z_k1_s0007
                                 - 0.0870000589765303 z_k1_s0010
                                 - 0.108999958085543 z_k1_s0011
                                 - 0.322000236959111 z_k1_s0012
                                 - 0.311999987364118 z_k1_s0013
                                 - 0.307999237526147 z_k1_s0014
                                 - 0.164999243317593 z_k1_s0015
                                 - 0.199000741940545 z_k1_s0016
                                 - 0.122999982518547 z_k1_s0017
                                 - 0.313998645781762 z_k1_s0020
                                 - 0.257999945368973 z_k1_s0021
                                 - 0.625999862989837 z_k1_s0022
                                 - 0.305000158207348 z_k1_s0023
                                 - 0.305000158207348 z_k1_s0024
                                 - 0.393999252655417 z_k1_s0025
                                 - 0.500000918962394 z_k1_s0026
                                 - 0.171999867779384 z_k1_s0027
                                 - 0.363001110901004 z_k1_s0030
                                 - 0.322000808618025 z_k1_s0031
                                 - 0.83100279652615 z_k1_s0032
                                 - 1.11900320490909 z_k1_s0033
                                 - 0.263000583011399 z_k1_s0034
                                 - 0.39800102994162 z_k1_s0035
                                 - 0.364001038507682 z_k1_s0036
                                 - 0.428002537352288 z_k1_s0037
                                 - 0.690000129581618 z_k1_s0040
                                 - 0.191000014240703 z_k1_s0041
                                 - 0.363000279053864 z_k1_s0042
                                 - 0.283000155884952 z_k1_s0043
                                 - 0.804000864747465 z_k1_s0044
                                 - 0.559000953917796 z_k1_s0045
                                 - 0.421998597665467 z_k1_s0046
                                 - 0.418998872871707 z_k1_s0047
                                 - 0.127999981467433 z_k1_s0050
                                 - 0.250000207459744 z_k1_s0051
                                 - 0.261999447210084 z_k1_s0052
                                 - 0.151000038521716 z_k1_s0053
                                 - 0.323999813251989 z_k1_s0054
                                 - 0.0770000259296146 z_k1_s0055
                                 - 0.0829999170046888 z_k1_s0056
                                 - 0.0809999533129974 z_k1_s0057
                                 - 1.51500110688537 z_k1_s0060
                                 - 0.368999592830021 z_k1_s0061
                                 - 1.22899679419462 z_k1_s0062
                                 - 0.464999638088123 z_k1_s0063
                                 - 0.346999802345705 z_k1_s0064
                                 - 0.3879998687934 z_k1_s0065
                                 - 0.33499973386349 z_k1_s0066
                                 - 0.377999644191682 z_k1_s0067
                                 - 0.256000586278226 z_k1_s0070
                                 - 0.584000718361241 z_k1_s0071
                                 - 0.430002453514068 z_k1_s0072
                                 - 0.582003471240356 z_k1_s0073
                                 - 0.965001886237405 z_k1_s0074
                                 - 0.23700045906244 z_k1_s0075
                                 - 0.221000649839384 z_k1_s0076
                                 - 0.768002163135715 z_k1_s0077
                                 - 0.192000013482841 z_k1_s0100
                                 - 0.223000851378801 z_k1_s0101
                                 - 0.230999856190236 z_k1_s0102
                                 - 0.247999875165286 z_k1_s0103
                                 - 0.277000108358088 z_k1_s0104
                                 - 0.163000121690159 z_k1_s0105
                                 - 0.131999966594438 z_k1_s0106
                                 - 0.24099946570483 z_k1_s0107
                                 - 0.17099955778934 z_k1_s0110
                                 - 0.186998928899529 z_k1_s0111
                                 - 0.248999852569391 z_k1_s0112
                                 - 0.121999768080522 z_k1_s0113
                                 - 0.214998889590111 z_k1_s0114
                                 - 0.063999889935659 z_k1_s0115
                                 - 0.198999518479922 z_k1_s0116
                                 - 0.207998899417465 z_k1_s0117
                                 - 0.0450000038507124 z_k1_s0120
                                 - 0.0700001882123968 z_k1_s0121
                                 - 0.0340000211316671 z_k1_s0122
                                 - 0.0919999896503275 z_k1_s0123
                                 - 0.0190000335147763 z_k1_s0124
                                 - 0.0380000260295126 z_k1_s0125
                                 - 0.0589998774930319 z_k1_s0126
                                 - 0.0529999316463235 z_k1_s0127
                                 - 0.0619999201140232 z_k1_s0130
                                 - 0.0819998943443532 z_k1_s0131
                                 - 0.0839998917673862 z_k1_s0132
                                 - 0.0270000114109322 z_k1_s0133
                                 - 0.0419999920836798 z_k1_s0134
                                 - 0.0879998866134522 z_k1_s0135
                                 - 0.0320000049685147 z_k1_s0136
                                 - 0.0449999882182293 z_k1_s0137
                                 - 0.0749997762369516 z_k1_s0140
                                 - 0.193999914318347 z_k1_s0141
                                 - 0.038999898363178 z_k1_s0142
                                 - 0.038999898363178 z_k1_s0143
                                 - 0.169999934402651 z_k1_s0144
                                 - 0.0769998512297343 z_k1_s0145
                                 - 0.14199980450139 z_k1_s0146
                                 - 0.183999539354433 z_k1_s0147
                                 - 0.0590000291367299 z_k1_s0150
                                 - 0.0470000003716279 z_k1_s0151
                                 - 0.0940000007432559 z_k1_s0152
                                 - 0.0750002471568996 z_k1_s0153
                                 - 0.0440000336053601 z_k1_s0154
                                 - 0.0949998459319847 z_k1_s0155
                                 - 0.050000021037906 z_k1_s0156
                                 - 0.0750002471568996 z_k1_s0157
                                 - 0.167999688177336 z_k1_s0160
                                 - 0.127999729468498 z_k1_s0161
                                 - 0.0969999085189167 z_k1_s0162
                                 - 0.471998613165704 z_k1_s0163
                                 - 0.34699939094956 z_k1_s0164
                                 - 0.128999840886041 z_k1_s0165
                                 - 0.233999429747218 z_k1_s0166
                                 - 0.154999796746813 z_k1_s0167
                                 - 0.0809998289610357 z_k1_s0170
                                 - 0.0660002957462384 z_k1_s0171
                                 - 0.0719997886320032 z_k1_s0172
                                 - 0.0630002823032275 z_k1_s0173
                                 - 0.0609999173410491 z_k1_s0174
                                 - 0.101000274581279 z_k1_s0175
                                 - 0.0590000863791275 z_k1_s0176
                                 - 0.0309999965110435 z_k1_s0177
                                 - 0.525002009736584 z_k1_s1000
                                 - 0.563003788389627 z_k1_s1001
                                 - 0.713000944302972 z_k1_s1002
                                 - 0.459000389765142 z_k1_s1003
                                 - 0.365001321283276 z_k1_s1004
                                 - 0.202000849232234 z_k1_s1005
                                 - 0.348001218225013 z_k1_s1006
                                 - 0.813004365596258 z_k1_s1007
                                 - 0.213000088520405 z_k1_s1010
                                 - 0.593000270375117 z_k1_s1011
                                 - 0.403000375497758 z_k1_s1012
                                 - 0.437999221106758 z_k1_s1013
                                 - 0.424001251263141 z_k1_s1014
                                 - 0.64000129685003 z_k1_s1015
                                 - 0.900998737934218 z_k1_s1016
                                 - 0.308999890947913 z_k1_s1017
                                 - 0.132000145988454 z_k1_s1020
                                 - 0.066000072994227 z_k1_s1021
                                 - 0.0970000424974985 z_k1_s1022
                                 - 0.139000227766718 z_k1_s1023
                                 - 0.268999695960599 z_k1_s1024
                                 - 0.268999695960599 z_k1_s1025
                                 - 0.265000478174012 z_k1_s1026
                                 - 0.290000552096123 z_k1_s1027
                                 - 0.182000396362503 z_k1_s1030
                                 - 0.115000161143173 z_k1_s1031
                                 - 0.262001134837188 z_k1_s1032
                                 - 0.335000252594922 z_k1_s1033
                                 - 0.260000830089784 z_k1_s1034
                                 - 0.118000248863586 z_k1_s1035
                                 - 0.172000350228254 z_k1_s1036
                                 - 0.403001378989339 z_k1_s1037
                                 - 0.0650000998500032 z_k1_s1040
                                 - 0.284000368790684 z_k1_s1041
                                 - 0.144000332344786 z_k1_s1042
                                 - 0.19500055755039 z_k1_s1043
                                 - 0.0870001792916101 z_k1_s1044
                                 - 0.288000664689573 z_k1_s1045
                                 - 0.132000734650022 z_k1_s1046
                                 - 0.095000126088437 z_k1_s1047
                                 - 0.122000344050721 z_k1_s1050
                                 - 0.177999824466957 z_k1_s1051
                                 - 0.100999656893996 z_k1_s1052
                                 - 0.0650000197121515 z_k1_s1053
                                 - 0.12400021749408 z_k1_s1054
                                 - 0.230000565997202 z_k1_s1055
                                 - 0.116999652441596 z_k1_s1056
                                 - 0.0710000432425196 z_k1_s1057
                                 - 0.461999339902038 z_k1_s1060
                                 - 0.199000428234215 z_k1_s1061
                                 - 0.845003546400225 z_k1_s1062
                                 - 0.298000417784931 z_k1_s1063
                                 - 0.88600270380328 z_k1_s1064
                                 - 0.415000697209126 z_k1_s1065
                                 - 0.509999807230665 z_k1_s1066
                                 - 0.970000082871046 z_k1_s1067
                                 - 0.153000512460168 z_k1_s1070
                                 - 0.0720000775692849 z_k1_s1071
                                 - 0.130000583528433 z_k1_s1072
                                 - 0.071000049311344 z_k1_s1073
                                 - 0.0670000392797099 z_k1_s1074
                                 - 0.139999836106547 z_k1_s1075
                                 - 0.197000107807495 z_k1_s1076
                                 - 0.200000501581707 z_k1_s1077
                                 - 0.58199948497724 z_k1_s1100
                                 - 0.780002142379988 z_k1_s1101
                                 - 1.38800213490294 z_k1_s1102
                                 - 0.371999461736106 z_k1_s1103
                                 - 1.01599809376987 z_k1_s1104
                                 - 0.498999877656906 z_k1_s1105
                                 - 1.2399960251218 z_k1_s1106
                                 - 0.665000400696705 z_k1_s1107
                                 - 0.516999787717585 z_k1_s1110
                                 - 0.559000378291954 z_k1_s1111
                                 - 0.535000383506612 z_k1_s1112
                                 - 0.553999679728319 z_k1_s1113
                                 - 0.543999881801099 z_k1_s1114
                                 - 0.676000252920496 z_k1_s1115
                                 - 2.56199844383529 z_k1_s1116
                                 - 2.04899905504861 z_k1_s1117
                                 - 0.818001366714912 z_k1_s1120
                                 - 0.683000849192495 z_k1_s1121
                                 - 0.282000864423856 z_k1_s1122
                                 - 0.46100156867216 z_k1_s1123
                                 - 0.968004107969511 z_k1_s1124
                                 - 0.518999931667971 z_k1_s1125
                                 - 0.642999793256815 z_k1_s1126
                                 - 0.500001014153985 z_k1_s1127
                                 - 0.320000073573665 z_k1_s1130
                                 - 0.903002902277223 z_k1_s1131
                                 - 1.04000064508978 z_k1_s1132
                                 - 0.34900024770612 z_k1_s1133
                                 - 0.610998349113384 z_k1_s1134
                                 - 0.513000224516317 z_k1_s1135
                                 - 0.458000304906866 z_k1_s1136
                                 - 1.21900330647777 z_k1_s1137
                                 - 0.276999546053868 z_k1_s1140
                                 - 0.376998809129599 z_k1_s1141
                                 - 0.529999018337862 z_k1_s1142
                                 - 0.364000770743513 z_k1_s1143
                                 - 0.324999693593617 z_k1_s1144
                                 - 0.565999477092255 z_k1_s1145
                                 - 0.536999300928762 z_k1_s1146
                                 - 0.504998506369766 z_k1_s1147
                                 - 1.50399284015123 z_k1_s1150
                                 - 1.75399507070307 z_k1_s1151
                                 - 1.14699598687663 z_k1_s1152
                                 - 2.16799049777995 z_k1_s1153
                                 - 0.681998653694523 z_k1_s1154
                                 - 0.658998052736664 z_k1_s1155
                                 - 1.36699838948124 z_k1_s1156
                                 - 1.71199091438707 z_k1_s1157
                                 - 0.733000362912936 z_k1_s1160
                                 - 0.280999730036006 z_k1_s1161
                                 - 0.712999016078805 z_k1_s1162
                                 - 0.586000173073744 z_k1_s1163
                                 - 0.506997336876796 z_k1_s1164
                                 - 0.290999951853459 z_k1_s1165
                                 - 0.478998160906688 z_k1_s1166
                                 - 0.926000489273341 z_k1_s1167
                                 - 0.722000302064252 z_k1_s1170
                                 - 0.471997801107701 z_k1_s1171
                                 - 0.440001157606217 z_k1_s1172
                                 - 0.237999801096773 z_k1_s1173
                                 - 0.304999652609807 z_k1_s1174
                                 - 0.234999801981269 z_k1_s1175
                                 - 1.01599867058494 z_k1_s1176
                                 - 0.433999614576488 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.00206 w_k1_ADM_s000
                                 + 1.25854 w_k1_AXP_s000
                                 + 0.93798 w_k1_DIS_s000
                                 + 0.956569 w_k1_FDX_s000
                                 + 1.25351 w_k1_IBM_s000
                                 + 1.02332 w_k1_INTC_s000
                                 + 1.04293 w_k1_KO_s000 + 1.14372 w_k1_MAT_s000
                                 + 0.924759 w_k1_MSFT_s000
                                 + 1.01935 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.08128 w_k1_ADM_s000
                                 + 1.28944 w_k1_AXP_s000 + 1.1152 w_k1_DIS_s000
                                 + 1.20009 w_k1_FDX_s000
                                 + 0.776079 w_k1_IBM_s000
                                 + 1.19753 w_k1_INTC_s000
                                 + 1.00128 w_k1_KO_s000 + 1.11418 w_k1_MAT_s000
                                 + 1.15581 w_k1_MSFT_s000
                                 + 1.06515 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.707091 w_k1_ADM_s000
                                 + 0.952836 w_k1_AXP_s000
                                 + 0.833555 w_k1_DIS_s000
                                 + 0.780775 w_k1_FDX_s000
                                 + 1.44287 w_k1_IBM_s000
                                 + 1.0704 w_k1_INTC_s000 + 1.16238 w_k1_KO_s000
                                 + 1.69387 w_k1_MAT_s000
                                 + 0.851735 w_k1_MSFT_s000
                                 + 1.11339 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.715364 w_k1_ADM_s000
                                 + 0.68954 w_k1_AXP_s000
                                 + 0.61904 w_k1_DIS_s000
                                 + 0.73089 w_k1_FDX_s000
                                 + 0.785794 w_k1_IBM_s000
                                 + 0.769378 w_k1_INTC_s000
                                 + 0.672534 w_k1_KO_s000
                                 + 0.596299 w_k1_MAT_s000
                                 + 0.71297 w_k1_MSFT_s000
                                 + 0.756808 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.25992 w_k1_ADM_s000
                                 + 1.65179 w_k1_AXP_s000
                                 + 1.51274 w_k1_DIS_s000
                                 + 1.21683 w_k1_FDX_s000
                                 + 1.20703 w_k1_IBM_s000
                                 + 1.32589 w_k1_INTC_s000 + 1.314 w_k1_KO_s000
                                 + 1.11643 w_k1_MAT_s000
                                 + 1.29724 w_k1_MSFT_s000
                                 + 1.21733 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.95305 w_k1_ADM_s000
                                 + 1.12233 w_k1_AXP_s000
                                 + 0.63688 w_k1_DIS_s000
                                 + 0.805432 w_k1_FDX_s000
                                 + 1.1164 w_k1_IBM_s000
                                 + 0.917997 w_k1_INTC_s000
                                 + 0.935057 w_k1_KO_s000 + 1.0407 w_k1_MAT_s000
                                 + 0.746867 w_k1_MSFT_s000
                                 + 1.02077 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.880701 w_k1_ADM_s000
                                 + 0.964707 w_k1_AXP_s000
                                 + 0.947022 w_k1_DIS_s000
                                 + 0.886141 w_k1_FDX_s000
                                 + 0.99881 w_k1_IBM_s000
                                 + 0.916511 w_k1_INTC_s000
                                 + 1.00086 w_k1_KO_s000 + 0.89911 w_k1_MAT_s000
                                 + 0.8617 w_k1_MSFT_s000
                                 + 0.858548 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.853688 w_k1_ADM_s000
                                 + 0.989251 w_k1_AXP_s000
                                 + 0.856139 w_k1_DIS_s000
                                 + 0.966908 w_k1_FDX_s000
                                 + 0.60794 w_k1_IBM_s000
                                 + 0.885401 w_k1_INTC_s000
                                 + 0.751246 w_k1_KO_s000
                                 + 0.674844 w_k1_MAT_s000
                                 + 0.93807 w_k1_MSFT_s000
                                 + 1.04548 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.77845 w_k1_ADM_s001
                                 + 1.53552 w_k1_AXP_s001
                                 + 1.69064 w_k1_DIS_s001
                                 + 1.60621 w_k1_FDX_s001
                                 + 1.10271 w_k1_IBM_s001
                                 + 1.51151 w_k1_INTC_s001
                                 + 1.30336 w_k1_KO_s001
                                 + 0.976691 w_k1_MAT_s001
                                 + 1.52458 w_k1_MSFT_s001
                                 + 1.24084 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.16204 w_k1_ADM_s001
                                 + 1.19132 w_k1_AXP_s001
                                 + 1.67172 w_k1_DIS_s001
                                 + 1.21234 w_k1_FDX_s001
                                 + 1.22625 w_k1_IBM_s001
                                 + 1.25196 w_k1_INTC_s001
                                 + 1.45366 w_k1_KO_s001 + 1.41538 w_k1_MAT_s001
                                 + 1.50926 w_k1_MSFT_s001
                                 + 1.14824 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.56872 w_k1_ADM_s001
                                 + 1.06949 w_k1_AXP_s001
                                 + 0.932749 w_k1_DIS_s001
                                 + 0.890557 w_k1_FDX_s001
                                 + 1.49086 w_k1_IBM_s001
                                 + 0.779668 w_k1_INTC_s001
                                 + 1.24267 w_k1_KO_s001 + 1.67984 w_k1_MAT_s001
                                 + 0.865364 w_k1_MSFT_s001
                                 + 0.975002 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.12499 w_k1_ADM_s001
                                 + 0.866247 w_k1_AXP_s001
                                 + 0.770118 w_k1_DIS_s001
                                 + 0.887257 w_k1_FDX_s001
                                 + 0.884466 w_k1_IBM_s001
                                 + 0.903923 w_k1_INTC_s001
                                 + 0.913366 w_k1_KO_s001
                                 + 0.666547 w_k1_MAT_s001
                                 + 0.877676 w_k1_MSFT_s001
                                 + 0.907677 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.19476 w_k1_ADM_s001
                                 + 1.18808 w_k1_AXP_s001
                                 + 1.22257 w_k1_DIS_s001
                                 + 1.20377 w_k1_FDX_s001
                                 + 0.822855 w_k1_IBM_s001
                                 + 0.994766 w_k1_INTC_s001
                                 + 1.02695 w_k1_KO_s001
                                 + 0.896419 w_k1_MAT_s001
                                 + 1.02712 w_k1_MSFT_s001
                                 + 1.12887 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.852125 w_k1_ADM_s001
                                 + 0.832385 w_k1_AXP_s001
                                 + 0.742043 w_k1_DIS_s001
                                 + 0.821985 w_k1_FDX_s001
                                 + 0.603691 w_k1_IBM_s001
                                 + 0.830849 w_k1_INTC_s001
                                 + 0.756992 w_k1_KO_s001
                                 + 0.959935 w_k1_MAT_s001
                                 + 0.934269 w_k1_MSFT_s001
                                 + 0.802581 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.40403 w_k1_ADM_s001
                                 + 1.10986 w_k1_AXP_s001
                                 + 1.19436 w_k1_DIS_s001
                                 + 1.06753 w_k1_FDX_s001
                                 + 1.24475 w_k1_IBM_s001
                                 + 1.08504 w_k1_INTC_s001
                                 + 1.28149 w_k1_KO_s001 + 1.13172 w_k1_MAT_s001
                                 + 1.07603 w_k1_MSFT_s001
                                 + 0.994393 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.27016 w_k1_ADM_s001
                                 + 1.17975 w_k1_AXP_s001
                                 + 0.906355 w_k1_DIS_s001
                                 + 1.03956 w_k1_FDX_s001
                                 + 1.04962 w_k1_IBM_s001
                                 + 1.14144 w_k1_INTC_s001
                                 + 1.22498 w_k1_KO_s001 + 1.54609 w_k1_MAT_s001
                                 + 1.09558 w_k1_MSFT_s001
                                 + 0.975191 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.14003 w_k1_ADM_s002
                                 + 0.835613 w_k1_AXP_s002
                                 + 0.653742 w_k1_DIS_s002
                                 + 0.771164 w_k1_FDX_s002
                                 + 0.688014 w_k1_IBM_s002
                                 + 0.871859 w_k1_INTC_s002
                                 + 0.646632 w_k1_KO_s002
                                 + 0.615324 w_k1_MAT_s002
                                 + 0.78686 w_k1_MSFT_s002
                                 + 0.912679 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.996191 w_k1_ADM_s002
                                 + 0.75746 w_k1_AXP_s002
                                 + 0.95984 w_k1_DIS_s002
                                 + 0.935524 w_k1_FDX_s002
                                 + 0.849925 w_k1_IBM_s002
                                 + 0.995749 w_k1_INTC_s002
                                 + 0.834061 w_k1_KO_s002
                                 + 0.742285 w_k1_MAT_s002
                                 + 0.97767 w_k1_MSFT_s002
                                 + 0.960266 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.830739 w_k1_ADM_s002
                                 + 0.887197 w_k1_AXP_s002
                                 + 1.01645 w_k1_DIS_s002
                                 + 0.863241 w_k1_FDX_s002
                                 + 1.13229 w_k1_IBM_s002
                                 + 0.859252 w_k1_INTC_s002
                                 + 0.994145 w_k1_KO_s002
                                 + 0.912326 w_k1_MAT_s002
                                 + 0.872442 w_k1_MSFT_s002
                                 + 1.12927 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.27032 w_k1_ADM_s002
                                 + 1.80164 w_k1_AXP_s002
                                 + 1.71336 w_k1_DIS_s002
                                 + 1.18119 w_k1_FDX_s002
                                 + 1.71356 w_k1_IBM_s002
                                 + 1.37221 w_k1_INTC_s002
                                 + 1.49836 w_k1_KO_s002 + 1.38256 w_k1_MAT_s002
                                 + 1.41587 w_k1_MSFT_s002
                                 + 1.35924 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.665705 w_k1_ADM_s002
                                 + 0.707555 w_k1_AXP_s002
                                 + 0.606978 w_k1_DIS_s002
                                 + 0.661607 w_k1_FDX_s002
                                 + 0.880562 w_k1_IBM_s002
                                 + 0.82007 w_k1_INTC_s002
                                 + 0.717393 w_k1_KO_s002
                                 + 0.730757 w_k1_MAT_s002
                                 + 0.698215 w_k1_MSFT_s002
                                 + 0.797171 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.22215 w_k1_ADM_s002
                                 + 1.34983 w_k1_AXP_s002
                                 + 1.30402 w_k1_DIS_s002
                                 + 1.20494 w_k1_FDX_s002
                                 + 0.847275 w_k1_IBM_s002
                                 + 1.0048 w_k1_INTC_s002
                                 + 0.885603 w_k1_KO_s002
                                 + 0.634972 w_k1_MAT_s002
                                 + 1.01761 w_k1_MSFT_s002
                                 + 1.01077 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.30614 w_k1_ADM_s002
                                 + 1.55793 w_k1_AXP_s002 + 1.3848 w_k1_DIS_s002
                                 + 1.11857 w_k1_FDX_s002 + 1.2714 w_k1_IBM_s002
                                 + 1.23598 w_k1_INTC_s002
                                 + 1.07044 w_k1_KO_s002 + 1.08187 w_k1_MAT_s002
                                 + 1.13227 w_k1_MSFT_s002
                                 + 1.13407 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.924265 w_k1_ADM_s002
                                 + 1.15662 w_k1_AXP_s002
                                 + 0.938653 w_k1_DIS_s002
                                 + 1.04807 w_k1_FDX_s002
                                 + 0.713273 w_k1_IBM_s002
                                 + 1.20786 w_k1_INTC_s002
                                 + 0.766839 w_k1_KO_s002
                                 + 0.656785 w_k1_MAT_s002
                                 + 1.07232 w_k1_MSFT_s002
                                 + 0.998961 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.779561 w_k1_ADM_s003
                                 + 0.738879 w_k1_AXP_s003
                                 + 0.800381 w_k1_DIS_s003
                                 + 0.919937 w_k1_FDX_s003
                                 + 0.863169 w_k1_IBM_s003
                                 + 0.819991 w_k1_INTC_s003
                                 + 0.887186 w_k1_KO_s003
                                 + 0.83472 w_k1_MAT_s003
                                 + 0.943587 w_k1_MSFT_s003
                                 + 0.874972 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.42481 w_k1_ADM_s003
                                 + 1.21477 w_k1_AXP_s003 + 1.5785 w_k1_DIS_s003
                                 + 1.29396 w_k1_FDX_s003
                                 + 1.68453 w_k1_IBM_s003
                                 + 0.8985 w_k1_INTC_s003 + 1.50968 w_k1_KO_s003
                                 + 1.63887 w_k1_MAT_s003
                                 + 1.1558 w_k1_MSFT_s003 + 1.17287 w_k1_PG_s003
                                 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.36335 w_k1_ADM_s003
                                 + 1.0994 w_k1_AXP_s003
                                 + 0.954785 w_k1_DIS_s003
                                 + 1.07842 w_k1_FDX_s003
                                 + 0.787253 w_k1_IBM_s003
                                 + 0.895764 w_k1_INTC_s003
                                 + 0.806635 w_k1_KO_s003
                                 + 0.800057 w_k1_MAT_s003
                                 + 0.93513 w_k1_MSFT_s003
                                 + 0.98179 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.11724 w_k1_ADM_s003
                                 + 1.08445 w_k1_AXP_s003
                                 + 1.02951 w_k1_DIS_s003
                                 + 1.06825 w_k1_FDX_s003
                                 + 1.21001 w_k1_IBM_s003
                                 + 0.946946 w_k1_INTC_s003
                                 + 1.05295 w_k1_KO_s003 + 1.11152 w_k1_MAT_s003
                                 + 0.993167 w_k1_MSFT_s003
                                 + 0.998382 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.929084 w_k1_ADM_s003
                                 + 0.762531 w_k1_AXP_s003
                                 + 0.668639 w_k1_DIS_s003
                                 + 0.849971 w_k1_FDX_s003
                                 + 0.660456 w_k1_IBM_s003
                                 + 0.76925 w_k1_INTC_s003
                                 + 0.75556 w_k1_KO_s003 + 0.65856 w_k1_MAT_s003
                                 + 0.892514 w_k1_MSFT_s003
                                 + 0.719663 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.744671 w_k1_ADM_s003
                                 + 0.538834 w_k1_AXP_s003
                                 + 0.50867 w_k1_DIS_s003
                                 + 0.650326 w_k1_FDX_s003
                                 + 0.77593 w_k1_IBM_s003
                                 + 0.759659 w_k1_INTC_s003
                                 + 0.685647 w_k1_KO_s003
                                 + 0.487967 w_k1_MAT_s003
                                 + 0.710464 w_k1_MSFT_s003
                                 + 0.695983 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.49275 w_k1_ADM_s003
                                 + 1.68228 w_k1_AXP_s003
                                 + 1.76302 w_k1_DIS_s003
                                 + 1.65412 w_k1_FDX_s003
                                 + 1.07263 w_k1_IBM_s003
                                 + 1.24477 w_k1_INTC_s003 + 1.1347 w_k1_KO_s003
                                 + 0.747535 w_k1_MAT_s003
                                 + 1.32489 w_k1_MSFT_s003
                                 + 1.07995 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.815949 w_k1_ADM_s003
                                 + 0.694223 w_k1_AXP_s003
                                 + 0.726297 w_k1_DIS_s003
                                 + 0.756555 w_k1_FDX_s003
                                 + 1.1406 w_k1_IBM_s003
                                 + 0.817424 w_k1_INTC_s003
                                 + 0.948984 w_k1_KO_s003
                                 + 0.91971 w_k1_MAT_s003
                                 + 0.891586 w_k1_MSFT_s003
                                 + 0.884247 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.855455 w_k1_ADM_s004
                                 + 1.08961 w_k1_AXP_s004
                                 + 0.918854 w_k1_DIS_s004
                                 + 0.95096 w_k1_FDX_s004 + 1.3508 w_k1_IBM_s004
                                 + 1.1613 w_k1_INTC_s004 + 1.01704 w_k1_KO_s004
                                 + 1.24926 w_k1_MAT_s004 + 1.04 w_k1_MSFT_s004
                                 + 1.01767 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.19508 w_k1_ADM_s004
                                 + 0.906794 w_k1_AXP_s004
                                 + 1.13461 w_k1_DIS_s004
                                 + 1.17817 w_k1_FDX_s004
                                 + 0.776299 w_k1_IBM_s004
                                 + 1.29064 w_k1_INTC_s004
                                 + 0.958131 w_k1_KO_s004
                                 + 0.687339 w_k1_MAT_s004
                                 + 1.26403 w_k1_MSFT_s004
                                 + 1.15673 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.965803 w_k1_ADM_s004
                                 + 0.843751 w_k1_AXP_s004
                                 + 0.724488 w_k1_DIS_s004
                                 + 0.928172 w_k1_FDX_s004
                                 + 0.851227 w_k1_IBM_s004
                                 + 1.08021 w_k1_INTC_s004
                                 + 0.761682 w_k1_KO_s004
                                 + 0.721203 w_k1_MAT_s004
                                 + 0.927168 w_k1_MSFT_s004
                                 + 0.923051 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.03301 w_k1_ADM_s004
                                 + 1.3752 w_k1_AXP_s004 + 1.13671 w_k1_DIS_s004
                                 + 1.14234 w_k1_FDX_s004
                                 + 1.70778 w_k1_IBM_s004
                                 + 1.32249 w_k1_INTC_s004
                                 + 1.17169 w_k1_KO_s004 + 1.85631 w_k1_MAT_s004
                                 + 1.18483 w_k1_MSFT_s004
                                 + 1.22292 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.95769 w_k1_ADM_s004
                                 + 1.03752 w_k1_AXP_s004
                                 + 1.11226 w_k1_DIS_s004
                                 + 1.12115 w_k1_FDX_s004
                                 + 0.903995 w_k1_IBM_s004
                                 + 1.24047 w_k1_INTC_s004
                                 + 0.901082 w_k1_KO_s004
                                 + 0.894089 w_k1_MAT_s004
                                 + 1.1904 w_k1_MSFT_s004 + 1.10689 w_k1_PG_s004
                                 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.723015 w_k1_ADM_s004
                                 + 0.647024 w_k1_AXP_s004
                                 + 0.739261 w_k1_DIS_s004
                                 + 0.777145 w_k1_FDX_s004
                                 + 0.798603 w_k1_IBM_s004
                                 + 1.06917 w_k1_INTC_s004
                                 + 0.847783 w_k1_KO_s004
                                 + 0.570478 w_k1_MAT_s004
                                 + 0.979919 w_k1_MSFT_s004
                                 + 0.886559 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.13133 w_k1_ADM_s004
                                 + 1.43942 w_k1_AXP_s004
                                 + 1.57427 w_k1_DIS_s004
                                 + 1.34269 w_k1_FDX_s004 + 1.1618 w_k1_IBM_s004
                                 + 1.53139 w_k1_INTC_s004
                                 + 1.25106 w_k1_KO_s004 + 1.06625 w_k1_MAT_s004
                                 + 1.32423 w_k1_MSFT_s004
                                 + 1.28883 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.93737 w_k1_ADM_s004
                                 + 1.14589 w_k1_AXP_s004
                                 + 0.837569 w_k1_DIS_s004
                                 + 0.914865 w_k1_FDX_s004
                                 + 1.0319 w_k1_IBM_s004
                                 + 0.96918 w_k1_INTC_s004
                                 + 0.932409 w_k1_KO_s004
                                 + 1.27589 w_k1_MAT_s004
                                 + 0.962065 w_k1_MSFT_s004
                                 + 0.952753 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.05061 w_k1_ADM_s005
                                 + 1.64134 w_k1_AXP_s005
                                 + 1.21981 w_k1_DIS_s005
                                 + 1.28202 w_k1_FDX_s005
                                 + 0.873128 w_k1_IBM_s005
                                 + 1.23519 w_k1_INTC_s005
                                 + 1.00287 w_k1_KO_s005
                                 + 0.988546 w_k1_MAT_s005
                                 + 1.17355 w_k1_MSFT_s005
                                 + 1.30225 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.857872 w_k1_ADM_s005
                                 + 1.24827 w_k1_AXP_s005
                                 + 1.08067 w_k1_DIS_s005
                                 + 1.09741 w_k1_FDX_s005
                                 + 1.14562 w_k1_IBM_s005
                                 + 1.53559 w_k1_INTC_s005
                                 + 1.03451 w_k1_KO_s005 + 1.27819 w_k1_MAT_s005
                                 + 1.35048 w_k1_MSFT_s005
                                 + 1.02439 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.75499 w_k1_ADM_s005
                                 + 0.738499 w_k1_AXP_s005
                                 + 0.639635 w_k1_DIS_s005
                                 + 0.800771 w_k1_FDX_s005
                                 + 0.886625 w_k1_IBM_s005
                                 + 0.997219 w_k1_INTC_s005
                                 + 0.817459 w_k1_KO_s005
                                 + 0.948733 w_k1_MAT_s005
                                 + 0.955426 w_k1_MSFT_s005
                                 + 0.875595 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.964612 w_k1_ADM_s005
                                 + 1.24854 w_k1_AXP_s005
                                 + 1.66297 w_k1_DIS_s005 + 1.2445 w_k1_FDX_s005
                                 + 1.31212 w_k1_IBM_s005
                                 + 1.31363 w_k1_INTC_s005
                                 + 1.57481 w_k1_KO_s005 + 1.38434 w_k1_MAT_s005
                                 + 1.36525 w_k1_MSFT_s005
                                 + 1.31943 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.964017 w_k1_ADM_s005
                                 + 0.825778 w_k1_AXP_s005
                                 + 0.967717 w_k1_DIS_s005
                                 + 1.03079 w_k1_FDX_s005
                                 + 0.887162 w_k1_IBM_s005
                                 + 1.05728 w_k1_INTC_s005
                                 + 1.00349 w_k1_KO_s005
                                 + 0.821688 w_k1_MAT_s005
                                 + 1.19089 w_k1_MSFT_s005
                                 + 1.05173 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.0139 w_k1_ADM_s005
                                 + 0.761301 w_k1_AXP_s005
                                 + 0.659589 w_k1_DIS_s005
                                 + 0.878135 w_k1_FDX_s005
                                 + 0.586478 w_k1_IBM_s005
                                 + 1.12778 w_k1_INTC_s005
                                 + 0.800147 w_k1_KO_s005
                                 + 0.529201 w_k1_MAT_s005
                                 + 1.02007 w_k1_MSFT_s005
                                 + 0.872205 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.81145 w_k1_ADM_s005
                                 + 0.845424 w_k1_AXP_s005
                                 + 0.705581 w_k1_DIS_s005
                                 + 0.72292 w_k1_FDX_s005
                                 + 1.21425 w_k1_IBM_s005
                                 + 0.986632 w_k1_INTC_s005
                                 + 1.02283 w_k1_KO_s005 + 1.10616 w_k1_MAT_s005
                                 + 0.842086 w_k1_MSFT_s005
                                 + 0.950752 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.979451 w_k1_ADM_s005
                                 + 0.919878 w_k1_AXP_s005
                                 + 1.06985 w_k1_DIS_s005
                                 + 1.05368 w_k1_FDX_s005
                                 + 1.55692 w_k1_IBM_s005
                                 + 1.35714 w_k1_INTC_s005
                                 + 1.31532 w_k1_KO_s005 + 1.48589 w_k1_MAT_s005
                                 + 1.10984 w_k1_MSFT_s005
                                 + 1.11424 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.26703 w_k1_ADM_s006
                                 + 1.0174 w_k1_AXP_s006
                                 + 0.830648 w_k1_DIS_s006
                                 + 0.956538 w_k1_FDX_s006
                                 + 0.751242 w_k1_IBM_s006
                                 + 0.942449 w_k1_INTC_s006
                                 + 0.803317 w_k1_KO_s006
                                 + 0.789227 w_k1_MAT_s006
                                 + 0.912945 w_k1_MSFT_s006
                                 + 0.982864 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.28202 w_k1_ADM_s006
                                 + 1.49383 w_k1_AXP_s006
                                 + 1.88764 w_k1_DIS_s006
                                 + 1.52532 w_k1_FDX_s006
                                 + 1.03586 w_k1_IBM_s006
                                 + 1.31252 w_k1_INTC_s006 + 1.2005 w_k1_KO_s006
                                 + 1.3719 w_k1_MAT_s006
                                 + 1.31837 w_k1_MSFT_s006
                                 + 1.28135 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.803015 w_k1_ADM_s006
                                 + 0.784607 w_k1_AXP_s006
                                 + 0.726194 w_k1_DIS_s006
                                 + 0.816091 w_k1_FDX_s006
                                 + 1.01166 w_k1_IBM_s006
                                 + 0.847337 w_k1_INTC_s006
                                 + 0.930772 w_k1_KO_s006 + 1.1445 w_k1_MAT_s006
                                 + 0.83509 w_k1_MSFT_s006
                                 + 0.996065 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.17587 w_k1_ADM_s006
                                 + 0.997071 w_k1_AXP_s006
                                 + 1.02483 w_k1_DIS_s006
                                 + 0.874018 w_k1_FDX_s006
                                 + 1.54102 w_k1_IBM_s006
                                 + 1.23225 w_k1_INTC_s006
                                 + 1.22187 w_k1_KO_s006 + 1.40625 w_k1_MAT_s006
                                 + 0.929128 w_k1_MSFT_s006
                                 + 1.15537 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.06229 w_k1_ADM_s006
                                 + 1.17939 w_k1_AXP_s006
                                 + 1.43047 w_k1_DIS_s006
                                 + 1.17923 w_k1_FDX_s006
                                 + 1.42604 w_k1_IBM_s006
                                 + 0.968077 w_k1_INTC_s006
                                 + 1.10493 w_k1_KO_s006 + 1.15134 w_k1_MAT_s006
                                 + 0.965581 w_k1_MSFT_s006
                                 + 1.21298 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.29235 w_k1_ADM_s006
                                 + 1.48505 w_k1_AXP_s006
                                 + 1.16751 w_k1_DIS_s006
                                 + 1.28349 w_k1_FDX_s006
                                 + 0.879492 w_k1_IBM_s006
                                 + 1.19567 w_k1_INTC_s006
                                 + 1.05076 w_k1_KO_s006 + 1.49987 w_k1_MAT_s006
                                 + 1.04484 w_k1_MSFT_s006
                                 + 1.08593 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.22943 w_k1_ADM_s006
                                 + 0.858551 w_k1_AXP_s006
                                 + 1.27708 w_k1_DIS_s006
                                 + 1.14397 w_k1_FDX_s006
                                 + 0.83826 w_k1_IBM_s006
                                 + 1.21633 w_k1_INTC_s006
                                 + 1.18034 w_k1_KO_s006 + 0.69984 w_k1_MAT_s006
                                 + 1.16456 w_k1_MSFT_s006
                                 + 1.20917 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.24911 w_k1_ADM_s006
                                 + 1.36524 w_k1_AXP_s006
                                 + 1.19296 w_k1_DIS_s006
                                 + 1.13257 w_k1_FDX_s006
                                 + 1.22586 w_k1_IBM_s006
                                 + 0.959662 w_k1_INTC_s006
                                 + 1.18558 w_k1_KO_s006 + 1.28067 w_k1_MAT_s006
                                 + 0.981705 w_k1_MSFT_s006
                                 + 1.12584 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.964253 w_k1_ADM_s007
                                 + 0.855601 w_k1_AXP_s007
                                 + 0.708608 w_k1_DIS_s007
                                 + 0.722978 w_k1_FDX_s007
                                 + 1.26681 w_k1_IBM_s007
                                 + 0.775781 w_k1_INTC_s007
                                 + 1.03903 w_k1_KO_s007 + 1.36584 w_k1_MAT_s007
                                 + 0.803208 w_k1_MSFT_s007
                                 + 0.940141 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.975231 w_k1_ADM_s007
                                 + 0.772216 w_k1_AXP_s007
                                 + 0.951548 w_k1_DIS_s007
                                 + 0.896065 w_k1_FDX_s007
                                 + 1.06354 w_k1_IBM_s007
                                 + 0.777996 w_k1_INTC_s007
                                 + 1.1391 w_k1_KO_s007 + 1.34904 w_k1_MAT_s007
                                 + 0.871348 w_k1_MSFT_s007
                                 + 1.06509 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.946886 w_k1_ADM_s007
                                 + 0.760755 w_k1_AXP_s007
                                 + 0.786618 w_k1_DIS_s007
                                 + 0.842495 w_k1_FDX_s007
                                 + 0.732802 w_k1_IBM_s007
                                 + 0.71481 w_k1_INTC_s007
                                 + 0.843711 w_k1_KO_s007
                                 + 0.831502 w_k1_MAT_s007
                                 + 0.735636 w_k1_MSFT_s007
                                 + 0.90322 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.989605 w_k1_ADM_s007
                                 + 1.10403 w_k1_AXP_s007
                                 + 0.901359 w_k1_DIS_s007
                                 + 1.01027 w_k1_FDX_s007
                                 + 0.948049 w_k1_IBM_s007
                                 + 0.983434 w_k1_INTC_s007
                                 + 1.05268 w_k1_KO_s007
                                 + 0.881951 w_k1_MAT_s007
                                 + 0.897032 w_k1_MSFT_s007
                                 + 0.954449 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.18406 w_k1_ADM_s007
                                 + 1.03911 w_k1_AXP_s007
                                 + 1.26479 w_k1_DIS_s007
                                 + 1.25113 w_k1_FDX_s007
                                 + 0.887968 w_k1_IBM_s007
                                 + 1.17236 w_k1_INTC_s007
                                 + 1.12819 w_k1_KO_s007
                                 + 0.969423 w_k1_MAT_s007
                                 + 1.10315 w_k1_MSFT_s007
                                 + 1.12984 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.843928 w_k1_ADM_s007
                                 + 1.08514 w_k1_AXP_s007
                                 + 0.740809 w_k1_DIS_s007
                                 + 0.970975 w_k1_FDX_s007
                                 + 0.705005 w_k1_IBM_s007
                                 + 1.03164 w_k1_INTC_s007
                                 + 0.742143 w_k1_KO_s007
                                 + 0.807677 w_k1_MAT_s007
                                 + 0.809253 w_k1_MSFT_s007
                                 + 0.930794 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.833705 w_k1_ADM_s007
                                 + 0.53224 w_k1_AXP_s007
                                 + 0.765636 w_k1_DIS_s007
                                 + 0.78254 w_k1_FDX_s007
                                 + 0.737746 w_k1_IBM_s007
                                 + 1.01193 w_k1_INTC_s007
                                 + 0.86584 w_k1_KO_s007
                                 + 0.332058 w_k1_MAT_s007
                                 + 0.946465 w_k1_MSFT_s007
                                 + 0.857385 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.11507 w_k1_ADM_s007
                                 + 1.34442 w_k1_AXP_s007 + 1.4153 w_k1_DIS_s007
                                 + 1.10901 w_k1_FDX_s007
                                 + 1.41974 w_k1_IBM_s007
                                 + 1.15883 w_k1_INTC_s007
                                 + 1.43699 w_k1_KO_s007 + 1.41694 w_k1_MAT_s007
                                 + 1.12987 w_k1_MSFT_s007
                                 + 1.01513 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.30706 w_k1_ADM_s010
                                 + 1.10037 w_k1_AXP_s010
                                 + 0.878965 w_k1_DIS_s010
                                 + 1.10135 w_k1_FDX_s010
                                 + 0.914541 w_k1_IBM_s010
                                 + 1.10771 w_k1_INTC_s010
                                 + 0.893645 w_k1_KO_s010
                                 + 0.844606 w_k1_MAT_s010
                                 + 1.04776 w_k1_MSFT_s010
                                 + 0.995575 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.918186 w_k1_ADM_s010
                                 + 0.735929 w_k1_AXP_s010
                                 + 0.636553 w_k1_DIS_s010
                                 + 0.749896 w_k1_FDX_s010
                                 + 0.837642 w_k1_IBM_s010
                                 + 0.804664 w_k1_INTC_s010
                                 + 0.798476 w_k1_KO_s010
                                 + 0.705556 w_k1_MAT_s010
                                 + 0.812964 w_k1_MSFT_s010
                                 + 0.818777 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.16125 w_k1_ADM_s010
                                 + 1.01935 w_k1_AXP_s010
                                 + 1.19685 w_k1_DIS_s010
                                 + 1.13037 w_k1_FDX_s010
                                 + 0.881024 w_k1_IBM_s010
                                 + 1.00889 w_k1_INTC_s010
                                 + 1.19956 w_k1_KO_s010
                                 + 0.662016 w_k1_MAT_s010
                                 + 1.24872 w_k1_MSFT_s010
                                 + 1.02947 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.22825 w_k1_ADM_s010
                                 + 1.25129 w_k1_AXP_s010
                                 + 1.28908 w_k1_DIS_s010
                                 + 1.11469 w_k1_FDX_s010
                                 + 1.52473 w_k1_IBM_s010
                                 + 1.26559 w_k1_INTC_s010 + 1.3077 w_k1_KO_s010
                                 + 0.992575 w_k1_MAT_s010
                                 + 1.32667 w_k1_MSFT_s010
                                 + 1.05964 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.789891 w_k1_ADM_s010
                                 + 0.803577 w_k1_AXP_s010
                                 + 0.96221 w_k1_DIS_s010
                                 + 0.894214 w_k1_FDX_s010
                                 + 1.28726 w_k1_IBM_s010
                                 + 1.04283 w_k1_INTC_s010
                                 + 0.953735 w_k1_KO_s010
                                 + 0.969936 w_k1_MAT_s010
                                 + 1.09801 w_k1_MSFT_s010
                                 + 0.972178 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.23678 w_k1_ADM_s010
                                 + 1.30534 w_k1_AXP_s010
                                 + 1.65035 w_k1_DIS_s010
                                 + 1.41821 w_k1_FDX_s010
                                 + 1.13976 w_k1_IBM_s010
                                 + 1.48413 w_k1_INTC_s010
                                 + 1.34947 w_k1_KO_s010 + 1.25712 w_k1_MAT_s010
                                 + 1.54163 w_k1_MSFT_s010
                                 + 1.23371 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.912364 w_k1_ADM_s010
                                 + 0.842587 w_k1_AXP_s010
                                 + 0.778882 w_k1_DIS_s010
                                 + 0.963659 w_k1_FDX_s010
                                 + 0.701331 w_k1_IBM_s010
                                 + 1.03573 w_k1_INTC_s010
                                 + 0.854198 w_k1_KO_s010
                                 + 0.63191 w_k1_MAT_s010
                                 + 1.04496 w_k1_MSFT_s010
                                 + 0.881046 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.08382 w_k1_ADM_s010
                                 + 1.02262 w_k1_AXP_s010
                                 + 0.838845 w_k1_DIS_s010
                                 + 0.936112 w_k1_FDX_s010
                                 + 1.02721 w_k1_IBM_s010
                                 + 0.99484 w_k1_INTC_s010
                                 + 0.891992 w_k1_KO_s010
                                 + 0.787028 w_k1_MAT_s010
                                 + 0.880304 w_k1_MSFT_s010
                                 + 0.963143 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.831817 w_k1_ADM_s011
                                 + 0.975653 w_k1_AXP_s011
                                 + 1.05086 w_k1_DIS_s011
                                 + 1.05237 w_k1_FDX_s011
                                 + 0.808095 w_k1_IBM_s011
                                 + 1.06224 w_k1_INTC_s011
                                 + 1.01092 w_k1_KO_s011
                                 + 0.878783 w_k1_MAT_s011
                                 + 0.920136 w_k1_MSFT_s011
                                 + 1.02521 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.08247 w_k1_ADM_s011
                                 + 1.41414 w_k1_AXP_s011
                                 + 0.942385 w_k1_DIS_s011
                                 + 1.14695 w_k1_FDX_s011 + 1.1163 w_k1_IBM_s011
                                 + 1.08365 w_k1_INTC_s011
                                 + 1.08382 w_k1_KO_s011 + 1.45176 w_k1_MAT_s011
                                 + 0.883059 w_k1_MSFT_s011
                                 + 1.24977 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.784376 w_k1_ADM_s011
                                 + 1.00226 w_k1_AXP_s011
                                 + 0.764523 w_k1_DIS_s011
                                 + 0.870718 w_k1_FDX_s011
                                 + 1.18856 w_k1_IBM_s011
                                 + 1.0063 w_k1_INTC_s011 + 1.03774 w_k1_KO_s011
                                 + 0.914425 w_k1_MAT_s011
                                 + 0.915052 w_k1_MSFT_s011
                                 + 1.07973 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.772653 w_k1_ADM_s011
                                 + 0.743043 w_k1_AXP_s011
                                 + 0.711388 w_k1_DIS_s011
                                 + 0.895396 w_k1_FDX_s011
                                 + 0.74337 w_k1_IBM_s011
                                 + 0.795764 w_k1_INTC_s011
                                 + 0.844248 w_k1_KO_s011 + 0.9403 w_k1_MAT_s011
                                 + 0.83852 w_k1_MSFT_s011
                                 + 1.01153 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.13016 w_k1_ADM_s011
                                 + 1.52248 w_k1_AXP_s011
                                 + 1.47298 w_k1_DIS_s011
                                 + 1.42281 w_k1_FDX_s011
                                 + 1.35192 w_k1_IBM_s011
                                 + 1.19557 w_k1_INTC_s011
                                 + 1.54961 w_k1_KO_s011 + 1.97801 w_k1_MAT_s011
                                 + 1.36059 w_k1_MSFT_s011
                                 + 1.27755 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.01415 w_k1_ADM_s011
                                 + 0.597076 w_k1_AXP_s011
                                 + 0.967743 w_k1_DIS_s011
                                 + 1.02728 w_k1_FDX_s011
                                 + 0.865186 w_k1_IBM_s011
                                 + 0.967227 w_k1_INTC_s011
                                 + 1.23964 w_k1_KO_s011
                                 + 0.870879 w_k1_MAT_s011
                                 + 1.18305 w_k1_MSFT_s011
                                 + 1.24244 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.777419 w_k1_ADM_s011
                                 + 0.94592 w_k1_AXP_s011
                                 + 1.18805 w_k1_DIS_s011
                                 + 1.08024 w_k1_FDX_s011
                                 + 1.23541 w_k1_IBM_s011
                                 + 1.11832 w_k1_INTC_s011
                                 + 1.44094 w_k1_KO_s011 + 1.28022 w_k1_MAT_s011
                                 + 1.04919 w_k1_MSFT_s011
                                 + 1.30549 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.85044 w_k1_ADM_s011
                                 + 0.625272 w_k1_AXP_s011
                                 + 0.593085 w_k1_DIS_s011
                                 + 0.715207 w_k1_FDX_s011
                                 + 0.834792 w_k1_IBM_s011
                                 + 0.724874 w_k1_INTC_s011
                                 + 0.863482 w_k1_KO_s011 + 0.6888 w_k1_MAT_s011
                                 + 0.702757 w_k1_MSFT_s011
                                 + 0.934018 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.705428 w_k1_ADM_s012
                                 + 0.784599 w_k1_AXP_s012
                                 + 0.796961 w_k1_DIS_s012
                                 + 0.833232 w_k1_FDX_s012
                                 + 1.0624 w_k1_IBM_s012
                                 + 0.936277 w_k1_INTC_s012
                                 + 0.924031 w_k1_KO_s012
                                 + 1.16158 w_k1_MAT_s012
                                 + 0.870229 w_k1_MSFT_s012
                                 + 0.958138 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.704551 w_k1_ADM_s012
                                 + 0.906432 w_k1_AXP_s012
                                 + 0.827878 w_k1_DIS_s012
                                 + 0.914726 w_k1_FDX_s012
                                 + 0.759683 w_k1_IBM_s012
                                 + 1.01692 w_k1_INTC_s012
                                 + 0.864478 w_k1_KO_s012
                                 + 0.828374 w_k1_MAT_s012
                                 + 0.987077 w_k1_MSFT_s012
                                 + 0.930778 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.855785 w_k1_ADM_s012
                                 + 1.18576 w_k1_AXP_s012
                                 + 1.27782 w_k1_DIS_s012
                                 + 1.02743 w_k1_FDX_s012
                                 + 1.43698 w_k1_IBM_s012
                                 + 1.21777 w_k1_INTC_s012 + 1.3986 w_k1_KO_s012
                                 + 1.58057 w_k1_MAT_s012
                                 + 1.2803 w_k1_MSFT_s012 + 1.07218 w_k1_PG_s012
                                 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.40683 w_k1_ADM_s012
                                 + 1.62523 w_k1_AXP_s012
                                 + 1.48973 w_k1_DIS_s012
                                 + 1.21621 w_k1_FDX_s012
                                 + 1.28394 w_k1_IBM_s012
                                 + 1.00638 w_k1_INTC_s012
                                 + 1.26756 w_k1_KO_s012 + 1.2991 w_k1_MAT_s012
                                 + 1.31413 w_k1_MSFT_s012
                                 + 1.21479 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.903883 w_k1_ADM_s012
                                 + 1.33813 w_k1_AXP_s012
                                 + 0.928257 w_k1_DIS_s012
                                 + 0.940167 w_k1_FDX_s012
                                 + 0.997343 w_k1_IBM_s012
                                 + 0.795702 w_k1_INTC_s012
                                 + 0.859407 w_k1_KO_s012
                                 + 1.12056 w_k1_MAT_s012
                                 + 0.89449 w_k1_MSFT_s012
                                 + 1.11187 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.08582 w_k1_ADM_s012
                                 + 0.925764 w_k1_AXP_s012
                                 + 1.02873 w_k1_DIS_s012
                                 + 0.855094 w_k1_FDX_s012
                                 + 1.16606 w_k1_IBM_s012
                                 + 0.907966 w_k1_INTC_s012
                                 + 1.16484 w_k1_KO_s012 + 1.28865 w_k1_MAT_s012
                                 + 1.01275 w_k1_MSFT_s012
                                 + 1.22122 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.823157 w_k1_ADM_s012
                                 + 0.803743 w_k1_AXP_s012
                                 + 1.0127 w_k1_DIS_s012
                                 + 0.961284 w_k1_FDX_s012
                                 + 0.946679 w_k1_IBM_s012
                                 + 0.91868 w_k1_INTC_s012
                                 + 1.00186 w_k1_KO_s012
                                 + 0.928912 w_k1_MAT_s012
                                 + 1.13253 w_k1_MSFT_s012
                                 + 0.94035 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.893414 w_k1_ADM_s012
                                 + 1.04869 w_k1_AXP_s012
                                 + 0.890254 w_k1_DIS_s012
                                 + 1.00335 w_k1_FDX_s012
                                 + 0.733761 w_k1_IBM_s012
                                 + 0.976064 w_k1_INTC_s012
                                 + 0.800899 w_k1_KO_s012
                                 + 0.862355 w_k1_MAT_s012
                                 + 1.09723 w_k1_MSFT_s012
                                 + 1.04459 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.04602 w_k1_ADM_s013
                                 + 1.6883 w_k1_AXP_s013 + 1.5578 w_k1_DIS_s013
                                 + 1.37556 w_k1_FDX_s013
                                 + 1.51936 w_k1_IBM_s013
                                 + 1.32508 w_k1_INTC_s013
                                 + 1.48128 w_k1_KO_s013 + 2.185 w_k1_MAT_s013
                                 + 1.40248 w_k1_MSFT_s013
                                 + 1.27796 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.05945 w_k1_ADM_s013
                                 + 0.887536 w_k1_AXP_s013
                                 + 1.08112 w_k1_DIS_s013
                                 + 0.988317 w_k1_FDX_s013
                                 + 1.07162 w_k1_IBM_s013
                                 + 0.815939 w_k1_INTC_s013
                                 + 1.06531 w_k1_KO_s013 + 0.94916 w_k1_MAT_s013
                                 + 0.978654 w_k1_MSFT_s013
                                 + 0.982762 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.888609 w_k1_ADM_s013
                                 + 1.08446 w_k1_AXP_s013
                                 + 0.965373 w_k1_DIS_s013
                                 + 0.90512 w_k1_FDX_s013
                                 + 1.45259 w_k1_IBM_s013
                                 + 0.90881 w_k1_INTC_s013
                                 + 1.22103 w_k1_KO_s013 + 1.80156 w_k1_MAT_s013
                                 + 1.02473 w_k1_MSFT_s013
                                 + 1.07237 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.12278 w_k1_ADM_s013
                                 + 1.17364 w_k1_AXP_s013
                                 + 1.38715 w_k1_DIS_s013
                                 + 1.18072 w_k1_FDX_s013
                                 + 1.04531 w_k1_IBM_s013
                                 + 1.18186 w_k1_INTC_s013
                                 + 1.23391 w_k1_KO_s013 + 1.54382 w_k1_MAT_s013
                                 + 1.49219 w_k1_MSFT_s013
                                 + 1.18549 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.25663 w_k1_ADM_s013
                                 + 1.14129 w_k1_AXP_s013
                                 + 0.874058 w_k1_DIS_s013
                                 + 1.08577 w_k1_FDX_s013
                                 + 0.771779 w_k1_IBM_s013
                                 + 0.757004 w_k1_INTC_s013
                                 + 0.753128 w_k1_KO_s013
                                 + 1.11192 w_k1_MAT_s013
                                 + 0.915101 w_k1_MSFT_s013
                                 + 0.881622 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.887045 w_k1_ADM_s013
                                 + 0.743992 w_k1_AXP_s013
                                 + 0.643272 w_k1_DIS_s013
                                 + 0.776097 w_k1_FDX_s013
                                 + 0.727917 w_k1_IBM_s013
                                 + 0.726229 w_k1_INTC_s013
                                 + 0.765014 w_k1_KO_s013
                                 + 0.872413 w_k1_MAT_s013
                                 + 0.780324 w_k1_MSFT_s013
                                 + 0.84433 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.37309 w_k1_ADM_s013
                                 + 1.23389 w_k1_AXP_s013
                                 + 1.07692 w_k1_DIS_s013 + 1.2247 w_k1_FDX_s013
                                 + 0.740253 w_k1_IBM_s013
                                 + 1.06076 w_k1_INTC_s013
                                 + 0.956372 w_k1_KO_s013
                                 + 1.35101 w_k1_MAT_s013
                                 + 1.21395 w_k1_MSFT_s013
                                 + 1.14775 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.980432 w_k1_ADM_s013
                                 + 0.800509 w_k1_AXP_s013
                                 + 0.910352 w_k1_DIS_s013
                                 + 0.853076 w_k1_FDX_s013
                                 + 1.01515 w_k1_IBM_s013
                                 + 0.71825 w_k1_INTC_s013
                                 + 1.01458 w_k1_KO_s013 + 1.2831 w_k1_MAT_s013
                                 + 1.01204 w_k1_MSFT_s013
                                 + 0.971533 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.689625 w_k1_ADM_s014
                                 + 0.693392 w_k1_AXP_s014
                                 + 0.630888 w_k1_DIS_s014
                                 + 0.860082 w_k1_FDX_s014
                                 + 0.807631 w_k1_IBM_s014
                                 + 0.878113 w_k1_INTC_s014
                                 + 0.756695 w_k1_KO_s014
                                 + 1.02596 w_k1_MAT_s014
                                 + 0.878762 w_k1_MSFT_s014
                                 + 0.924712 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.95816 w_k1_ADM_s014
                                 + 1.18053 w_k1_AXP_s014
                                 + 1.29484 w_k1_DIS_s014
                                 + 1.24747 w_k1_FDX_s014
                                 + 1.06764 w_k1_IBM_s014
                                 + 1.30269 w_k1_INTC_s014
                                 + 1.20172 w_k1_KO_s014 + 1.53196 w_k1_MAT_s014
                                 + 1.21879 w_k1_MSFT_s014
                                 + 1.22012 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.613055 w_k1_ADM_s014
                                 + 0.43833 w_k1_AXP_s014
                                 + 0.57463 w_k1_DIS_s014
                                 + 0.698844 w_k1_FDX_s014
                                 + 0.737477 w_k1_IBM_s014
                                 + 0.824751 w_k1_INTC_s014
                                 + 0.720284 w_k1_KO_s014
                                 + 0.523455 w_k1_MAT_s014
                                 + 0.877202 w_k1_MSFT_s014
                                 + 0.803639 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.741116 w_k1_ADM_s014
                                 + 0.676629 w_k1_AXP_s014
                                 + 0.636756 w_k1_DIS_s014
                                 + 0.767575 w_k1_FDX_s014
                                 + 1.01977 w_k1_IBM_s014
                                 + 1.00992 w_k1_INTC_s014
                                 + 0.85533 w_k1_KO_s014
                                 + 0.894915 w_k1_MAT_s014
                                 + 0.81694 w_k1_MSFT_s014
                                 + 0.765423 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.942769 w_k1_ADM_s014
                                 + 0.832785 w_k1_AXP_s014
                                 + 0.828966 w_k1_DIS_s014
                                 + 0.971013 w_k1_FDX_s014
                                 + 0.974465 w_k1_IBM_s014
                                 + 1.17314 w_k1_INTC_s014
                                 + 0.94535 w_k1_KO_s014 + 1.04375 w_k1_MAT_s014
                                 + 1.01655 w_k1_MSFT_s014
                                 + 1.01581 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.18639 w_k1_ADM_s014
                                 + 1.09225 w_k1_AXP_s014
                                 + 1.65708 w_k1_DIS_s014
                                 + 1.34164 w_k1_FDX_s014
                                 + 1.49867 w_k1_IBM_s014
                                 + 1.52909 w_k1_INTC_s014
                                 + 1.47803 w_k1_KO_s014 + 1.43028 w_k1_MAT_s014
                                 + 1.41562 w_k1_MSFT_s014
                                 + 1.31849 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.04938 w_k1_ADM_s014
                                 + 0.905265 w_k1_AXP_s014
                                 + 1.07335 w_k1_DIS_s014
                                 + 0.922894 w_k1_FDX_s014
                                 + 1.40204 w_k1_IBM_s014
                                 + 0.945814 w_k1_INTC_s014
                                 + 1.21126 w_k1_KO_s014 + 1.13236 w_k1_MAT_s014
                                 + 0.968809 w_k1_MSFT_s014
                                 + 1.03418 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.00242 w_k1_ADM_s014
                                 + 0.969601 w_k1_AXP_s014
                                 + 0.785924 w_k1_DIS_s014
                                 + 1.07563 w_k1_FDX_s014
                                 + 0.651923 w_k1_IBM_s014
                                 + 1.11664 w_k1_INTC_s014
                                 + 0.774399 w_k1_KO_s014
                                 + 0.617291 w_k1_MAT_s014
                                 + 0.94613 w_k1_MSFT_s014
                                 + 0.850667 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.1241 w_k1_ADM_s015
                                 + 1.13395 w_k1_AXP_s015
                                 + 1.14988 w_k1_DIS_s015
                                 + 1.16562 w_k1_FDX_s015
                                 + 0.898631 w_k1_IBM_s015
                                 + 1.02491 w_k1_INTC_s015
                                 + 1.03099 w_k1_KO_s015 + 1.06384 w_k1_MAT_s015
                                 + 1.05165 w_k1_MSFT_s015 + 1.0489 w_k1_PG_s015
                                 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.844342 w_k1_ADM_s015
                                 + 1.13457 w_k1_AXP_s015
                                 + 1.30369 w_k1_DIS_s015
                                 + 1.07217 w_k1_FDX_s015
                                 + 1.38321 w_k1_IBM_s015
                                 + 1.15964 w_k1_INTC_s015 + 1.2753 w_k1_KO_s015
                                 + 1.20314 w_k1_MAT_s015
                                 + 1.06141 w_k1_MSFT_s015
                                 + 1.04878 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.828559 w_k1_ADM_s015
                                 + 0.949463 w_k1_AXP_s015
                                 + 0.991773 w_k1_DIS_s015
                                 + 0.91472 w_k1_FDX_s015
                                 + 0.901258 w_k1_IBM_s015
                                 + 0.798237 w_k1_INTC_s015
                                 + 0.968604 w_k1_KO_s015
                                 + 1.37332 w_k1_MAT_s015
                                 + 0.881388 w_k1_MSFT_s015
                                 + 0.977339 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.24852 w_k1_ADM_s015
                                 + 1.23851 w_k1_AXP_s015
                                 + 1.70514 w_k1_DIS_s015
                                 + 1.27885 w_k1_FDX_s015
                                 + 1.18291 w_k1_IBM_s015
                                 + 1.58425 w_k1_INTC_s015
                                 + 1.55921 w_k1_KO_s015 + 1.05623 w_k1_MAT_s015
                                 + 1.32186 w_k1_MSFT_s015
                                 + 1.12757 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.889475 w_k1_ADM_s015
                                 + 1.14675 w_k1_AXP_s015
                                 + 1.04918 w_k1_DIS_s015
                                 + 0.877245 w_k1_FDX_s015
                                 + 1.42973 w_k1_IBM_s015
                                 + 0.824575 w_k1_INTC_s015
                                 + 1.2252 w_k1_KO_s015 + 1.73138 w_k1_MAT_s015
                                 + 0.836852 w_k1_MSFT_s015
                                 + 0.945049 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.20403 w_k1_ADM_s015
                                 + 1.47883 w_k1_AXP_s015
                                 + 1.34459 w_k1_DIS_s015
                                 + 1.26173 w_k1_FDX_s015
                                 + 1.07954 w_k1_IBM_s015
                                 + 1.33846 w_k1_INTC_s015
                                 + 1.38747 w_k1_KO_s015 + 1.78661 w_k1_MAT_s015
                                 + 1.35393 w_k1_MSFT_s015
                                 + 1.20391 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.03414 w_k1_ADM_s015
                                 + 1.11394 w_k1_AXP_s015
                                 + 0.705878 w_k1_DIS_s015
                                 + 0.92707 w_k1_FDX_s015
                                 + 0.785042 w_k1_IBM_s015
                                 + 0.807236 w_k1_INTC_s015
                                 + 0.831059 w_k1_KO_s015
                                 + 1.05193 w_k1_MAT_s015
                                 + 0.946539 w_k1_MSFT_s015
                                 + 0.875861 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.582715 w_k1_ADM_s015
                                 + 0.651685 w_k1_AXP_s015
                                 + 0.588597 w_k1_DIS_s015
                                 + 0.606618 w_k1_FDX_s015
                                 + 0.818464 w_k1_IBM_s015
                                 + 0.777299 w_k1_INTC_s015
                                 + 0.839056 w_k1_KO_s015
                                 + 0.967598 w_k1_MAT_s015
                                 + 0.706957 w_k1_MSFT_s015
                                 + 0.75662 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.779731 w_k1_ADM_s016
                                 + 1.00703 w_k1_AXP_s016
                                 + 0.924857 w_k1_DIS_s016
                                 + 0.902992 w_k1_FDX_s016
                                 + 1.36715 w_k1_IBM_s016
                                 + 0.958475 w_k1_INTC_s016
                                 + 1.12684 w_k1_KO_s016 + 1.49382 w_k1_MAT_s016
                                 + 0.954097 w_k1_MSFT_s016
                                 + 1.0115 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.15784 w_k1_ADM_s016
                                 + 0.935193 w_k1_AXP_s016
                                 + 1.42724 w_k1_DIS_s016
                                 + 1.06528 w_k1_FDX_s016
                                 + 1.20453 w_k1_IBM_s016
                                 + 1.09218 w_k1_INTC_s016
                                 + 1.26963 w_k1_KO_s016 + 1.06113 w_k1_MAT_s016
                                 + 1.25036 w_k1_MSFT_s016
                                 + 1.11211 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.19754 w_k1_ADM_s016
                                 + 1.37741 w_k1_AXP_s016
                                 + 1.14118 w_k1_DIS_s016
                                 + 1.22463 w_k1_FDX_s016
                                 + 0.659968 w_k1_IBM_s016
                                 + 0.947561 w_k1_INTC_s016
                                 + 0.766817 w_k1_KO_s016
                                 + 0.844886 w_k1_MAT_s016
                                 + 1.05528 w_k1_MSFT_s016
                                 + 0.99669 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.864349 w_k1_ADM_s016
                                 + 0.767889 w_k1_AXP_s016
                                 + 0.712786 w_k1_DIS_s016
                                 + 0.798057 w_k1_FDX_s016
                                 + 0.870808 w_k1_IBM_s016
                                 + 0.916766 w_k1_INTC_s016
                                 + 0.895273 w_k1_KO_s016
                                 + 0.780563 w_k1_MAT_s016
                                 + 0.947828 w_k1_MSFT_s016
                                 + 0.873495 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.08436 w_k1_ADM_s016
                                 + 0.906294 w_k1_AXP_s016
                                 + 1.04452 w_k1_DIS_s016
                                 + 1.02906 w_k1_FDX_s016
                                 + 0.739578 w_k1_IBM_s016
                                 + 0.866453 w_k1_INTC_s016
                                 + 0.878127 w_k1_KO_s016
                                 + 0.643192 w_k1_MAT_s016
                                 + 1.12359 w_k1_MSFT_s016
                                 + 0.981176 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.05975 w_k1_ADM_s016
                                 + 1.46688 w_k1_AXP_s016
                                 + 1.47773 w_k1_DIS_s016
                                 + 1.25487 w_k1_FDX_s016 + 1.7495 w_k1_IBM_s016
                                 + 1.38151 w_k1_INTC_s016
                                 + 1.51275 w_k1_KO_s016 + 1.97109 w_k1_MAT_s016
                                 + 1.27732 w_k1_MSFT_s016
                                 + 1.24139 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.23817 w_k1_ADM_s016
                                 + 1.1087 w_k1_AXP_s016 + 1.41697 w_k1_DIS_s016
                                 + 1.32851 w_k1_FDX_s016
                                 + 0.961512 w_k1_IBM_s016
                                 + 1.21159 w_k1_INTC_s016
                                 + 0.999934 w_k1_KO_s016
                                 + 0.596614 w_k1_MAT_s016
                                 + 1.22448 w_k1_MSFT_s016
                                 + 1.14363 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.993155 w_k1_ADM_s016
                                 + 1.34066 w_k1_AXP_s016
                                 + 1.04182 w_k1_DIS_s016
                                 + 1.06137 w_k1_FDX_s016
                                 + 1.13404 w_k1_IBM_s016
                                 + 1.13841 w_k1_INTC_s016
                                 + 1.11251 w_k1_KO_s016 + 1.46573 w_k1_MAT_s016
                                 + 1.00593 w_k1_MSFT_s016
                                 + 1.01893 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.30268 w_k1_ADM_s017
                                 + 1.14561 w_k1_AXP_s017
                                 + 1.67089 w_k1_DIS_s017
                                 + 1.28233 w_k1_FDX_s017 + 1.4574 w_k1_IBM_s017
                                 + 1.40453 w_k1_INTC_s017
                                 + 1.53975 w_k1_KO_s017 + 1.48253 w_k1_MAT_s017
                                 + 1.59193 w_k1_MSFT_s017
                                 + 1.19541 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.10814 w_k1_ADM_s017
                                 + 0.898678 w_k1_AXP_s017
                                 + 1.07463 w_k1_DIS_s017
                                 + 0.981049 w_k1_FDX_s017
                                 + 0.886058 w_k1_IBM_s017
                                 + 1.12752 w_k1_INTC_s017
                                 + 1.11849 w_k1_KO_s017
                                 + 0.857332 w_k1_MAT_s017
                                 + 1.2697 w_k1_MSFT_s017
                                 + 0.948632 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.92139 w_k1_ADM_s017
                                 + 1.10153 w_k1_AXP_s017
                                 + 0.880814 w_k1_DIS_s017
                                 + 0.949032 w_k1_FDX_s017
                                 + 0.933059 w_k1_IBM_s017
                                 + 1.35384 w_k1_INTC_s017
                                 + 1.05351 w_k1_KO_s017 + 1.19751 w_k1_MAT_s017
                                 + 1.06204 w_k1_MSFT_s017
                                 + 0.988229 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.1339 w_k1_ADM_s017
                                 + 1.2337 w_k1_AXP_s017
                                 + 0.909942 w_k1_DIS_s017
                                 + 0.843716 w_k1_FDX_s017
                                 + 1.34292 w_k1_IBM_s017
                                 + 0.982417 w_k1_INTC_s017
                                 + 1.27354 w_k1_KO_s017 + 1.02843 w_k1_MAT_s017
                                 + 0.876586 w_k1_MSFT_s017
                                 + 0.978295 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.27106 w_k1_ADM_s017
                                 + 1.26927 w_k1_AXP_s017
                                 + 1.35383 w_k1_DIS_s017
                                 + 1.16299 w_k1_FDX_s017
                                 + 1.20444 w_k1_IBM_s017
                                 + 1.41319 w_k1_INTC_s017
                                 + 1.38137 w_k1_KO_s017 + 1.14556 w_k1_MAT_s017
                                 + 1.21633 w_k1_MSFT_s017
                                 + 1.05104 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.846052 w_k1_ADM_s017
                                 + 0.611222 w_k1_AXP_s017
                                 + 0.712848 w_k1_DIS_s017
                                 + 0.743675 w_k1_FDX_s017
                                 + 0.783043 w_k1_IBM_s017
                                 + 1.02127 w_k1_INTC_s017
                                 + 0.867762 w_k1_KO_s017
                                 + 0.531845 w_k1_MAT_s017
                                 + 0.970636 w_k1_MSFT_s017
                                 + 0.721267 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.26238 w_k1_ADM_s017
                                 + 1.11733 w_k1_AXP_s017
                                 + 1.19139 w_k1_DIS_s017
                                 + 1.10186 w_k1_FDX_s017
                                 + 0.853063 w_k1_IBM_s017
                                 + 1.2857 w_k1_INTC_s017 + 1.10674 w_k1_KO_s017
                                 + 0.924688 w_k1_MAT_s017
                                 + 1.18981 w_k1_MSFT_s017
                                 + 1.12699 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.09623 w_k1_ADM_s017
                                 + 1.46766 w_k1_AXP_s017
                                 + 1.57623 w_k1_DIS_s017
                                 + 1.27587 w_k1_FDX_s017
                                 + 0.92122 w_k1_IBM_s017
                                 + 1.50955 w_k1_INTC_s017
                                 + 1.33615 w_k1_KO_s017 + 1.12111 w_k1_MAT_s017
                                 + 1.45885 w_k1_MSFT_s017 + 1.0884 w_k1_PG_s017
                                 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.09343 w_k1_ADM_s100
                                 + 1.14405 w_k1_AXP_s100 + 1.4559 w_k1_DIS_s100
                                 + 1.25004 w_k1_FDX_s100
                                 + 1.05246 w_k1_IBM_s100
                                 + 1.15235 w_k1_INTC_s100
                                 + 1.31557 w_k1_KO_s100 + 1.19464 w_k1_MAT_s100
                                 + 1.23929 w_k1_MSFT_s100
                                 + 1.20069 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.879589 w_k1_ADM_s100
                                 + 1.05417 w_k1_AXP_s100
                                 + 1.07817 w_k1_DIS_s100
                                 + 1.08705 w_k1_FDX_s100
                                 + 0.809661 w_k1_IBM_s100
                                 + 1.14512 w_k1_INTC_s100
                                 + 1.00127 w_k1_KO_s100
                                 + 0.730288 w_k1_MAT_s100
                                 + 1.04318 w_k1_MSFT_s100
                                 + 1.11007 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.769419 w_k1_ADM_s100
                                 + 1.00815 w_k1_AXP_s100
                                 + 1.16981 w_k1_DIS_s100
                                 + 0.916286 w_k1_FDX_s100
                                 + 1.38064 w_k1_IBM_s100
                                 + 1.10043 w_k1_INTC_s100 + 1.3408 w_k1_KO_s100
                                 + 1.35775 w_k1_MAT_s100
                                 + 1.05807 w_k1_MSFT_s100
                                 + 1.12702 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.635384 w_k1_ADM_s100
                                 + 0.675975 w_k1_AXP_s100
                                 + 0.728249 w_k1_DIS_s100
                                 + 0.800718 w_k1_FDX_s100
                                 + 0.661032 w_k1_IBM_s100
                                 + 0.895443 w_k1_INTC_s100
                                 + 0.797007 w_k1_KO_s100
                                 + 0.681106 w_k1_MAT_s100
                                 + 0.916275 w_k1_MSFT_s100
                                 + 0.938779 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.671447 w_k1_ADM_s100
                                 + 0.93838 w_k1_AXP_s100
                                 + 0.776325 w_k1_DIS_s100
                                 + 0.871636 w_k1_FDX_s100
                                 + 0.898987 w_k1_IBM_s100
                                 + 0.802697 w_k1_INTC_s100
                                 + 0.876389 w_k1_KO_s100
                                 + 1.24383 w_k1_MAT_s100
                                 + 0.865198 w_k1_MSFT_s100
                                 + 0.994841 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.05404 w_k1_ADM_s100
                                 + 1.84517 w_k1_AXP_s100
                                 + 1.72857 w_k1_DIS_s100
                                 + 1.34353 w_k1_FDX_s100
                                 + 1.65552 w_k1_IBM_s100
                                 + 1.30428 w_k1_INTC_s100
                                 + 1.68224 w_k1_KO_s100 + 2.19102 w_k1_MAT_s100
                                 + 1.30231 w_k1_MSFT_s100
                                 + 1.39172 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.855563 w_k1_ADM_s100
                                 + 0.667365 w_k1_AXP_s100
                                 + 0.864623 w_k1_DIS_s100
                                 + 0.873469 w_k1_FDX_s100
                                 + 0.851222 w_k1_IBM_s100
                                 + 0.99946 w_k1_INTC_s100
                                 + 0.92616 w_k1_KO_s100
                                 + 0.779086 w_k1_MAT_s100
                                 + 1.04733 w_k1_MSFT_s100
                                 + 0.919528 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.74347 w_k1_ADM_s100
                                 + 0.833489 w_k1_AXP_s100
                                 + 0.914394 w_k1_DIS_s100
                                 + 0.903099 w_k1_FDX_s100
                                 + 1.0609 w_k1_IBM_s100
                                 + 0.880457 w_k1_INTC_s100
                                 + 0.955662 w_k1_KO_s100
                                 + 1.26456 w_k1_MAT_s100
                                 + 0.973952 w_k1_MSFT_s100
                                 + 0.985262 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.19664 w_k1_ADM_s101
                                 + 0.836428 w_k1_AXP_s101
                                 + 0.963342 w_k1_DIS_s101
                                 + 0.843586 w_k1_FDX_s101
                                 + 1.03613 w_k1_IBM_s101
                                 + 0.892932 w_k1_INTC_s101
                                 + 0.879189 w_k1_KO_s101
                                 + 0.810111 w_k1_MAT_s101
                                 + 0.935292 w_k1_MSFT_s101
                                 + 0.894416 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.19683 w_k1_ADM_s101
                                 + 0.913818 w_k1_AXP_s101
                                 + 1.03656 w_k1_DIS_s101
                                 + 0.977408 w_k1_FDX_s101
                                 + 0.832923 w_k1_IBM_s101
                                 + 1.10427 w_k1_INTC_s101
                                 + 0.916252 w_k1_KO_s101
                                 + 0.743552 w_k1_MAT_s101
                                 + 1.14505 w_k1_MSFT_s101
                                 + 0.981181 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.858978 w_k1_ADM_s101
                                 + 0.628611 w_k1_AXP_s101
                                 + 0.691891 w_k1_DIS_s101
                                 + 0.678632 w_k1_FDX_s101
                                 + 0.753193 w_k1_IBM_s101
                                 + 1.0533 w_k1_INTC_s101
                                 + 0.649473 w_k1_KO_s101
                                 + 0.469575 w_k1_MAT_s101
                                 + 0.827789 w_k1_MSFT_s101
                                 + 0.83618 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.52357 w_k1_ADM_s101
                                 + 1.53453 w_k1_AXP_s101
                                 + 1.50909 w_k1_DIS_s101
                                 + 1.41264 w_k1_FDX_s101
                                 + 1.03507 w_k1_IBM_s101
                                 + 1.43883 w_k1_INTC_s101
                                 + 1.10713 w_k1_KO_s101 + 1.43074 w_k1_MAT_s101
                                 + 1.45715 w_k1_MSFT_s101
                                 + 1.15724 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.946574 w_k1_ADM_s101
                                 + 0.865007 w_k1_AXP_s101
                                 + 0.750817 w_k1_DIS_s101
                                 + 0.753828 w_k1_FDX_s101
                                 + 1.20592 w_k1_IBM_s101
                                 + 0.784973 w_k1_INTC_s101
                                 + 0.998162 w_k1_KO_s101
                                 + 0.990418 w_k1_MAT_s101
                                 + 0.827539 w_k1_MSFT_s101
                                 + 0.967615 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.14814 w_k1_ADM_s101
                                 + 1.37552 w_k1_AXP_s101
                                 + 0.861123 w_k1_DIS_s101
                                 + 1.04112 w_k1_FDX_s101
                                 + 0.828364 w_k1_IBM_s101
                                 + 1.21226 w_k1_INTC_s101
                                 + 0.760529 w_k1_KO_s101 + 0.6787 w_k1_MAT_s101
                                 + 0.996065 w_k1_MSFT_s101
                                 + 0.927356 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.12772 w_k1_ADM_s101
                                 + 1.26287 w_k1_AXP_s101
                                 + 1.20075 w_k1_DIS_s101
                                 + 1.10018 w_k1_FDX_s101 + 1.2221 w_k1_IBM_s101
                                 + 1.28084 w_k1_INTC_s101
                                 + 0.995064 w_k1_KO_s101
                                 + 1.21616 w_k1_MAT_s101
                                 + 1.21524 w_k1_MSFT_s101
                                 + 1.05987 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.37258 w_k1_ADM_s101
                                 + 1.45977 w_k1_AXP_s101
                                 + 1.63204 w_k1_DIS_s101
                                 + 1.24438 w_k1_FDX_s101
                                 + 1.53197 w_k1_IBM_s101
                                 + 1.15295 w_k1_INTC_s101
                                 + 1.21668 w_k1_KO_s101 + 1.27185 w_k1_MAT_s101
                                 + 1.16761 w_k1_MSFT_s101
                                 + 1.22931 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.984235 w_k1_ADM_s102
                                 + 0.925644 w_k1_AXP_s102
                                 + 0.54526 w_k1_DIS_s102
                                 + 0.71044 w_k1_FDX_s102
                                 + 0.976906 w_k1_IBM_s102
                                 + 0.94342 w_k1_INTC_s102
                                 + 0.889795 w_k1_KO_s102
                                 + 0.876986 w_k1_MAT_s102
                                 + 0.746961 w_k1_MSFT_s102
                                 + 0.849763 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.18677 w_k1_ADM_s102
                                 + 1.17564 w_k1_AXP_s102
                                 + 1.45611 w_k1_DIS_s102
                                 + 0.947511 w_k1_FDX_s102
                                 + 1.96397 w_k1_IBM_s102
                                 + 1.18662 w_k1_INTC_s102 + 1.8626 w_k1_KO_s102
                                 + 1.83704 w_k1_MAT_s102
                                 + 1.30853 w_k1_MSFT_s102
                                 + 1.09252 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.34595 w_k1_ADM_s102
                                 + 1.38548 w_k1_AXP_s102
                                 + 1.10854 w_k1_DIS_s102 + 1.1356 w_k1_FDX_s102
                                 + 1.36672 w_k1_IBM_s102
                                 + 1.28566 w_k1_INTC_s102
                                 + 1.35166 w_k1_KO_s102 + 1.56826 w_k1_MAT_s102
                                 + 0.988695 w_k1_MSFT_s102
                                 + 0.970057 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.04352 w_k1_ADM_s102
                                 + 1.43824 w_k1_AXP_s102
                                 + 1.53465 w_k1_DIS_s102
                                 + 1.17827 w_k1_FDX_s102
                                 + 1.47291 w_k1_IBM_s102
                                 + 1.32367 w_k1_INTC_s102
                                 + 1.52882 w_k1_KO_s102 + 1.74472 w_k1_MAT_s102
                                 + 1.20169 w_k1_MSFT_s102
                                 + 1.29279 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.1698 w_k1_ADM_s102
                                 + 1.05016 w_k1_AXP_s102
                                 + 1.14778 w_k1_DIS_s102
                                 + 1.13936 w_k1_FDX_s102
                                 + 0.808478 w_k1_IBM_s102
                                 + 1.27666 w_k1_INTC_s102
                                 + 1.02157 w_k1_KO_s102
                                 + 0.852172 w_k1_MAT_s102
                                 + 1.22049 w_k1_MSFT_s102
                                 + 0.985608 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.782548 w_k1_ADM_s102
                                 + 0.792289 w_k1_AXP_s102
                                 + 0.694356 w_k1_DIS_s102
                                 + 0.83939 w_k1_FDX_s102
                                 + 0.680315 w_k1_IBM_s102
                                 + 0.848361 w_k1_INTC_s102
                                 + 0.890111 w_k1_KO_s102
                                 + 0.835288 w_k1_MAT_s102
                                 + 0.792804 w_k1_MSFT_s102
                                 + 0.838641 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.25687 w_k1_ADM_s102
                                 + 1.14178 w_k1_AXP_s102
                                 + 1.12393 w_k1_DIS_s102
                                 + 1.05829 w_k1_FDX_s102
                                 + 1.19931 w_k1_IBM_s102
                                 + 1.29986 w_k1_INTC_s102
                                 + 1.27865 w_k1_KO_s102
                                 + 0.873205 w_k1_MAT_s102
                                 + 1.07537 w_k1_MSFT_s102
                                 + 1.03095 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.12283 w_k1_ADM_s102
                                 + 1.06391 w_k1_AXP_s102
                                 + 0.97319 w_k1_DIS_s102
                                 + 0.963455 w_k1_FDX_s102
                                 + 1.02731 w_k1_IBM_s102
                                 + 1.32083 w_k1_INTC_s102
                                 + 1.19951 w_k1_KO_s102 + 1.24073 w_k1_MAT_s102
                                 + 1.00881 w_k1_MSFT_s102
                                 + 0.952951 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.4395 w_k1_ADM_s103
                                 + 1.23037 w_k1_AXP_s103
                                 + 1.66396 w_k1_DIS_s103
                                 + 1.54241 w_k1_FDX_s103
                                 + 0.938109 w_k1_IBM_s103
                                 + 1.23941 w_k1_INTC_s103
                                 + 1.22047 w_k1_KO_s103
                                 + 0.910386 w_k1_MAT_s103
                                 + 1.32686 w_k1_MSFT_s103
                                 + 1.07141 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.42662 w_k1_ADM_s103
                                 + 1.10418 w_k1_AXP_s103
                                 + 1.28377 w_k1_DIS_s103
                                 + 1.32571 w_k1_FDX_s103
                                 + 0.870204 w_k1_IBM_s103
                                 + 1.05907 w_k1_INTC_s103
                                 + 1.16293 w_k1_KO_s103
                                 + 0.946579 w_k1_MAT_s103
                                 + 1.37446 w_k1_MSFT_s103
                                 + 1.05757 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.38797 w_k1_ADM_s103
                                 + 1.18487 w_k1_AXP_s103
                                 + 1.43447 w_k1_DIS_s103
                                 + 1.24781 w_k1_FDX_s103
                                 + 1.37574 w_k1_IBM_s103
                                 + 1.03054 w_k1_INTC_s103
                                 + 1.29859 w_k1_KO_s103 + 1.11186 w_k1_MAT_s103
                                 + 1.14404 w_k1_MSFT_s103
                                 + 1.14252 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.912087 w_k1_ADM_s103
                                 + 0.927959 w_k1_AXP_s103
                                 + 0.72423 w_k1_DIS_s103
                                 + 0.809461 w_k1_FDX_s103
                                 + 1.22982 w_k1_IBM_s103
                                 + 0.991481 w_k1_INTC_s103
                                 + 0.952192 w_k1_KO_s103
                                 + 1.19907 w_k1_MAT_s103
                                 + 0.900401 w_k1_MSFT_s103
                                 + 0.830562 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.961036 w_k1_ADM_s103
                                 + 0.792908 w_k1_AXP_s103
                                 + 0.832598 w_k1_DIS_s103
                                 + 1.08184 w_k1_FDX_s103
                                 + 0.651882 w_k1_IBM_s103
                                 + 1.14135 w_k1_INTC_s103
                                 + 0.798395 w_k1_KO_s103
                                 + 0.631523 w_k1_MAT_s103
                                 + 1.05137 w_k1_MSFT_s103
                                 + 0.832547 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.13486 w_k1_ADM_s103
                                 + 1.15773 w_k1_AXP_s103
                                 + 1.11321 w_k1_DIS_s103
                                 + 0.971672 w_k1_FDX_s103
                                 + 1.8444 w_k1_IBM_s103
                                 + 1.02428 w_k1_INTC_s103
                                 + 1.34524 w_k1_KO_s103 + 1.20649 w_k1_MAT_s103
                                 + 1.06426 w_k1_MSFT_s103
                                 + 0.978556 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.1768 w_k1_ADM_s103
                                 + 1.13233 w_k1_AXP_s103
                                 + 1.17885 w_k1_DIS_s103
                                 + 1.16213 w_k1_FDX_s103
                                 + 1.04229 w_k1_IBM_s103
                                 + 1.12658 w_k1_INTC_s103
                                 + 1.03884 w_k1_KO_s103 + 1.09367 w_k1_MAT_s103
                                 + 1.13463 w_k1_MSFT_s103
                                 + 0.96458 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.871047 w_k1_ADM_s103
                                 + 0.8121 w_k1_AXP_s103
                                 + 0.684932 w_k1_DIS_s103
                                 + 0.896654 w_k1_FDX_s103
                                 + 0.88868 w_k1_IBM_s103
                                 + 0.862491 w_k1_INTC_s103
                                 + 0.877648 w_k1_KO_s103
                                 + 1.04863 w_k1_MAT_s103
                                 + 0.899716 w_k1_MSFT_s103
                                 + 0.764221 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.826255 w_k1_ADM_s104
                                 + 1.00039 w_k1_AXP_s104
                                 + 0.919697 w_k1_DIS_s104
                                 + 0.791423 w_k1_FDX_s104
                                 + 1.61841 w_k1_IBM_s104
                                 + 1.14069 w_k1_INTC_s104
                                 + 1.04641 w_k1_KO_s104
                                 + 0.861837 w_k1_MAT_s104
                                 + 0.923077 w_k1_MSFT_s104
                                 + 0.847152 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.800881 w_k1_ADM_s104
                                 + 0.742112 w_k1_AXP_s104
                                 + 0.646098 w_k1_DIS_s104
                                 + 0.785752 w_k1_FDX_s104
                                 + 0.815184 w_k1_IBM_s104
                                 + 1.00391 w_k1_INTC_s104
                                 + 0.812631 w_k1_KO_s104
                                 + 0.488677 w_k1_MAT_s104
                                 + 0.848778 w_k1_MSFT_s104
                                 + 0.900993 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.08873 w_k1_ADM_s104
                                 + 1.48034 w_k1_AXP_s104
                                 + 1.67842 w_k1_DIS_s104
                                 + 1.57122 w_k1_FDX_s104
                                 + 1.22886 w_k1_IBM_s104
                                 + 1.24773 w_k1_INTC_s104 + 1.4193 w_k1_KO_s104
                                 + 0.973756 w_k1_MAT_s104
                                 + 1.42588 w_k1_MSFT_s104
                                 + 1.16936 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.99332 w_k1_ADM_s104
                                 + 1.12207 w_k1_AXP_s104
                                 + 0.810065 w_k1_DIS_s104
                                 + 0.962213 w_k1_FDX_s104
                                 + 1.01547 w_k1_IBM_s104
                                 + 0.836979 w_k1_INTC_s104
                                 + 0.843036 w_k1_KO_s104
                                 + 0.822927 w_k1_MAT_s104
                                 + 0.891743 w_k1_MSFT_s104
                                 + 1.02552 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.11096 w_k1_ADM_s104
                                 + 0.755618 w_k1_AXP_s104
                                 + 1.13757 w_k1_DIS_s104 + 1.1844 w_k1_FDX_s104
                                 + 0.684596 w_k1_IBM_s104
                                 + 1.13267 w_k1_INTC_s104
                                 + 0.919713 w_k1_KO_s104
                                 + 0.37525 w_k1_MAT_s104
                                 + 1.34128 w_k1_MSFT_s104
                                 + 0.985765 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.977866 w_k1_ADM_s104
                                 + 1.22242 w_k1_AXP_s104
                                 + 1.12129 w_k1_DIS_s104
                                 + 1.22289 w_k1_FDX_s104
                                 + 0.941934 w_k1_IBM_s104
                                 + 1.31835 w_k1_INTC_s104
                                 + 1.06621 w_k1_KO_s104 + 1.10332 w_k1_MAT_s104
                                 + 1.15729 w_k1_MSFT_s104
                                 + 1.02667 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.741968 w_k1_ADM_s104
                                 + 0.935487 w_k1_AXP_s104
                                 + 1.0504 w_k1_DIS_s104 + 1.00813 w_k1_FDX_s104
                                 + 0.845038 w_k1_IBM_s104
                                 + 1.00185 w_k1_INTC_s104
                                 + 0.890741 w_k1_KO_s104
                                 + 0.824809 w_k1_MAT_s104
                                 + 1.05985 w_k1_MSFT_s104
                                 + 0.910069 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.883834 w_k1_ADM_s104
                                 + 0.863978 w_k1_AXP_s104
                                 + 1.14256 w_k1_DIS_s104
                                 + 0.955228 w_k1_FDX_s104
                                 + 1.35426 w_k1_IBM_s104
                                 + 1.24757 w_k1_INTC_s104 + 1.1975 w_k1_KO_s104
                                 + 0.770113 w_k1_MAT_s104
                                 + 1.29523 w_k1_MSFT_s104
                                 + 1.03014 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.05941 w_k1_ADM_s105
                                 + 1.49594 w_k1_AXP_s105
                                 + 1.54936 w_k1_DIS_s105
                                 + 1.42261 w_k1_FDX_s105
                                 + 1.21608 w_k1_IBM_s105
                                 + 1.22841 w_k1_INTC_s105
                                 + 1.29785 w_k1_KO_s105 + 1.65323 w_k1_MAT_s105
                                 + 1.3932 w_k1_MSFT_s105 + 1.22838 w_k1_PG_s105
                                 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.653386 w_k1_ADM_s105
                                 + 0.771672 w_k1_AXP_s105
                                 + 0.762162 w_k1_DIS_s105
                                 + 0.913063 w_k1_FDX_s105
                                 + 0.766658 w_k1_IBM_s105
                                 + 0.763302 w_k1_INTC_s105
                                 + 0.797846 w_k1_KO_s105
                                 + 0.921623 w_k1_MAT_s105
                                 + 0.857071 w_k1_MSFT_s105
                                 + 0.999002 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.09192 w_k1_ADM_s105
                                 + 1.03297 w_k1_AXP_s105
                                 + 1.45073 w_k1_DIS_s105
                                 + 1.24526 w_k1_FDX_s105
                                 + 1.47692 w_k1_IBM_s105
                                 + 1.21363 w_k1_INTC_s105
                                 + 1.34285 w_k1_KO_s105 + 1.19839 w_k1_MAT_s105
                                 + 1.19008 w_k1_MSFT_s105
                                 + 1.21014 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.663682 w_k1_ADM_s105
                                 + 0.648042 w_k1_AXP_s105
                                 + 0.78861 w_k1_DIS_s105
                                 + 0.745665 w_k1_FDX_s105
                                 + 1.41348 w_k1_IBM_s105
                                 + 0.75265 w_k1_INTC_s105
                                 + 1.13374 w_k1_KO_s105 + 1.22979 w_k1_MAT_s105
                                 + 0.77048 w_k1_MSFT_s105
                                 + 0.809931 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.977723 w_k1_ADM_s105
                                 + 1.01301 w_k1_AXP_s105
                                 + 0.964883 w_k1_DIS_s105
                                 + 1.01566 w_k1_FDX_s105
                                 + 1.41453 w_k1_IBM_s105
                                 + 0.939923 w_k1_INTC_s105
                                 + 1.33393 w_k1_KO_s105 + 1.20453 w_k1_MAT_s105
                                 + 1.0511 w_k1_MSFT_s105 + 1.00643 w_k1_PG_s105
                                 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.848146 w_k1_ADM_s105
                                 + 0.69372 w_k1_AXP_s105
                                 + 0.874711 w_k1_DIS_s105
                                 + 0.901318 w_k1_FDX_s105
                                 + 0.973003 w_k1_IBM_s105
                                 + 0.798368 w_k1_INTC_s105
                                 + 0.895719 w_k1_KO_s105
                                 + 0.888661 w_k1_MAT_s105
                                 + 0.855082 w_k1_MSFT_s105
                                 + 1.02623 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.41151 w_k1_ADM_s105
                                 + 1.16921 w_k1_AXP_s105
                                 + 1.45343 w_k1_DIS_s105 + 1.3897 w_k1_FDX_s105
                                 + 0.902477 w_k1_IBM_s105
                                 + 1.26387 w_k1_INTC_s105
                                 + 1.01282 w_k1_KO_s105
                                 + 0.722639 w_k1_MAT_s105
                                 + 1.191 w_k1_MSFT_s105 + 1.11021 w_k1_PG_s105
                                 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.989452 w_k1_ADM_s105
                                 + 1.15321 w_k1_AXP_s105
                                 + 0.878032 w_k1_DIS_s105
                                 + 1.15255 w_k1_FDX_s105
                                 + 0.947864 w_k1_IBM_s105
                                 + 1.03504 w_k1_INTC_s105
                                 + 1.05802 w_k1_KO_s105 + 1.11356 w_k1_MAT_s105
                                 + 0.915503 w_k1_MSFT_s105
                                 + 0.913604 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.952248 w_k1_ADM_s106
                                 + 1.16008 w_k1_AXP_s106
                                 + 0.840567 w_k1_DIS_s106
                                 + 1.00025 w_k1_FDX_s106
                                 + 1.03445 w_k1_IBM_s106
                                 + 0.909545 w_k1_INTC_s106
                                 + 0.874158 w_k1_KO_s106
                                 + 1.07905 w_k1_MAT_s106
                                 + 0.974187 w_k1_MSFT_s106
                                 + 0.880662 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.07206 w_k1_ADM_s106
                                 + 1.175 w_k1_AXP_s106 + 0.901333 w_k1_DIS_s106
                                 + 0.893628 w_k1_FDX_s106
                                 + 1.32222 w_k1_IBM_s106
                                 + 0.780445 w_k1_INTC_s106
                                 + 1.00907 w_k1_KO_s106 + 1.13208 w_k1_MAT_s106
                                 + 0.82137 w_k1_MSFT_s106
                                 + 0.968058 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.806965 w_k1_ADM_s106
                                 + 0.874593 w_k1_AXP_s106
                                 + 0.805939 w_k1_DIS_s106
                                 + 0.796901 w_k1_FDX_s106
                                 + 1.23482 w_k1_IBM_s106
                                 + 0.844661 w_k1_INTC_s106
                                 + 0.892871 w_k1_KO_s106
                                 + 1.24458 w_k1_MAT_s106
                                 + 0.936847 w_k1_MSFT_s106
                                 + 0.836846 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.843301 w_k1_ADM_s106
                                 + 0.861504 w_k1_AXP_s106
                                 + 0.896499 w_k1_DIS_s106
                                 + 0.847482 w_k1_FDX_s106
                                 + 0.932115 w_k1_IBM_s106
                                 + 0.789229 w_k1_INTC_s106
                                 + 0.869934 w_k1_KO_s106
                                 + 0.911733 w_k1_MAT_s106
                                 + 0.865851 w_k1_MSFT_s106
                                 + 0.956354 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.10839 w_k1_ADM_s106
                                 + 0.990379 w_k1_AXP_s106
                                 + 0.993156 w_k1_DIS_s106
                                 + 1.07776 w_k1_FDX_s106
                                 + 0.738279 w_k1_IBM_s106
                                 + 0.977323 w_k1_INTC_s106
                                 + 0.942184 w_k1_KO_s106
                                 + 0.725144 w_k1_MAT_s106
                                 + 1.0703 w_k1_MSFT_s106
                                 + 0.859373 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.742327 w_k1_ADM_s106
                                 + 0.682233 w_k1_AXP_s106
                                 + 0.653228 w_k1_DIS_s106
                                 + 0.762171 w_k1_FDX_s106
                                 + 0.773828 w_k1_IBM_s106
                                 + 0.708941 w_k1_INTC_s106
                                 + 0.795594 w_k1_KO_s106
                                 + 0.616376 w_k1_MAT_s106
                                 + 0.656578 w_k1_MSFT_s106
                                 + 0.741257 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.26617 w_k1_ADM_s106
                                 + 1.29437 w_k1_AXP_s106
                                 + 1.63586 w_k1_DIS_s106
                                 + 1.45563 w_k1_FDX_s106
                                 + 0.961056 w_k1_IBM_s106
                                 + 1.23237 w_k1_INTC_s106 + 1.2199 w_k1_KO_s106
                                 + 1.1996 w_k1_MAT_s106
                                 + 1.49837 w_k1_MSFT_s106
                                 + 1.21583 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.28935 w_k1_ADM_s106
                                 + 1.08004 w_k1_AXP_s106
                                 + 1.27442 w_k1_DIS_s106
                                 + 1.07244 w_k1_FDX_s106
                                 + 1.42593 w_k1_IBM_s106
                                 + 0.9035 w_k1_INTC_s106 + 1.39264 w_k1_KO_s106
                                 + 1.22087 w_k1_MAT_s106
                                 + 1.17695 w_k1_MSFT_s106
                                 + 1.06523 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.27314 w_k1_ADM_s107
                                 + 0.974523 w_k1_AXP_s107
                                 + 1.11339 w_k1_DIS_s107
                                 + 0.995779 w_k1_FDX_s107
                                 + 1.47348 w_k1_IBM_s107
                                 + 1.11713 w_k1_INTC_s107
                                 + 1.52395 w_k1_KO_s107 + 1.24462 w_k1_MAT_s107
                                 + 1.11342 w_k1_MSFT_s107
                                 + 1.15916 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.08587 w_k1_ADM_s107
                                 + 0.643293 w_k1_AXP_s107
                                 + 0.97919 w_k1_DIS_s107
                                 + 0.858801 w_k1_FDX_s107
                                 + 0.851891 w_k1_IBM_s107
                                 + 0.890732 w_k1_INTC_s107
                                 + 1.14222 w_k1_KO_s107
                                 + 0.803841 w_k1_MAT_s107
                                 + 1.08492 w_k1_MSFT_s107
                                 + 1.04094 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.01393 w_k1_ADM_s107
                                 + 0.816276 w_k1_AXP_s107
                                 + 0.862269 w_k1_DIS_s107
                                 + 0.967428 w_k1_FDX_s107
                                 + 0.820605 w_k1_IBM_s107
                                 + 1.00687 w_k1_INTC_s107
                                 + 1.09009 w_k1_KO_s107
                                 + 0.714712 w_k1_MAT_s107
                                 + 0.97996 w_k1_MSFT_s107
                                 + 0.961543 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.876153 w_k1_ADM_s107
                                 + 0.605549 w_k1_AXP_s107
                                 + 0.658437 w_k1_DIS_s107
                                 + 0.676672 w_k1_FDX_s107
                                 + 0.848433 w_k1_IBM_s107
                                 + 0.727758 w_k1_INTC_s107
                                 + 0.992384 w_k1_KO_s107
                                 + 0.76672 w_k1_MAT_s107
                                 + 0.812683 w_k1_MSFT_s107
                                 + 0.850286 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.08819 w_k1_ADM_s107
                                 + 0.973276 w_k1_AXP_s107
                                 + 0.696169 w_k1_DIS_s107
                                 + 0.772358 w_k1_FDX_s107
                                 + 1.09893 w_k1_IBM_s107
                                 + 0.879994 w_k1_INTC_s107
                                 + 1.17259 w_k1_KO_s107 + 1.18677 w_k1_MAT_s107
                                 + 0.861375 w_k1_MSFT_s107
                                 + 0.937691 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.948258 w_k1_ADM_s107
                                 + 0.854945 w_k1_AXP_s107
                                 + 0.717568 w_k1_DIS_s107
                                 + 0.820965 w_k1_FDX_s107
                                 + 0.849798 w_k1_IBM_s107
                                 + 0.887447 w_k1_INTC_s107
                                 + 0.833339 w_k1_KO_s107
                                 + 0.956039 w_k1_MAT_s107
                                 + 0.805072 w_k1_MSFT_s107
                                 + 0.853468 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.18833 w_k1_ADM_s107
                                 + 1.32889 w_k1_AXP_s107
                                 + 0.94622 w_k1_DIS_s107
                                 + 1.13632 w_k1_FDX_s107
                                 + 0.965208 w_k1_IBM_s107
                                 + 1.12145 w_k1_INTC_s107
                                 + 1.08053 w_k1_KO_s107 + 1.27793 w_k1_MAT_s107
                                 + 1.06497 w_k1_MSFT_s107
                                 + 1.10407 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.31169 w_k1_ADM_s107
                                 + 1.33231 w_k1_AXP_s107
                                 + 1.36387 w_k1_DIS_s107
                                 + 1.37388 w_k1_FDX_s107
                                 + 1.00583 w_k1_IBM_s107
                                 + 1.21497 w_k1_INTC_s107
                                 + 1.39762 w_k1_KO_s107 + 1.42215 w_k1_MAT_s107
                                 + 1.34938 w_k1_MSFT_s107 + 1.1854 w_k1_PG_s107
                                 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.42832 w_k1_ADM_s110
                                 + 1.49926 w_k1_AXP_s110
                                 + 1.73672 w_k1_DIS_s110
                                 + 1.39133 w_k1_FDX_s110
                                 + 1.15817 w_k1_IBM_s110
                                 + 1.2503 w_k1_INTC_s110 + 1.3062 w_k1_KO_s110
                                 + 1.0895 w_k1_MAT_s110
                                 + 1.20072 w_k1_MSFT_s110
                                 + 1.43562 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.09303 w_k1_ADM_s110
                                 + 0.852935 w_k1_AXP_s110
                                 + 0.691666 w_k1_DIS_s110
                                 + 0.927042 w_k1_FDX_s110
                                 + 0.704084 w_k1_IBM_s110
                                 + 0.971332 w_k1_INTC_s110
                                 + 0.820865 w_k1_KO_s110
                                 + 0.495841 w_k1_MAT_s110
                                 + 0.864486 w_k1_MSFT_s110
                                 + 0.893797 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.3502 w_k1_ADM_s110
                                 + 1.31094 w_k1_AXP_s110
                                 + 0.945653 w_k1_DIS_s110
                                 + 1.15937 w_k1_FDX_s110
                                 + 1.01035 w_k1_IBM_s110
                                 + 1.05755 w_k1_INTC_s110
                                 + 0.972883 w_k1_KO_s110
                                 + 0.910093 w_k1_MAT_s110
                                 + 0.923291 w_k1_MSFT_s110
                                 + 1.06527 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.662021 w_k1_ADM_s110
                                 + 0.643781 w_k1_AXP_s110
                                 + 0.572801 w_k1_DIS_s110
                                 + 0.693293 w_k1_FDX_s110
                                 + 0.871984 w_k1_IBM_s110
                                 + 0.884821 w_k1_INTC_s110
                                 + 0.818986 w_k1_KO_s110
                                 + 0.529517 w_k1_MAT_s110
                                 + 0.757243 w_k1_MSFT_s110
                                 + 0.812558 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.797691 w_k1_ADM_s110
                                 + 0.930312 w_k1_AXP_s110
                                 + 0.716803 w_k1_DIS_s110
                                 + 0.876697 w_k1_FDX_s110
                                 + 1.15973 w_k1_IBM_s110
                                 + 1.03421 w_k1_INTC_s110
                                 + 0.909526 w_k1_KO_s110
                                 + 0.812061 w_k1_MAT_s110
                                 + 0.891323 w_k1_MSFT_s110
                                 + 0.948354 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.918475 w_k1_ADM_s110
                                 + 0.853128 w_k1_AXP_s110
                                 + 0.968331 w_k1_DIS_s110
                                 + 0.915374 w_k1_FDX_s110
                                 + 0.976483 w_k1_IBM_s110 + 1.09 w_k1_INTC_s110
                                 + 0.995554 w_k1_KO_s110
                                 + 0.503378 w_k1_MAT_s110
                                 + 1.00298 w_k1_MSFT_s110
                                 + 1.02093 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.975536 w_k1_ADM_s110
                                 + 1.12574 w_k1_AXP_s110
                                 + 1.27033 w_k1_DIS_s110
                                 + 1.08252 w_k1_FDX_s110 + 1.2482 w_k1_IBM_s110
                                 + 1.17488 w_k1_INTC_s110
                                 + 1.37714 w_k1_KO_s110 + 1.04609 w_k1_MAT_s110
                                 + 1.00348 w_k1_MSFT_s110
                                 + 1.18445 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.795877 w_k1_ADM_s110
                                 + 0.772376 w_k1_AXP_s110
                                 + 0.804278 w_k1_DIS_s110
                                 + 0.849425 w_k1_FDX_s110
                                 + 0.763842 w_k1_IBM_s110
                                 + 0.98779 w_k1_INTC_s110
                                 + 0.880504 w_k1_KO_s110
                                 + 0.512458 w_k1_MAT_s110
                                 + 0.926958 w_k1_MSFT_s110
                                 + 0.938557 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.771015 w_k1_ADM_s111
                                 + 1.00705 w_k1_AXP_s111
                                 + 0.981727 w_k1_DIS_s111
                                 + 0.906813 w_k1_FDX_s111
                                 + 1.38175 w_k1_IBM_s111
                                 + 1.05348 w_k1_INTC_s111
                                 + 0.974379 w_k1_KO_s111
                                 + 1.40882 w_k1_MAT_s111
                                 + 0.994904 w_k1_MSFT_s111
                                 + 0.991435 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.692374 w_k1_ADM_s111
                                 + 0.920627 w_k1_AXP_s111
                                 + 0.725597 w_k1_DIS_s111
                                 + 0.811434 w_k1_FDX_s111
                                 + 1.07539 w_k1_IBM_s111
                                 + 0.803952 w_k1_INTC_s111
                                 + 0.916485 w_k1_KO_s111
                                 + 1.12536 w_k1_MAT_s111
                                 + 0.807215 w_k1_MSFT_s111
                                 + 0.933745 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.04457 w_k1_ADM_s111
                                 + 1.38868 w_k1_AXP_s111
                                 + 1.07324 w_k1_DIS_s111
                                 + 1.17132 w_k1_FDX_s111
                                 + 1.14318 w_k1_IBM_s111
                                 + 1.32694 w_k1_INTC_s111
                                 + 1.04632 w_k1_KO_s111 + 1.14863 w_k1_MAT_s111
                                 + 1.17722 w_k1_MSFT_s111
                                 + 1.11109 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.01482 w_k1_ADM_s111
                                 + 1.27405 w_k1_AXP_s111
                                 + 1.17156 w_k1_DIS_s111
                                 + 0.995684 w_k1_FDX_s111
                                 + 1.84499 w_k1_IBM_s111
                                 + 1.05715 w_k1_INTC_s111 + 1.4711 w_k1_KO_s111
                                 + 2.03577 w_k1_MAT_s111
                                 + 1.09816 w_k1_MSFT_s111
                                 + 1.22224 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.880725 w_k1_ADM_s111
                                 + 1.09146 w_k1_AXP_s111
                                 + 1.41222 w_k1_DIS_s111
                                 + 1.10479 w_k1_FDX_s111
                                 + 1.11643 w_k1_IBM_s111
                                 + 1.18617 w_k1_INTC_s111
                                 + 1.21117 w_k1_KO_s111 + 1.0339 w_k1_MAT_s111
                                 + 1.13804 w_k1_MSFT_s111
                                 + 1.01547 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.31825 w_k1_ADM_s111
                                 + 1.46896 w_k1_AXP_s111
                                 + 1.48313 w_k1_DIS_s111 + 1.3217 w_k1_FDX_s111
                                 + 1.3273 w_k1_IBM_s111
                                 + 1.35842 w_k1_INTC_s111
                                 + 1.47418 w_k1_KO_s111 + 1.11142 w_k1_MAT_s111
                                 + 1.35628 w_k1_MSFT_s111
                                 + 1.24777 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.840377 w_k1_ADM_s111
                                 + 0.696007 w_k1_AXP_s111
                                 + 0.770445 w_k1_DIS_s111
                                 + 0.856732 w_k1_FDX_s111
                                 + 0.877134 w_k1_IBM_s111
                                 + 0.928786 w_k1_INTC_s111
                                 + 0.79295 w_k1_KO_s111
                                 + 0.920392 w_k1_MAT_s111
                                 + 0.947304 w_k1_MSFT_s111
                                 + 0.890836 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.948668 w_k1_ADM_s111
                                 + 0.966297 w_k1_AXP_s111
                                 + 0.920333 w_k1_DIS_s111
                                 + 1.08299 w_k1_FDX_s111
                                 + 0.725761 w_k1_IBM_s111
                                 + 1.10379 w_k1_INTC_s111
                                 + 0.824879 w_k1_KO_s111
                                 + 0.730216 w_k1_MAT_s111
                                 + 1.13255 w_k1_MSFT_s111
                                 + 1.01385 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.887272 w_k1_ADM_s112
                                 + 0.735051 w_k1_AXP_s112
                                 + 0.744067 w_k1_DIS_s112
                                 + 0.816544 w_k1_FDX_s112
                                 + 0.90851 w_k1_IBM_s112
                                 + 0.791461 w_k1_INTC_s112
                                 + 0.777591 w_k1_KO_s112
                                 + 0.686634 w_k1_MAT_s112
                                 + 0.905523 w_k1_MSFT_s112
                                 + 0.891574 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.28299 w_k1_ADM_s112
                                 + 1.24874 w_k1_AXP_s112
                                 + 1.15064 w_k1_DIS_s112
                                 + 1.22318 w_k1_FDX_s112
                                 + 0.794134 w_k1_IBM_s112
                                 + 1.09939 w_k1_INTC_s112
                                 + 0.968223 w_k1_KO_s112
                                 + 0.965069 w_k1_MAT_s112
                                 + 1.20762 w_k1_MSFT_s112
                                 + 1.02134 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.11914 w_k1_ADM_s112
                                 + 1.68377 w_k1_AXP_s112
                                 + 1.18266 w_k1_DIS_s112
                                 + 1.19397 w_k1_FDX_s112
                                 + 1.26792 w_k1_IBM_s112
                                 + 1.17146 w_k1_INTC_s112
                                 + 1.30081 w_k1_KO_s112 + 1.44209 w_k1_MAT_s112
                                 + 1.13064 w_k1_MSFT_s112
                                 + 1.18996 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.07617 w_k1_ADM_s112
                                 + 1.0897 w_k1_AXP_s112 + 1.10065 w_k1_DIS_s112
                                 + 0.923889 w_k1_FDX_s112
                                 + 1.60369 w_k1_IBM_s112
                                 + 1.04827 w_k1_INTC_s112
                                 + 1.26253 w_k1_KO_s112 + 1.62539 w_k1_MAT_s112
                                 + 1.12369 w_k1_MSFT_s112
                                 + 1.13738 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.04223 w_k1_ADM_s112
                                 + 0.99165 w_k1_AXP_s112 + 1.1786 w_k1_DIS_s112
                                 + 0.982637 w_k1_FDX_s112
                                 + 1.13867 w_k1_IBM_s112
                                 + 1.1873 w_k1_INTC_s112 + 1.04766 w_k1_KO_s112
                                 + 1.00826 w_k1_MAT_s112
                                 + 1.24777 w_k1_MSFT_s112
                                 + 1.12028 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.924116 w_k1_ADM_s112
                                 + 0.910158 w_k1_AXP_s112
                                 + 0.809684 w_k1_DIS_s112
                                 + 0.843661 w_k1_FDX_s112
                                 + 1.26459 w_k1_IBM_s112
                                 + 0.725758 w_k1_INTC_s112
                                 + 0.916135 w_k1_KO_s112
                                 + 1.13624 w_k1_MAT_s112
                                 + 0.913823 w_k1_MSFT_s112
                                 + 0.873918 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.796143 w_k1_ADM_s112
                                 + 0.945855 w_k1_AXP_s112
                                 + 0.728096 w_k1_DIS_s112
                                 + 0.901221 w_k1_FDX_s112
                                 + 0.724376 w_k1_IBM_s112
                                 + 0.939123 w_k1_INTC_s112
                                 + 0.811199 w_k1_KO_s112
                                 + 0.698869 w_k1_MAT_s112
                                 + 1.01609 w_k1_MSFT_s112
                                 + 0.925103 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.06754 w_k1_ADM_s112
                                 + 1.52713 w_k1_AXP_s112
                                 + 1.62039 w_k1_DIS_s112 + 1.4098 w_k1_FDX_s112
                                 + 1.26588 w_k1_IBM_s112
                                 + 1.22384 w_k1_INTC_s112
                                 + 1.25183 w_k1_KO_s112 + 2.02649 w_k1_MAT_s112
                                 + 1.51619 w_k1_MSFT_s112
                                 + 1.26107 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.866669 w_k1_ADM_s113
                                 + 0.839681 w_k1_AXP_s113
                                 + 0.689006 w_k1_DIS_s113
                                 + 0.750352 w_k1_FDX_s113
                                 + 1.18446 w_k1_IBM_s113
                                 + 0.80704 w_k1_INTC_s113
                                 + 0.871525 w_k1_KO_s113
                                 + 0.947261 w_k1_MAT_s113
                                 + 0.915038 w_k1_MSFT_s113
                                 + 0.923149 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.24148 w_k1_ADM_s113
                                 + 1.36603 w_k1_AXP_s113
                                 + 1.55467 w_k1_DIS_s113
                                 + 1.32157 w_k1_FDX_s113
                                 + 1.51103 w_k1_IBM_s113
                                 + 1.19791 w_k1_INTC_s113
                                 + 1.45651 w_k1_KO_s113 + 1.71622 w_k1_MAT_s113
                                 + 1.3784 w_k1_MSFT_s113 + 1.29258 w_k1_PG_s113
                                 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.00297 w_k1_ADM_s113
                                 + 0.78953 w_k1_AXP_s113
                                 + 0.788143 w_k1_DIS_s113
                                 + 0.842418 w_k1_FDX_s113
                                 + 0.890682 w_k1_IBM_s113
                                 + 0.852308 w_k1_INTC_s113
                                 + 0.799394 w_k1_KO_s113
                                 + 1.12612 w_k1_MAT_s113
                                 + 0.894743 w_k1_MSFT_s113
                                 + 1.01579 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.793683 w_k1_ADM_s113
                                 + 0.576137 w_k1_AXP_s113
                                 + 0.611842 w_k1_DIS_s113
                                 + 0.714488 w_k1_FDX_s113
                                 + 0.731815 w_k1_IBM_s113
                                 + 0.647769 w_k1_INTC_s113
                                 + 0.554126 w_k1_KO_s113
                                 + 0.651996 w_k1_MAT_s113
                                 + 0.791137 w_k1_MSFT_s113
                                 + 0.816304 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.15517 w_k1_ADM_s113
                                 + 1.10775 w_k1_AXP_s113
                                 + 1.00888 w_k1_DIS_s113
                                 + 1.18566 w_k1_FDX_s113
                                 + 0.747406 w_k1_IBM_s113
                                 + 1.1155 w_k1_INTC_s113 + 0.86965 w_k1_KO_s113
                                 + 0.559972 w_k1_MAT_s113
                                 + 1.1114 w_k1_MSFT_s113 + 1.08214 w_k1_PG_s113
                                 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.02911 w_k1_ADM_s113
                                 + 0.866392 w_k1_AXP_s113
                                 + 0.752827 w_k1_DIS_s113
                                 + 0.969233 w_k1_FDX_s113
                                 + 0.691341 w_k1_IBM_s113
                                 + 0.964424 w_k1_INTC_s113
                                 + 0.772265 w_k1_KO_s113
                                 + 0.758415 w_k1_MAT_s113
                                 + 1.01638 w_k1_MSFT_s113
                                 + 1.02567 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.20917 w_k1_ADM_s113
                                 + 0.948366 w_k1_AXP_s113
                                 + 1.29694 w_k1_DIS_s113
                                 + 1.11557 w_k1_FDX_s113
                                 + 0.915988 w_k1_IBM_s113
                                 + 0.962286 w_k1_INTC_s113
                                 + 1.07023 w_k1_KO_s113 + 1.32524 w_k1_MAT_s113
                                 + 1.31406 w_k1_MSFT_s113
                                 + 1.09359 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.944319 w_k1_ADM_s113
                                 + 0.935632 w_k1_AXP_s113
                                 + 1.09385 w_k1_DIS_s113
                                 + 0.937236 w_k1_FDX_s113
                                 + 1.2751 w_k1_IBM_s113
                                 + 1.04357 w_k1_INTC_s113 + 1.1907 w_k1_KO_s113
                                 + 1.02561 w_k1_MAT_s113
                                 + 1.13562 w_k1_MSFT_s113
                                 + 1.00814 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.01457 w_k1_ADM_s114
                                 + 1.29453 w_k1_AXP_s114
                                 + 1.30914 w_k1_DIS_s114
                                 + 1.00366 w_k1_FDX_s114
                                 + 1.76444 w_k1_IBM_s114
                                 + 0.98742 w_k1_INTC_s114
                                 + 1.18094 w_k1_KO_s114 + 1.86177 w_k1_MAT_s114
                                 + 0.983449 w_k1_MSFT_s114
                                 + 1.18745 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.1628 w_k1_ADM_s114
                                 + 1.19188 w_k1_AXP_s114
                                 + 1.41786 w_k1_DIS_s114
                                 + 1.26005 w_k1_FDX_s114
                                 + 0.821082 w_k1_IBM_s114
                                 + 1.25028 w_k1_INTC_s114
                                 + 0.967612 w_k1_KO_s114
                                 + 0.486639 w_k1_MAT_s114
                                 + 1.30968 w_k1_MSFT_s114
                                 + 1.18666 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.20663 w_k1_ADM_s114
                                 + 1.08269 w_k1_AXP_s114
                                 + 1.08161 w_k1_DIS_s114
                                 + 1.10116 w_k1_FDX_s114
                                 + 1.06421 w_k1_IBM_s114
                                 + 1.1512 w_k1_INTC_s114
                                 + 0.955542 w_k1_KO_s114
                                 + 1.03382 w_k1_MAT_s114
                                 + 1.03822 w_k1_MSFT_s114
                                 + 1.19232 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.27219 w_k1_ADM_s114
                                 + 0.851904 w_k1_AXP_s114
                                 + 0.832954 w_k1_DIS_s114
                                 + 1.05734 w_k1_FDX_s114
                                 + 0.659064 w_k1_IBM_s114
                                 + 1.06621 w_k1_INTC_s114
                                 + 0.704243 w_k1_KO_s114
                                 + 0.491051 w_k1_MAT_s114
                                 + 0.910035 w_k1_MSFT_s114
                                 + 0.986861 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.33688 w_k1_ADM_s114
                                 + 1.45548 w_k1_AXP_s114
                                 + 1.64739 w_k1_DIS_s114
                                 + 1.31944 w_k1_FDX_s114
                                 + 1.40976 w_k1_IBM_s114
                                 + 1.39057 w_k1_INTC_s114
                                 + 1.26711 w_k1_KO_s114 + 1.29635 w_k1_MAT_s114
                                 + 1.07262 w_k1_MSFT_s114
                                 + 1.13905 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.782817 w_k1_ADM_s114
                                 + 0.714326 w_k1_AXP_s114
                                 + 0.58661 w_k1_DIS_s114
                                 + 0.714157 w_k1_FDX_s114
                                 + 0.715411 w_k1_IBM_s114
                                 + 0.85567 w_k1_INTC_s114
                                 + 0.65669 w_k1_KO_s114
                                 + 0.644248 w_k1_MAT_s114
                                 + 0.842879 w_k1_MSFT_s114
                                 + 0.880991 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.14816 w_k1_ADM_s114
                                 + 0.888142 w_k1_AXP_s114
                                 + 0.867168 w_k1_DIS_s114
                                 + 0.938393 w_k1_FDX_s114
                                 + 0.959083 w_k1_IBM_s114
                                 + 0.934559 w_k1_INTC_s114
                                 + 0.811644 w_k1_KO_s114
                                 + 0.712627 w_k1_MAT_s114
                                 + 0.875584 w_k1_MSFT_s114
                                 + 1.05914 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.899972 w_k1_ADM_s114
                                 + 0.98439 w_k1_AXP_s114
                                 + 0.753415 w_k1_DIS_s114
                                 + 0.792427 w_k1_FDX_s114
                                 + 1.29404 w_k1_IBM_s114
                                 + 1.13638 w_k1_INTC_s114
                                 + 0.91778 w_k1_KO_s114 + 1.14978 w_k1_MAT_s114
                                 + 0.728706 w_k1_MSFT_s114
                                 + 1.00984 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.861162 w_k1_ADM_s115
                                 + 0.84169 w_k1_AXP_s115
                                 + 0.791815 w_k1_DIS_s115
                                 + 0.947327 w_k1_FDX_s115
                                 + 0.68439 w_k1_IBM_s115
                                 + 0.838378 w_k1_INTC_s115
                                 + 0.804577 w_k1_KO_s115
                                 + 0.589281 w_k1_MAT_s115
                                 + 0.865299 w_k1_MSFT_s115
                                 + 0.955904 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.925888 w_k1_ADM_s115
                                 + 0.767653 w_k1_AXP_s115
                                 + 1.03572 w_k1_DIS_s115
                                 + 1.01335 w_k1_FDX_s115
                                 + 0.855814 w_k1_IBM_s115
                                 + 0.951375 w_k1_INTC_s115
                                 + 1.07038 w_k1_KO_s115
                                 + 0.701396 w_k1_MAT_s115
                                 + 1.06628 w_k1_MSFT_s115
                                 + 0.995653 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.64295 w_k1_ADM_s115
                                 + 1.47479 w_k1_AXP_s115 + 1.6088 w_k1_DIS_s115
                                 + 1.43799 w_k1_FDX_s115
                                 + 1.00026 w_k1_IBM_s115
                                 + 1.09257 w_k1_INTC_s115
                                 + 1.35316 w_k1_KO_s115 + 1.37613 w_k1_MAT_s115
                                 + 1.34599 w_k1_MSFT_s115
                                 + 1.22074 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.991445 w_k1_ADM_s115
                                 + 0.971508 w_k1_AXP_s115
                                 + 0.799455 w_k1_DIS_s115
                                 + 0.885187 w_k1_FDX_s115
                                 + 1.08204 w_k1_IBM_s115
                                 + 0.798636 w_k1_INTC_s115
                                 + 1.11082 w_k1_KO_s115 + 1.1061 w_k1_MAT_s115
                                 + 0.821847 w_k1_MSFT_s115
                                 + 0.949694 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.713073 w_k1_ADM_s115
                                 + 0.648676 w_k1_AXP_s115
                                 + 0.631821 w_k1_DIS_s115
                                 + 0.701767 w_k1_FDX_s115
                                 + 0.652887 w_k1_IBM_s115
                                 + 0.764835 w_k1_INTC_s115
                                 + 0.70331 w_k1_KO_s115
                                 + 0.493788 w_k1_MAT_s115
                                 + 0.74664 w_k1_MSFT_s115
                                 + 0.768473 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.18114 w_k1_ADM_s115
                                 + 1.67668 w_k1_AXP_s115
                                 + 1.76447 w_k1_DIS_s115
                                 + 1.24601 w_k1_FDX_s115
                                 + 1.78746 w_k1_IBM_s115
                                 + 1.30019 w_k1_INTC_s115
                                 + 1.84931 w_k1_KO_s115 + 1.72952 w_k1_MAT_s115
                                 + 1.29662 w_k1_MSFT_s115
                                 + 1.39195 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.27677 w_k1_ADM_s115
                                 + 1.30375 w_k1_AXP_s115
                                 + 1.19656 w_k1_DIS_s115
                                 + 1.18068 w_k1_FDX_s115 + 0.7496 w_k1_IBM_s115
                                 + 1.1692 w_k1_INTC_s115 + 1.16278 w_k1_KO_s115
                                 + 0.842299 w_k1_MAT_s115
                                 + 0.988253 w_k1_MSFT_s115
                                 + 1.11025 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.13823 w_k1_ADM_s115
                                 + 1.14781 w_k1_AXP_s115
                                 + 1.17913 w_k1_DIS_s115
                                 + 0.987622 w_k1_FDX_s115
                                 + 1.42869 w_k1_IBM_s115
                                 + 1.18908 w_k1_INTC_s115
                                 + 1.41807 w_k1_KO_s115 + 1.59139 w_k1_MAT_s115
                                 + 1.01611 w_k1_MSFT_s115 + 1.0027 w_k1_PG_s115
                                 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.919756 w_k1_ADM_s116
                                 + 0.934639 w_k1_AXP_s116
                                 + 1.04142 w_k1_DIS_s116
                                 + 1.06868 w_k1_FDX_s116
                                 + 0.933476 w_k1_IBM_s116
                                 + 1.11251 w_k1_INTC_s116
                                 + 0.989754 w_k1_KO_s116
                                 + 1.06118 w_k1_MAT_s116
                                 + 1.13246 w_k1_MSFT_s116
                                 + 0.971761 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.920533 w_k1_ADM_s116
                                 + 1.35331 w_k1_AXP_s116
                                 + 1.09722 w_k1_DIS_s116
                                 + 1.06988 w_k1_FDX_s116 + 1.6195 w_k1_IBM_s116
                                 + 1.22425 w_k1_INTC_s116
                                 + 1.33822 w_k1_KO_s116 + 1.73749 w_k1_MAT_s116
                                 + 0.981778 w_k1_MSFT_s116
                                 + 1.21021 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.30808 w_k1_ADM_s116
                                 + 1.42877 w_k1_AXP_s116
                                 + 1.45227 w_k1_DIS_s116
                                 + 1.37731 w_k1_FDX_s116
                                 + 1.05591 w_k1_IBM_s116
                                 + 1.37275 w_k1_INTC_s116
                                 + 1.24152 w_k1_KO_s116
                                 + 0.810453 w_k1_MAT_s116
                                 + 1.28727 w_k1_MSFT_s116
                                 + 1.16687 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.12092 w_k1_ADM_s116
                                 + 1.11178 w_k1_AXP_s116
                                 + 1.03142 w_k1_DIS_s116
                                 + 1.20996 w_k1_FDX_s116
                                 + 0.71439 w_k1_IBM_s116
                                 + 1.03147 w_k1_INTC_s116
                                 + 0.848612 w_k1_KO_s116
                                 + 0.89003 w_k1_MAT_s116
                                 + 1.11069 w_k1_MSFT_s116
                                 + 1.02968 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.812956 w_k1_ADM_s116
                                 + 0.881253 w_k1_AXP_s116
                                 + 0.734716 w_k1_DIS_s116
                                 + 0.793833 w_k1_FDX_s116
                                 + 1.17911 w_k1_IBM_s116
                                 + 0.809622 w_k1_INTC_s116
                                 + 0.93339 w_k1_KO_s116 + 1.32487 w_k1_MAT_s116
                                 + 0.80896 w_k1_MSFT_s116
                                 + 0.917557 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.15024 w_k1_ADM_s116
                                 + 1.21619 w_k1_AXP_s116
                                 + 0.785637 w_k1_DIS_s116
                                 + 0.922376 w_k1_FDX_s116
                                 + 1.09704 w_k1_IBM_s116
                                 + 0.947962 w_k1_INTC_s116
                                 + 0.914077 w_k1_KO_s116
                                 + 1.11851 w_k1_MAT_s116
                                 + 0.77475 w_k1_MSFT_s116
                                 + 0.939165 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.943629 w_k1_ADM_s116
                                 + 0.911372 w_k1_AXP_s116
                                 + 1.15638 w_k1_DIS_s116
                                 + 1.02419 w_k1_FDX_s116
                                 + 1.45529 w_k1_IBM_s116
                                 + 1.15548 w_k1_INTC_s116
                                 + 1.25958 w_k1_KO_s116 + 1.09621 w_k1_MAT_s116
                                 + 1.09182 w_k1_MSFT_s116
                                 + 0.985275 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.839804 w_k1_ADM_s116
                                 + 0.735168 w_k1_AXP_s116
                                 + 0.64897 w_k1_DIS_s116
                                 + 0.843247 w_k1_FDX_s116
                                 + 0.738839 w_k1_IBM_s116
                                 + 0.936624 w_k1_INTC_s116
                                 + 0.746654 w_k1_KO_s116
                                 + 0.523392 w_k1_MAT_s116
                                 + 0.854873 w_k1_MSFT_s116
                                 + 0.900509 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.04521 w_k1_ADM_s117
                                 + 0.913928 w_k1_AXP_s117
                                 + 1.07011 w_k1_DIS_s117
                                 + 1.04274 w_k1_FDX_s117
                                 + 0.783603 w_k1_IBM_s117
                                 + 1.13046 w_k1_INTC_s117
                                 + 1.04963 w_k1_KO_s117 + 0.62927 w_k1_MAT_s117
                                 + 1.13645 w_k1_MSFT_s117
                                 + 1.00988 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.898394 w_k1_ADM_s117
                                 + 0.882914 w_k1_AXP_s117
                                 + 0.80034 w_k1_DIS_s117
                                 + 0.892842 w_k1_FDX_s117
                                 + 0.773102 w_k1_IBM_s117
                                 + 0.943176 w_k1_INTC_s117
                                 + 0.798601 w_k1_KO_s117
                                 + 0.572933 w_k1_MAT_s117
                                 + 0.916762 w_k1_MSFT_s117
                                 + 0.893916 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.5371 w_k1_ADM_s117
                                 + 1.31109 w_k1_AXP_s117
                                 + 1.31485 w_k1_DIS_s117
                                 + 1.18405 w_k1_FDX_s117
                                 + 1.11075 w_k1_IBM_s117
                                 + 1.24852 w_k1_INTC_s117
                                 + 1.11998 w_k1_KO_s117 + 0.82139 w_k1_MAT_s117
                                 + 1.13736 w_k1_MSFT_s117
                                 + 1.12682 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.15471 w_k1_ADM_s117
                                 + 1.4409 w_k1_AXP_s117 + 1.08377 w_k1_DIS_s117
                                 + 1.06173 w_k1_FDX_s117
                                 + 1.02405 w_k1_IBM_s117
                                 + 0.920739 w_k1_INTC_s117
                                 + 0.895903 w_k1_KO_s117
                                 + 0.890393 w_k1_MAT_s117
                                 + 1.02155 w_k1_MSFT_s117
                                 + 0.927268 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.34109 w_k1_ADM_s117
                                 + 1.05733 w_k1_AXP_s117
                                 + 1.78484 w_k1_DIS_s117
                                 + 1.30465 w_k1_FDX_s117 + 1.1447 w_k1_IBM_s117
                                 + 1.13951 w_k1_INTC_s117
                                 + 1.23203 w_k1_KO_s117
                                 + 0.903248 w_k1_MAT_s117
                                 + 1.58741 w_k1_MSFT_s117
                                 + 1.24988 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.739687 w_k1_ADM_s117
                                 + 0.600021 w_k1_AXP_s117
                                 + 0.632663 w_k1_DIS_s117
                                 + 0.736211 w_k1_FDX_s117
                                 + 0.631497 w_k1_IBM_s117
                                 + 0.74528 w_k1_INTC_s117
                                 + 0.730174 w_k1_KO_s117
                                 + 0.541703 w_k1_MAT_s117
                                 + 0.888155 w_k1_MSFT_s117
                                 + 0.857836 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.04783 w_k1_ADM_s117
                                 + 1.06426 w_k1_AXP_s117
                                 + 1.07839 w_k1_DIS_s117
                                 + 0.907263 w_k1_FDX_s117
                                 + 1.4333 w_k1_IBM_s117
                                 + 0.916757 w_k1_INTC_s117
                                 + 1.18315 w_k1_KO_s117
                                 + 0.916316 w_k1_MAT_s117
                                 + 0.834872 w_k1_MSFT_s117
                                 + 1.01998 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.67758 w_k1_ADM_s117
                                 + 0.697902 w_k1_AXP_s117
                                 + 0.797368 w_k1_DIS_s117
                                 + 0.750952 w_k1_FDX_s117
                                 + 0.958084 w_k1_IBM_s117
                                 + 0.780531 w_k1_INTC_s117
                                 + 0.797834 w_k1_KO_s117
                                 + 0.601272 w_k1_MAT_s117
                                 + 0.827672 w_k1_MSFT_s117
                                 + 0.866216 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.025847197678907 w_k1_ADM_s000
                                 + 0.0208173046242908 w_k1_ADM_s001
                                 + 0.0301908637592081 w_k1_ADM_s002
                                 + 0.0455823453565533 w_k1_ADM_s003
                                 + 0.035057794172293 w_k1_ADM_s004
                                 + 0.0122950707249866 w_k1_ADM_s005
                                 + 0.0568036660441483 w_k1_ADM_s006
                                 + 0.0418275876189962 w_k1_ADM_s007
                                 + 0.0183059346855686 w_k1_ADM_s010
                                 + 0.0127371928900836 w_k1_ADM_s011
                                 + 0.00393940057491769 w_k1_ADM_s012
                                 + 0.00475582701880425 w_k1_ADM_s013
                                 + 0.00875496304701677 w_k1_ADM_s014
                                 + 0.00526460247429331 w_k1_ADM_s015
                                 + 0.0175998030938616 w_k1_ADM_s016
                                 + 0.00587880024734609 w_k1_ADM_s017
                                 + 0.0326968020320183 w_k1_ADM_s100
                                 + 0.0455445688580506 w_k1_ADM_s101
                                 + 0.0166772628451488 w_k1_ADM_s102
                                 + 0.0203247726544704 w_k1_ADM_s103
                                 + 0.0119181703220991 w_k1_ADM_s104
                                 + 0.00950683105139698 w_k1_ADM_s105
                                 + 0.0477298950585183 w_k1_ADM_s106
                                 + 0.0116909672788606 w_k1_ADM_s107
                                 + 0.0681188991858906 w_k1_ADM_s110
                                 + 0.0737382325592953 w_k1_ADM_s111
                                 + 0.0494797740307774 w_k1_ADM_s112
                                 + 0.056571480335403 w_k1_ADM_s113
                                 + 0.0377060573688432 w_k1_ADM_s114
                                 + 0.119117506041599 w_k1_ADM_s115
                                 + 0.0449891185532463 w_k1_ADM_s116
                                 + 0.0407135008642027 w_k1_ADM_s117
                                 + 0.0308239212191328 w_k1_AXP_s000
                                 + 0.0174733206920108 w_k1_AXP_s001
                                 + 0.0328823299139683 w_k1_AXP_s002
                                 + 0.0411096177725171 w_k1_AXP_s003
                                 + 0.0390390622121886 w_k1_AXP_s004
                                 + 0.013750274778239 w_k1_AXP_s005
                                 + 0.053096238225379 w_k1_AXP_s006
                                 + 0.0404974950741547 w_k1_AXP_s007
                                 + 0.0171420713998663 w_k1_AXP_s010
                                 + 0.0145533038284334 w_k1_AXP_s011
                                 + 0.0045219918465787 w_k1_AXP_s012
                                 + 0.00489147903679387 w_k1_AXP_s013
                                 + 0.0085714032463058 w_k1_AXP_s014
                                 + 0.00597885172473172 w_k1_AXP_s015
                                 + 0.0175588431193178 w_k1_AXP_s016
                                 + 0.00559719579592973 w_k1_AXP_s017
                                 + 0.038483248188372 w_k1_AXP_s100
                                 + 0.0448151865410012 w_k1_AXP_s101
                                 + 0.0164080972958393 w_k1_AXP_s102
                                 + 0.0183702023238874 w_k1_AXP_s103
                                 + 0.0133111892336576 w_k1_AXP_s104
                                 + 0.00970159610047598 w_k1_AXP_s105
                                 + 0.0463391741842722 w_k1_AXP_s106
                                 + 0.0105768586112143 w_k1_AXP_s107
                                 + 0.068773828404095 w_k1_AXP_s110
                                 + 0.0783392503532811 w_k1_AXP_s111
                                 + 0.0523538553956974 w_k1_AXP_s112
                                 + 0.0522059383196577 w_k1_AXP_s113
                                 + 0.0354322572205805 w_k1_AXP_s114
                                 + 0.117047430911156 w_k1_AXP_s115
                                 + 0.0465360094262976 w_k1_AXP_s116
                                 + 0.0384407083280588 w_k1_AXP_s117
                                 + 0.0266690262825287 w_k1_DIS_s000
                                 + 0.0171807273156226 w_k1_DIS_s001
                                 + 0.0316454615800172 w_k1_DIS_s002
                                 + 0.0407517316362052 w_k1_DIS_s003
                                 + 0.0375818416390873 w_k1_DIS_s004
                                 + 0.0135454539394165 w_k1_DIS_s005
                                 + 0.0515213315764074 w_k1_DIS_s006
                                 + 0.0425220683960691 w_k1_DIS_s007
                                 + 0.0174739012964279 w_k1_DIS_s010
                                 + 0.0137148378127619 w_k1_DIS_s011
                                 + 0.00437977173442746 w_k1_DIS_s012
                                 + 0.00472525061438444 w_k1_DIS_s013
                                 + 0.00911303461992081 w_k1_DIS_s014
                                 + 0.00603567601588252 w_k1_DIS_s015
                                 + 0.0183131717950564 w_k1_DIS_s016
                                 + 0.00600749325974537 w_k1_DIS_s017
                                 + 0.0421653485988767 w_k1_DIS_s100
                                 + 0.0421532646973315 w_k1_DIS_s101
                                 + 0.0156452454118828 w_k1_DIS_s102
                                 + 0.0189555151877787 w_k1_DIS_s103
                                 + 0.0131202660657938 w_k1_DIS_s104
                                 + 0.0107569089482166 w_k1_DIS_s105
                                 + 0.0473739020885494 w_k1_DIS_s106
                                 + 0.0100597840799246 w_k1_DIS_s107
                                 + 0.0639743802892478 w_k1_DIS_s110
                                 + 0.0776687343104314 w_k1_DIS_s111
                                 + 0.0507492624204246 w_k1_DIS_s112
                                 + 0.0558758584746811 w_k1_DIS_s113
                                 + 0.0348717511140241 w_k1_DIS_s114
                                 + 0.118340686577298 w_k1_DIS_s115
                                 + 0.0446752895669545 w_k1_DIS_s116
                                 + 0.0412160250062639 w_k1_DIS_s117
                                 + 0.0263312859146721 w_k1_FDX_s000
                                 + 0.0168216138077447 w_k1_FDX_s001
                                 + 0.0280025110568017 w_k1_FDX_s002
                                 + 0.0425042167255567 w_k1_FDX_s003
                                 + 0.0382717316986179 w_k1_FDX_s004
                                 + 0.0138311488727101 w_k1_FDX_s005
                                 + 0.0513991709525475 w_k1_FDX_s006
                                 + 0.0412076921765877 w_k1_FDX_s007
                                 + 0.0174791617471046 w_k1_FDX_s010
                                 + 0.0145585786983105 w_k1_FDX_s011
                                 + 0.00408600302915761 w_k1_FDX_s012
                                 + 0.00465713530640318 w_k1_FDX_s013
                                 + 0.0096104964141727 w_k1_FDX_s014
                                 + 0.00551374335638609 w_k1_FDX_s015
                                 + 0.017778726617014 w_k1_FDX_s016
                                 + 0.00540717958195864 w_k1_FDX_s017
                                 + 0.0391686363605808 w_k1_FDX_s100
                                 + 0.0401322751268235 w_k1_FDX_s101
                                 + 0.015223800118583 w_k1_FDX_s102
                                 + 0.0198844977183471 w_k1_FDX_s103
                                 + 0.0136754153173176 w_k1_FDX_s104
                                 + 0.0108799426350051 w_k1_FDX_s105
                                 + 0.0461971903433119 w_k1_FDX_s106
                                 + 0.010533132192071 w_k1_FDX_s107
                                 + 0.0665464320477817 w_k1_FDX_s110
                                 + 0.0800914600888913 w_k1_FDX_s111
                                 + 0.049394249375 w_k1_FDX_s112
                                 + 0.0543404422550031 w_k1_FDX_s113
                                 + 0.034586579284875 w_k1_FDX_s114
                                 + 0.113751395563451 w_k1_FDX_s115
                                 + 0.0471735111276953 w_k1_FDX_s116
                                 + 0.0376657107162337 w_k1_FDX_s117
                                 + 0.0263056603182777 w_k1_IBM_s000
                                 + 0.0171546437370709 w_k1_IBM_s001
                                 + 0.0302754681997355 w_k1_IBM_s002
                                 + 0.0422510522083678 w_k1_IBM_s003
                                 + 0.0396849844806234 w_k1_IBM_s004
                                 + 0.0138808327598291 w_k1_IBM_s005
                                 + 0.0506053491834832 w_k1_IBM_s006
                                 + 0.0408966973362148 w_k1_IBM_s007
                                 + 0.0182652093836308 w_k1_IBM_s010
                                 + 0.0149908794430654 w_k1_IBM_s011
                                 + 0.00425968862797845 w_k1_IBM_s012
                                 + 0.00498154873184136 w_k1_IBM_s013
                                 + 0.00936325957921701 w_k1_IBM_s014
                                 + 0.00557568341260113 w_k1_IBM_s015
                                 + 0.0174198485817503 w_k1_IBM_s016
                                 + 0.00559760536591916 w_k1_IBM_s017
                                 + 0.0411748790804683 w_k1_IBM_s100
                                 + 0.0408747100528557 w_k1_IBM_s101
                                 + 0.0161210367653854 w_k1_IBM_s102
                                 + 0.0196778829259011 w_k1_IBM_s103
                                 + 0.0128272093321116 w_k1_IBM_s104
                                 + 0.0109795295763544 w_k1_IBM_s105
                                 + 0.0491078244455835 w_k1_IBM_s106
                                 + 0.0103760984531601 w_k1_IBM_s107
                                 + 0.0667125354902103 w_k1_IBM_s110
                                 + 0.0818812595097594 w_k1_IBM_s111
                                 + 0.0523968902297804 w_k1_IBM_s112
                                 + 0.0571040105313354 w_k1_IBM_s113
                                 + 0.0363383506217365 w_k1_IBM_s114
                                 + 0.11117364847951 w_k1_IBM_s115
                                 + 0.0460910349041999 w_k1_IBM_s116
                                 + 0.0403269278243019 w_k1_IBM_s117
                                 + 0.0277955578705824 w_k1_INTC_s000
                                 + 0.0160083978290522 w_k1_INTC_s001
                                 + 0.0295883645920571 w_k1_INTC_s002
                                 + 0.0369860875005722 w_k1_INTC_s003
                                 + 0.0446152657913812 w_k1_INTC_s004
                                 + 0.0162284738558388 w_k1_INTC_s005
                                 + 0.0509656533601094 w_k1_INTC_s006
                                 + 0.0402213256847285 w_k1_INTC_s007
                                 + 0.0184629032639688 w_k1_INTC_s010
                                 + 0.0142420231434838 w_k1_INTC_s011
                                 + 0.00402862644359514 w_k1_INTC_s012
                                 + 0.0041927973953023 w_k1_INTC_s013
                                 + 0.0104707208370541 w_k1_INTC_s014
                                 + 0.00570920476038723 w_k1_INTC_s015
                                 + 0.0176428577774111 w_k1_INTC_s016
                                 + 0.00659557564811194 w_k1_INTC_s017
                                 + 0.0406538909474617 w_k1_INTC_s100
                                 + 0.0451868787251792 w_k1_INTC_s101
                                 + 0.0181068304416443 w_k1_INTC_s102
                                 + 0.0190848982611074 w_k1_INTC_s103
                                 + 0.0143113309315009 w_k1_INTC_s104
                                 + 0.00978768526874077 w_k1_INTC_s105
                                 + 0.0418947555813851 w_k1_INTC_s106
                                 + 0.0106474122442867 w_k1_INTC_s107
                                 + 0.0699074212153176 w_k1_INTC_s110
                                 + 0.0849441413580473 w_k1_INTC_s111
                                 + 0.0495366409612985 w_k1_INTC_s112
                                 + 0.0534160242622885 w_k1_INTC_s113
                                 + 0.0375507642335722 w_k1_INTC_s114
                                 + 0.109266580262442 w_k1_INTC_s115
                                 + 0.048498153225488 w_k1_INTC_s116
                                 + 0.0382272025666611 w_k1_INTC_s117
                                 + 0.0265422678854636 w_k1_KO_s000
                                 + 0.0180384439440898 w_k1_KO_s001
                                 + 0.0273241302454556 w_k1_KO_s002
                                 + 0.0394753852069304 w_k1_KO_s003
                                 + 0.0360984609620776 w_k1_KO_s004
                                 + 0.0141714280886402 w_k1_KO_s005
                                 + 0.0500676270505753 w_k1_KO_s006
                                 + 0.0446624506239354 w_k1_KO_s007
                                 + 0.0176292047953976 w_k1_KO_s010
                                 + 0.0161578614397092 w_k1_KO_s011
                                 + 0.00428412879343402 w_k1_KO_s012
                                 + 0.00490289033129179 w_k1_KO_s013
                                 + 0.00940339589151384 w_k1_KO_s014
                                 + 0.00618958104873804 w_k1_KO_s015
                                 + 0.0175504671154666 w_k1_KO_s016
                                 + 0.00633252091304191 w_k1_KO_s017
                                 + 0.0433517312479331 w_k1_KO_s100
                                 + 0.0365973183783515 w_k1_KO_s101
                                 + 0.01784945242951 w_k1_KO_s102
                                 + 0.01913775990973 w_k1_KO_s103
                                 + 0.0128600192187608 w_k1_KO_s104
                                 + 0.0107471764692013 w_k1_KO_s105
                                 + 0.0475634495443743 w_k1_KO_s106
                                 + 0.0121519683558921 w_k1_KO_s107
                                 + 0.0676955186334847 w_k1_KO_s110
                                 + 0.0776797507523436 w_k1_KO_s111
                                 + 0.0488335573189228 w_k1_KO_s112
                                 + 0.054880409442945 w_k1_KO_s113
                                 + 0.0313751608052788 w_k1_KO_s114
                                 + 0.127634250907976 w_k1_KO_s115
                                 + 0.0451797989732846 w_k1_KO_s116
                                 + 0.0393648626833125 w_k1_KO_s117
                                 + 0.0264715823347104 w_k1_MAT_s000
                                 + 0.0183798714868703 w_k1_MAT_s001
                                 + 0.0250447978332401 w_k1_MAT_s002
                                 + 0.0377252056168661 w_k1_MAT_s003
                                 + 0.03802702782314 w_k1_MAT_s004
                                 + 0.0142282800782533 w_k1_MAT_s005
                                 + 0.0546240602571834 w_k1_MAT_s006
                                 + 0.0429685547558005 w_k1_MAT_s007
                                 + 0.014652565454359 w_k1_MAT_s010
                                 + 0.0164319606073638 w_k1_MAT_s011
                                 + 0.00454283901692908 w_k1_MAT_s012
                                 + 0.00630759535667015 w_k1_MAT_s013
                                 + 0.00991408102031043 w_k1_MAT_s014
                                 + 0.00698698827722318 w_k1_MAT_s015
                                 + 0.0168141688182196 w_k1_MAT_s016
                                 + 0.00540587323502268 w_k1_MAT_s017
                                 + 0.0451485143492298 w_k1_MAT_s100
                                 + 0.0377244964655379 w_k1_MAT_s101
                                 + 0.0167678317265126 w_k1_MAT_s102
                                 + 0.0188481826859263 w_k1_MAT_s103
                                 + 0.0102793628741272 w_k1_MAT_s104
                                 + 0.0108408254903836 w_k1_MAT_s105
                                 + 0.0474149190552216 w_k1_MAT_s106
                                 + 0.0114519624805662 w_k1_MAT_s107
                                 + 0.0519520675280281 w_k1_MAT_s110
                                 + 0.0826778616974953 w_k1_MAT_s111
                                 + 0.0540511644518959 w_k1_MAT_s112
                                 + 0.058399788606493 w_k1_MAT_s113
                                 + 0.0317511306332976 w_k1_MAT_s114
                                 + 0.114453290695507 w_k1_MAT_s115
                                 + 0.0437242521158 w_k1_MAT_s116
                                 + 0.0289280023205065 w_k1_MAT_s117
                                 + 0.0262211418769465 w_k1_MSFT_s000
                                 + 0.0166902308472119 w_k1_MSFT_s001
                                 + 0.028417691311645 w_k1_MSFT_s002
                                 + 0.0398450387721641 w_k1_MSFT_s003
                                 + 0.0409768483896596 w_k1_MSFT_s004
                                 + 0.0156849215521265 w_k1_MSFT_s005
                                 + 0.0482962342515443 w_k1_MSFT_s006
                                 + 0.0388613774696578 w_k1_MSFT_s007
                                 + 0.0190545197840431 w_k1_MSFT_s010
                                 + 0.0137582239121301 w_k1_MSFT_s011
                                 + 0.00453138170182688 w_k1_MSFT_s012
                                 + 0.00485060381064695 w_k1_MSFT_s013
                                 + 0.00961897610233836 w_k1_MSFT_s014
                                 + 0.00559717527565912 w_k1_MSFT_s015
                                 + 0.018671749949266 w_k1_MSFT_s016
                                 + 0.00632092223871529 w_k1_MSFT_s017
                                 + 0.0414808500678953 w_k1_MSFT_s100
                                 + 0.0429414361342693 w_k1_MSFT_s101
                                 + 0.0156700517233825 w_k1_MSFT_s102
                                 + 0.0195761007180357 w_k1_MSFT_s103
                                 + 0.013932108199173 w_k1_MSFT_s104
                                 + 0.0102416156498006 w_k1_MSFT_s105
                                 + 0.0478976876020191 w_k1_MSFT_s106
                                 + 0.0108366204817349 w_k1_MSFT_s107
                                 + 0.0620315124014329 w_k1_MSFT_s110
                                 + 0.0848731658907461 w_k1_MSFT_s111
                                 + 0.0549595160508523 w_k1_MSFT_s112
                                 + 0.0593078288294739 w_k1_MSFT_s113
                                 + 0.0331152543121182 w_k1_MSFT_s114
                                 + 0.109514438569361 w_k1_MSFT_s115
                                 + 0.0462484508624475 w_k1_MSFT_s116
                                 + 0.0389710922362779 w_k1_MSFT_s117
                                 + 0.0269283522202853 w_k1_PG_s000
                                 + 0.0162820713031884 w_k1_PG_s001
                                 + 0.0303605693370269 w_k1_PG_s002
                                 + 0.0386618041031217 w_k1_PG_s003
                                 + 0.039328969291988 w_k1_PG_s004
                                 + 0.0142851093147124 w_k1_PG_s005
                                 + 0.0539614826523846 w_k1_PG_s006
                                 + 0.0408657043395397 w_k1_PG_s007
                                 + 0.0169313967362409 w_k1_PG_s010
                                 + 0.0160952959629718 w_k1_PG_s011
                                 + 0.00434861351408586 w_k1_PG_s012
                                 + 0.00473683046078068 w_k1_PG_s013
                                 + 0.00944836488243967 w_k1_PG_s014
                                 + 0.00544108894069213 w_k1_PG_s015
                                 + 0.0174739802879237 w_k1_PG_s016
                                 + 0.00529417560108094 w_k1_PG_s017
                                 + 0.0425507060240833 w_k1_PG_s100
                                 + 0.0395477955977646 w_k1_PG_s101
                                 + 0.0149834989993325 w_k1_PG_s102
                                 + 0.0170002075874836 w_k1_PG_s103
                                 + 0.0127874908990292 w_k1_PG_s104
                                 + 0.0105814534640595 w_k1_PG_s105
                                 + 0.0431401617586217 w_k1_PG_s106
                                 + 0.0107456455134112 w_k1_PG_s107
                                 + 0.0687938572270077 w_k1_PG_s110
                                 + 0.0806169617065817 w_k1_PG_s111
                                 + 0.0505016489166782 w_k1_PG_s112
                                 + 0.0572106903344901 w_k1_PG_s113
                                 + 0.0371499558953364 w_k1_PG_s114
                                 + 0.113188678175059 w_k1_PG_s115
                                 + 0.0453205228185146 w_k1_PG_s116
                                 + 0.0386259174550045 w_k1_PG_s117 >= 1.16
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
