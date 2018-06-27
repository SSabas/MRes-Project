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
 w_k1_ADM_s0_return_constraint:  1.14452 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.968042 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.982686 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.27885 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.77111 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.995676 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.14517 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.662677 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.07373 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.20762 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.733145 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.10277 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.819192 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.878548 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.778421 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.1519 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.14083 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.85351 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.16971 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.949096 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.848259 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.02109 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.993828 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.802668 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.2768 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.19042 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.42223 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.17807 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.875301 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.04733 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.807146 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.23363 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.3425 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.938775 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.02689 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.27408 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.860664 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.10275 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.18509 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.907943 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.972219 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.3586 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.727851 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.2143 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.37136 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.560464 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.29402 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.37288 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.843327 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.07933 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.751745 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.800694 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.615834 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.06805 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.4528 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.919417 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.20346 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.15286 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.656393 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.898413 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.965866 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.708177 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.997372 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.2751 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.54911 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.23719 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.722058 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.06988 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.71915 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.43411 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.07092 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.932736 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.822339 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.30345 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.616478 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.969403 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.28321 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.875018 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.913545 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.56605 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.719581 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.24849 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.58894 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.529944 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.17375 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.04853 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.831229 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.00956 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.682268 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.01266 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.575676 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.14788 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.47759 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.805733 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.09657 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.754309 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.754755 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.947425 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.989813 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.53943 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.14665 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.30755 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.77209 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.12979 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.719892 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.861488 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.678346 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.50783 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.15312 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.837428 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.918736 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.48632 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.599496 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.93384 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.19702 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.911892 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.916261 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.24647 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.837687 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.11047 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.3536 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.697266 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.2515 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.07279 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.763645 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.19038 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.857152 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.9905 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.726014 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.12895 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.43481 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.918221 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.971153 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.951727 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.994685 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.12131 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.924789 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.650258 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.2749 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.12914 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.41819 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.16604 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.820704 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.843057 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.687664 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.27103 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.27673 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.870842 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.962884 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.36575 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.747243 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.00181 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.08543 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.00868 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.972226 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.16464 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.760613 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.20031 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.33408 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.551181 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.922256 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.50322 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.26211 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.636731 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.693626 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.867734 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.701615 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.02337 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.15131 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.875488 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.6731 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.998863 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.603704 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.597308 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.947754 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.00304 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.597508 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.56788 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.11033 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.827242 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.56762 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.41554 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.11811 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.70873 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.733422 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.34185 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.685251 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.02138 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.722905 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.0316 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.15951 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.907811 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.982991 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.11719 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.867246 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.06767 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.16364 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.674876 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.07249 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.961737 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.715967 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.862502 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.825847 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.887011 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.904048 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.973735 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.25148 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.03865 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.18895 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.05319 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.03261 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.02685 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.844818 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.691392 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.24201 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.2744 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.33464 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.22919 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.903936 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.887641 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.854329 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.2842 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.11898 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.998489 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.00606 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.55898 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.845976 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.98361 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.16274 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.954945 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.85346 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.19961 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.759862 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.09681 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.60108 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.659578 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.12028 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.23314 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.10151 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.910571 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.85326 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.10447 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.750407 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.15957 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.24125 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.953612 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.47921 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.919914 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.839434 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.836809 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.04071 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.744406 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.902248 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.35048 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.3619 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.983854 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.65129 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.02534 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.034 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.58969 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.983842 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.13133 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.885722 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.32295 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.795256 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.08331 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.05945 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.02207 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.10865 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.28617 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.576016 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.29199 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.09654 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.580708 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.1058 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.37915 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.19506 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.597938 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.661235 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.842981 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.590048 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.26976 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.23576 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.654573 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.69574 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.957724 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.610771 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.705778 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.2172 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.10875 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.848778 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.65276 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.47038 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.11316 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.605414 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.39843 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.875078 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.6089 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.805499 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.35942 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.667132 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.15447 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.622694 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.887169 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.13278 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.957878 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.969708 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.28377 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.818459 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.03034 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.18496 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.725961 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.11744 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.951334 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.83707 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.992357 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.825437 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.954038 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.801085 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.15694 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.34008 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.915213 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.08503 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.844989 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.02651 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.10215 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.947975 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.721175 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.28906 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.2459 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.31903 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.15446 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.788407 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.83801 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.771108 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.26516 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.23595 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.952922 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.929517 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.36548 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.792831 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.989424 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.10592 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.917178 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.972176 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.19183 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.915354 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.10381 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.19233 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.700639 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.05305 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.08826 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.902626 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.939238 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.796929 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.00631 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.74026 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.10127 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.05272 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.893372 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.07537 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.860904 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.823072 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.01556 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.922144 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.771042 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.938528 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.14621 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.19199 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.978811 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.76485 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.925454 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.891615 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.33687 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.0523 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.997678 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.943892 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.23334 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.811709 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.036 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.694999987315407 z_k1_s0000
                                 - 0.307999758393082 z_k1_s0001
                                 - 0.853000147374115 z_k1_s0002
                                 - 0.312999718970902 z_k1_s0003
                                 - 0.279999744857357 z_k1_s0004
                                 - 0.516998422945864 z_k1_s0005
                                 - 0.196999774465725 z_k1_s0006
                                 - 0.742000553946647 z_k1_s0007
                                 - 0.620000196142577 z_k1_s0010
                                 - 0.181999858590377 z_k1_s0011
                                 - 0.676000317677965 z_k1_s0012
                                 - 0.371000805171941 z_k1_s0013
                                 - 0.343998447718958 z_k1_s0014
                                 - 0.1389999185256 z_k1_s0015
                                 - 0.152999948909446 z_k1_s0016
                                 - 0.196999737887461 z_k1_s0017
                                 - 0.158000371972628 z_k1_s0020
                                 - 0.213000176328463 z_k1_s0021
                                 - 0.626000679006883 z_k1_s0022
                                 - 0.704001047279707 z_k1_s0023
                                 - 0.180000166874782 z_k1_s0024
                                 - 0.366999374844442 z_k1_s0025
                                 - 0.555001888632522 z_k1_s0026
                                 - 0.368001412261069 z_k1_s0027
                                 - 0.791000064882695 z_k1_s0030
                                 - 0.931999136427466 z_k1_s0031
                                 - 0.491000360173438 z_k1_s0032
                                 - 0.817001487977228 z_k1_s0033
                                 - 0.351000190562225 z_k1_s0034
                                 - 1.30899959728338 z_k1_s0035
                                 - 0.370000077715075 z_k1_s0036
                                 - 0.703001012259278 z_k1_s0037
                                 - 0.439001229272869 z_k1_s0040
                                 - 0.50700086140066 z_k1_s0041
                                 - 0.363000710304788 z_k1_s0042
                                 - 1.09200246347947 z_k1_s0043
                                 - 0.267000346040385 z_k1_s0044
                                 - 0.251000548863473 z_k1_s0045
                                 - 0.783999944187007 z_k1_s0046
                                 - 0.250000438008437 z_k1_s0047
                                 - 0.239000831040163 z_k1_s0050
                                 - 0.0810000404747004 z_k1_s0051
                                 - 0.171999971309241 z_k1_s0052
                                 - 0.142999991425706 z_k1_s0053
                                 - 0.194000711220885 z_k1_s0054
                                 - 0.170999929713257 z_k1_s0055
                                 - 0.350999182590373 z_k1_s0056
                                 - 0.400999510389569 z_k1_s0057
                                 - 1.01699807196483 z_k1_s0060
                                 - 0.794002013208356 z_k1_s0061
                                 - 0.713000685813413 z_k1_s0062
                                 - 0.845998004130162 z_k1_s0063
                                 - 0.505001956131037 z_k1_s0064
                                 - 0.303999847150606 z_k1_s0065
                                 - 0.484999939100742 z_k1_s0066
                                 - 0.257999834659407 z_k1_s0067
                                 - 0.286000629995554 z_k1_s0070
                                 - 0.539001499915457 z_k1_s0071
                                 - 0.299000584740627 z_k1_s0072
                                 - 0.339001008203756 z_k1_s0073
                                 - 0.748001335064537 z_k1_s0074
                                 - 0.619001534039742 z_k1_s0075
                                 - 0.576000245694857 z_k1_s0076
                                 - 0.657999772671017 z_k1_s0077
                                 - 0.133999946593809 z_k1_s0100
                                 - 0.158000029249406 z_k1_s0101
                                 - 0.117999945690071 z_k1_s0102
                                 - 0.161999988825345 z_k1_s0103
                                 - 0.216999432994502 z_k1_s0104
                                 - 0.210999493630594 z_k1_s0105
                                 - 0.462000209020403 z_k1_s0106
                                 - 0.163999968613314 z_k1_s0107
                                 - 0.187000283664294 z_k1_s0110
                                 - 0.099000199351806 z_k1_s0111
                                 - 0.158000375579791 z_k1_s0112
                                 - 0.125000364862659 z_k1_s0113
                                 - 0.36300045228927 z_k1_s0114
                                 - 0.341000222210635 z_k1_s0115
                                 - 0.294000186680124 z_k1_s0116
                                 - 0.105000186100338 z_k1_s0117
                                 - 0.215000924256348 z_k1_s0120
                                 - 0.146000631308977 z_k1_s0121
                                 - 0.161001052211982 z_k1_s0122
                                 - 0.0960003869701755 z_k1_s0123
                                 - 0.177000195036725 z_k1_s0124
                                 - 0.117000407432149 z_k1_s0125
                                 - 0.220001643892958 z_k1_s0126
                                 - 0.448001279192084 z_k1_s0127
                                 - 0.15600022165133 z_k1_s0130
                                 - 0.121000259329714 z_k1_s0131
                                 - 0.144000252278237 z_k1_s0132
                                 - 0.391001532927847 z_k1_s0133
                                 - 0.269999827395533 z_k1_s0134
                                 - 0.31099967110351 z_k1_s0135
                                 - 0.330000707262858 z_k1_s0136
                                 - 0.343000002632502 z_k1_s0137
                                 - 0.315999816240859 z_k1_s0140
                                 - 0.202999596287049 z_k1_s0141
                                 - 0.329000729157737 z_k1_s0142
                                 - 0.172000412906846 z_k1_s0143
                                 - 0.13500026810235 z_k1_s0144
                                 - 0.227001435980066 z_k1_s0145
                                 - 0.432001018407853 z_k1_s0146
                                 - 0.192000474914648 z_k1_s0147
                                 - 0.0240000077960993 z_k1_s0150
                                 - 0.056000186124436 z_k1_s0151
                                 - 0.105000291733248 z_k1_s0152
                                 - 0.0280000472621623 z_k1_s0153
                                 - 0.0490001056088118 z_k1_s0154
                                 - 0.105000291733248 z_k1_s0155
                                 - 0.056000186124436 z_k1_s0156
                                 - 0.035000036177675 z_k1_s0157
                                 - 0.13999973516685 z_k1_s0160
                                 - 0.0539999840215184 z_k1_s0161
                                 - 0.0439999813953101 z_k1_s0162
                                 - 0.11900007649189 z_k1_s0163
                                 - 0.185999777407415 z_k1_s0164
                                 - 0.0539999840215184 z_k1_s0165
                                 - 0.10399999715256 z_k1_s0166
                                 - 0.0530000289989077 z_k1_s0167
                                 - 0.0230000541419987 z_k1_s0170
                                 - 0.0400000660686479 z_k1_s0171
                                 - 0.114000056590433 z_k1_s0172
                                 - 0.0820001627757724 z_k1_s0173
                                 - 0.0510002345802695 z_k1_s0174
                                 - 0.0800002315374567 z_k1_s0175
                                 - 0.078000300299141 z_k1_s0176
                                 - 0.0290000466572677 z_k1_s0177
                                 - 0.360999677446471 z_k1_s1000
                                 - 0.757996532968725 z_k1_s1001
                                 - 0.605998206884038 z_k1_s1002
                                 - 1.27699725084178 z_k1_s1003
                                 - 0.647996413303873 z_k1_s1004
                                 - 0.428999342694355 z_k1_s1005
                                 - 1.15399808776076 z_k1_s1006
                                 - 0.386999450276671 z_k1_s1007
                                 - 0.0829999109294975 z_k1_s1010
                                 - 0.117999825364898 z_k1_s1011
                                 - 0.156999360042029 z_k1_s1012
                                 - 0.163999086789484 z_k1_s1013
                                 - 0.115999801794345 z_k1_s1014
                                 - 0.326999229042129 z_k1_s1015
                                 - 0.33499932332434 z_k1_s1016
                                 - 0.122999813141384 z_k1_s1017
                                 - 0.148000256327761 z_k1_s1020
                                 - 0.0899998603511938 z_k1_s1021
                                 - 0.116000108554356 z_k1_s1022
                                 - 0.0899998603511938 z_k1_s1023
                                 - 0.121000309612382 z_k1_s1024
                                 - 0.106999860549705 z_k1_s1025
                                 - 0.293999293234348 z_k1_s1026
                                 - 0.173000122619929 z_k1_s1027
                                 - 0.1899992517992 z_k1_s1030
                                 - 0.355999142589803 z_k1_s1031
                                 - 0.29699914070055 z_k1_s1032
                                 - 0.515997212636921 z_k1_s1033
                                 - 0.18699915692389 z_k1_s1034
                                 - 0.201999165302479 z_k1_s1035
                                 - 0.211999015555551 z_k1_s1036
                                 - 0.369997721349405 z_k1_s1037
                                 - 0.222999580388718 z_k1_s1040
                                 - 0.172999768044025 z_k1_s1041
                                 - 0.148999701102803 z_k1_s1042
                                 - 0.195999685854577 z_k1_s1043
                                 - 0.307998827515333 z_k1_s1044
                                 - 0.380998764270275 z_k1_s1045
                                 - 0.216999925202881 z_k1_s1046
                                 - 0.418998646444206 z_k1_s1047
                                 - 0.82199843381149 z_k1_s1050
                                 - 0.215999294482562 z_k1_s1051
                                 - 0.17899970348439 z_k1_s1052
                                 - 0.793998457467618 z_k1_s1053
                                 - 0.51199724286499 z_k1_s1054
                                 - 0.177999384701452 z_k1_s1055
                                 - 0.572000097880552 z_k1_s1056
                                 - 0.794999183049573 z_k1_s1057
                                 - 0.197000314349918 z_k1_s1060
                                 - 0.322001068187026 z_k1_s1061
                                 - 0.415001458517569 z_k1_s1062
                                 - 0.140000183276529 z_k1_s1063
                                 - 0.203000277725985 z_k1_s1064
                                 - 0.300999280367796 z_k1_s1065
                                 - 0.156000245279625 z_k1_s1066
                                 - 0.661001872939328 z_k1_s1067
                                 - 0.644002096269175 z_k1_s1070
                                 - 0.672001413148981 z_k1_s1071
                                 - 0.154000313246621 z_k1_s1072
                                 - 0.317000806691078 z_k1_s1073
                                 - 0.150000374492295 z_k1_s1074
                                 - 0.228000325003827 z_k1_s1075
                                 - 0.180000296750465 z_k1_s1076
                                 - 0.199000451034358 z_k1_s1077
                                 - 0.526999877945399 z_k1_s1100
                                 - 0.364001416837152 z_k1_s1101
                                 - 0.237000426976277 z_k1_s1102
                                 - 0.393000544462975 z_k1_s1103
                                 - 0.397000878791281 z_k1_s1104
                                 - 0.113000083412158 z_k1_s1105
                                 - 0.274000486959062 z_k1_s1106
                                 - 0.332001379214221 z_k1_s1107
                                 - 0.247000621475774 z_k1_s1110
                                 - 0.429001459380253 z_k1_s1111
                                 - 0.460001615322819 z_k1_s1112
                                 - 0.560004059095686 z_k1_s1113
                                 - 0.853001232341603 z_k1_s1114
                                 - 1.09400339174667 z_k1_s1115
                                 - 0.35300104601773 z_k1_s1116
                                 - 0.817004684297937 z_k1_s1117
                                 - 0.906000372308616 z_k1_s1120
                                 - 0.245000192987743 z_k1_s1121
                                 - 0.685999289804417 z_k1_s1122
                                 - 0.599999754826401 z_k1_s1123
                                 - 0.334000293400465 z_k1_s1124
                                 - 0.20500020933265 z_k1_s1125
                                 - 0.428000684870369 z_k1_s1126
                                 - 0.503999637734019 z_k1_s1127
                                 - 0.299999773666797 z_k1_s1130
                                 - 0.47300057384305 z_k1_s1131
                                 - 0.225999734647749 z_k1_s1132
                                 - 0.565001559617144 z_k1_s1133
                                 - 0.692997325308458 z_k1_s1134
                                 - 0.794997355059062 z_k1_s1135
                                 - 0.893000434816388 z_k1_s1136
                                 - 0.501001453773715 z_k1_s1137
                                 - 0.537000311772314 z_k1_s1140
                                 - 0.430999982574632 z_k1_s1141
                                 - 1.04000068744736 z_k1_s1142
                                 - 1.02999938518264 z_k1_s1143
                                 - 1.74500266505446 z_k1_s1144
                                 - 1.53600090013757 z_k1_s1145
                                 - 0.782003021839875 z_k1_s1146
                                 - 0.384999980960632 z_k1_s1147
                                 - 0.478000206883582 z_k1_s1150
                                 - 0.273000947460871 z_k1_s1151
                                 - 0.575000780684117 z_k1_s1152
                                 - 0.505001963596904 z_k1_s1153
                                 - 0.276001010250747 z_k1_s1154
                                 - 0.889004439880198 z_k1_s1155
                                 - 0.709000672487681 z_k1_s1156
                                 - 0.267000821881121 z_k1_s1157
                                 - 0.594998884067323 z_k1_s1160
                                 - 0.486000307852067 z_k1_s1161
                                 - 1.15000169374495 z_k1_s1162
                                 - 0.255999969103077 z_k1_s1163
                                 - 0.616000641623287 z_k1_s1164
                                 - 1.11000016435329 z_k1_s1165
                                 - 0.302000364152993 z_k1_s1166
                                 - 0.940000483191201 z_k1_s1167
                                 - 0.517999361753808 z_k1_s1170
                                 - 0.423000397424825 z_k1_s1171
                                 - 0.296000186831441 z_k1_s1172
                                 - 1.16400370856522 z_k1_s1173
                                 - 0.27800014938355 z_k1_s1174
                                 - 0.810001685155083 z_k1_s1175
                                 - 0.365000491248558 z_k1_s1176
                                 - 0.972001056984885 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.54147 w_k1_ADM_s000
                                 + 1.62436 w_k1_AXP_s000
                                 + 1.62855 w_k1_DIS_s000
                                 + 1.28438 w_k1_FDX_s000
                                 + 1.41559 w_k1_IBM_s000
                                 + 1.26826 w_k1_INTC_s000
                                 + 1.29028 w_k1_KO_s000 + 2.09216 w_k1_MAT_s000
                                 + 1.32458 w_k1_MSFT_s000
                                 + 1.24994 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.06726 w_k1_ADM_s000
                                 + 1.01313 w_k1_AXP_s000 + 1.159 w_k1_DIS_s000
                                 + 1.09945 w_k1_FDX_s000
                                 + 0.677485 w_k1_IBM_s000
                                 + 1.10048 w_k1_INTC_s000
                                 + 0.803343 w_k1_KO_s000
                                 + 1.04568 w_k1_MAT_s000
                                 + 1.10365 w_k1_MSFT_s000
                                 + 0.950824 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.19294 w_k1_ADM_s000
                                 + 1.0606 w_k1_AXP_s000
                                 + 0.967863 w_k1_DIS_s000
                                 + 1.02359 w_k1_FDX_s000
                                 + 0.77883 w_k1_IBM_s000
                                 + 0.937062 w_k1_INTC_s000
                                 + 0.840934 w_k1_KO_s000
                                 + 1.13527 w_k1_MAT_s000
                                 + 1.08345 w_k1_MSFT_s000
                                 + 0.934654 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.828146 w_k1_ADM_s000
                                 + 0.520576 w_k1_AXP_s000
                                 + 0.612017 w_k1_DIS_s000
                                 + 0.679373 w_k1_FDX_s000
                                 + 0.747502 w_k1_IBM_s000
                                 + 0.681055 w_k1_INTC_s000
                                 + 0.687888 w_k1_KO_s000
                                 + 0.747979 w_k1_MAT_s000
                                 + 0.833884 w_k1_MSFT_s000
                                 + 0.786643 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.9804 w_k1_ADM_s000
                                 + 0.715347 w_k1_AXP_s000
                                 + 0.66403 w_k1_DIS_s000
                                 + 0.801315 w_k1_FDX_s000
                                 + 0.634762 w_k1_IBM_s000
                                 + 0.703183 w_k1_INTC_s000
                                 + 0.644631 w_k1_KO_s000
                                 + 0.858586 w_k1_MAT_s000
                                 + 0.870326 w_k1_MSFT_s000
                                 + 0.975631 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.910295 w_k1_ADM_s000
                                 + 0.729892 w_k1_AXP_s000
                                 + 0.815713 w_k1_DIS_s000
                                 + 0.766646 w_k1_FDX_s000
                                 + 1.08272 w_k1_IBM_s000
                                 + 1.00641 w_k1_INTC_s000
                                 + 0.934944 w_k1_KO_s000 + 1.0004 w_k1_MAT_s000
                                 + 1.02241 w_k1_MSFT_s000
                                 + 0.978403 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.15984 w_k1_ADM_s000
                                 + 1.39336 w_k1_AXP_s000 + 1.3686 w_k1_DIS_s000
                                 + 1.00729 w_k1_FDX_s000
                                 + 2.12061 w_k1_IBM_s000
                                 + 0.91034 w_k1_INTC_s000
                                 + 1.34848 w_k1_KO_s000 + 2.79519 w_k1_MAT_s000
                                 + 1.09164 w_k1_MSFT_s000
                                 + 1.24299 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.993129 w_k1_ADM_s000
                                 + 1.07492 w_k1_AXP_s000
                                 + 0.990809 w_k1_DIS_s000
                                 + 0.935485 w_k1_FDX_s000
                                 + 1.24966 w_k1_IBM_s000
                                 + 1.03734 w_k1_INTC_s000
                                 + 1.09319 w_k1_KO_s000 + 1.72714 w_k1_MAT_s000
                                 + 1.05892 w_k1_MSFT_s000
                                 + 1.00639 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.952557 w_k1_ADM_s001
                                 + 0.854169 w_k1_AXP_s001
                                 + 0.895761 w_k1_DIS_s001
                                 + 0.986042 w_k1_FDX_s001
                                 + 0.796358 w_k1_IBM_s001
                                 + 1.18123 w_k1_INTC_s001
                                 + 0.834745 w_k1_KO_s001
                                 + 0.674213 w_k1_MAT_s001
                                 + 0.985855 w_k1_MSFT_s001
                                 + 0.92492 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.08412 w_k1_ADM_s001
                                 + 1.32241 w_k1_AXP_s001
                                 + 1.64189 w_k1_DIS_s001
                                 + 1.31455 w_k1_FDX_s001
                                 + 1.15028 w_k1_IBM_s001
                                 + 1.59713 w_k1_INTC_s001
                                 + 1.41141 w_k1_KO_s001 + 1.12986 w_k1_MAT_s001
                                 + 1.50973 w_k1_MSFT_s001
                                 + 1.16017 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.0079 w_k1_ADM_s001
                                 + 1.0288 w_k1_AXP_s001 + 1.1683 w_k1_DIS_s001
                                 + 1.06574 w_k1_FDX_s001
                                 + 0.992863 w_k1_IBM_s001
                                 + 1.15144 w_k1_INTC_s001
                                 + 1.06234 w_k1_KO_s001
                                 + 0.704563 w_k1_MAT_s001
                                 + 1.08757 w_k1_MSFT_s001
                                 + 0.987892 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.784734 w_k1_ADM_s001
                                 + 0.652319 w_k1_AXP_s001
                                 + 0.702329 w_k1_DIS_s001
                                 + 0.757964 w_k1_FDX_s001
                                 + 1.1511 w_k1_IBM_s001
                                 + 0.775491 w_k1_INTC_s001
                                 + 0.893747 w_k1_KO_s001
                                 + 0.784907 w_k1_MAT_s001
                                 + 0.678191 w_k1_MSFT_s001
                                 + 0.959268 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.936577 w_k1_ADM_s001
                                 + 1.1041 w_k1_AXP_s001 + 1.45858 w_k1_DIS_s001
                                 + 1.01205 w_k1_FDX_s001
                                 + 1.70928 w_k1_IBM_s001
                                 + 1.21221 w_k1_INTC_s001
                                 + 1.36438 w_k1_KO_s001 + 1.21365 w_k1_MAT_s001
                                 + 1.20335 w_k1_MSFT_s001
                                 + 1.24158 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.988984 w_k1_ADM_s001
                                 + 1.08919 w_k1_AXP_s001
                                 + 1.49567 w_k1_DIS_s001
                                 + 1.14463 w_k1_FDX_s001
                                 + 0.988293 w_k1_IBM_s001
                                 + 1.14249 w_k1_INTC_s001
                                 + 1.09786 w_k1_KO_s001
                                 + 0.925071 w_k1_MAT_s001
                                 + 1.12526 w_k1_MSFT_s001
                                 + 1.03668 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.50474 w_k1_ADM_s001
                                 + 1.31112 w_k1_AXP_s001
                                 + 1.72364 w_k1_DIS_s001
                                 + 1.57474 w_k1_FDX_s001
                                 + 0.887171 w_k1_IBM_s001
                                 + 1.63222 w_k1_INTC_s001
                                 + 1.17806 w_k1_KO_s001
                                 + 0.575422 w_k1_MAT_s001
                                 + 1.37976 w_k1_MSFT_s001
                                 + 1.21812 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.723188 w_k1_ADM_s001
                                 + 0.716158 w_k1_AXP_s001
                                 + 0.696885 w_k1_DIS_s001
                                 + 0.783978 w_k1_FDX_s001
                                 + 0.637847 w_k1_IBM_s001
                                 + 0.813877 w_k1_INTC_s001
                                 + 0.711954 w_k1_KO_s001
                                 + 0.599512 w_k1_MAT_s001
                                 + 0.756826 w_k1_MSFT_s001
                                 + 0.823228 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.792185 w_k1_ADM_s002
                                 + 0.577516 w_k1_AXP_s002
                                 + 0.63469 w_k1_DIS_s002
                                 + 0.700801 w_k1_FDX_s002
                                 + 0.688914 w_k1_IBM_s002
                                 + 0.685939 w_k1_INTC_s002
                                 + 0.665592 w_k1_KO_s002
                                 + 0.612318 w_k1_MAT_s002
                                 + 0.771718 w_k1_MSFT_s002
                                 + 0.753981 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.01952 w_k1_ADM_s002
                                 + 0.737923 w_k1_AXP_s002
                                 + 0.700279 w_k1_DIS_s002
                                 + 0.813714 w_k1_FDX_s002
                                 + 0.868888 w_k1_IBM_s002
                                 + 0.989761 w_k1_INTC_s002
                                 + 0.870053 w_k1_KO_s002
                                 + 0.65952 w_k1_MAT_s002
                                 + 0.910345 w_k1_MSFT_s002
                                 + 0.958464 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.15288 w_k1_ADM_s002
                                 + 0.927226 w_k1_AXP_s002
                                 + 1.12714 w_k1_DIS_s002
                                 + 1.07016 w_k1_FDX_s002
                                 + 0.889891 w_k1_IBM_s002
                                 + 1.02641 w_k1_INTC_s002
                                 + 0.892007 w_k1_KO_s002
                                 + 0.810227 w_k1_MAT_s002
                                 + 1.15887 w_k1_MSFT_s002
                                 + 1.04188 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.20105 w_k1_ADM_s002
                                 + 1.14111 w_k1_AXP_s002
                                 + 1.09639 w_k1_DIS_s002
                                 + 1.07561 w_k1_FDX_s002
                                 + 1.07422 w_k1_IBM_s002
                                 + 1.17226 w_k1_INTC_s002
                                 + 1.14003 w_k1_KO_s002 + 1.09561 w_k1_MAT_s002
                                 + 1.11665 w_k1_MSFT_s002
                                 + 1.14746 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.18892 w_k1_ADM_s002
                                 + 1.03779 w_k1_AXP_s002
                                 + 1.39177 w_k1_DIS_s002
                                 + 1.35171 w_k1_FDX_s002
                                 + 0.801362 w_k1_IBM_s002
                                 + 1.35687 w_k1_INTC_s002
                                 + 0.959825 w_k1_KO_s002
                                 + 0.709843 w_k1_MAT_s002
                                 + 1.31707 w_k1_MSFT_s002
                                 + 1.16669 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.46376 w_k1_ADM_s002
                                 + 1.50799 w_k1_AXP_s002
                                 + 1.56885 w_k1_DIS_s002
                                 + 1.41036 w_k1_FDX_s002
                                 + 1.16701 w_k1_IBM_s002
                                 + 1.38437 w_k1_INTC_s002
                                 + 1.26684 w_k1_KO_s002 + 1.52475 w_k1_MAT_s002
                                 + 1.5323 w_k1_MSFT_s002 + 1.21416 w_k1_PG_s002
                                 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.02798 w_k1_ADM_s002
                                 + 0.939573 w_k1_AXP_s002
                                 + 0.856717 w_k1_DIS_s002
                                 + 0.874235 w_k1_FDX_s002
                                 + 1.21194 w_k1_IBM_s002
                                 + 1.06769 w_k1_INTC_s002
                                 + 0.973466 w_k1_KO_s002
                                 + 0.936416 w_k1_MAT_s002
                                 + 0.987609 w_k1_MSFT_s002
                                 + 1.02528 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.899681 w_k1_ADM_s002
                                 + 0.694499 w_k1_AXP_s002
                                 + 0.777005 w_k1_DIS_s002
                                 + 0.86327 w_k1_FDX_s002
                                 + 0.646549 w_k1_IBM_s002
                                 + 0.834384 w_k1_INTC_s002
                                 + 0.803283 w_k1_KO_s002
                                 + 0.545495 w_k1_MAT_s002
                                 + 0.919338 w_k1_MSFT_s002
                                 + 0.827312 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.736722 w_k1_ADM_s003
                                 + 0.756803 w_k1_AXP_s003
                                 + 1.20903 w_k1_DIS_s003
                                 + 0.979553 w_k1_FDX_s003
                                 + 0.887063 w_k1_IBM_s003
                                 + 0.884154 w_k1_INTC_s003
                                 + 1.21997 w_k1_KO_s003
                                 + 0.938872 w_k1_MAT_s003
                                 + 1.03295 w_k1_MSFT_s003
                                 + 0.987809 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.89452 w_k1_ADM_s003
                                 + 1.17911 w_k1_AXP_s003
                                 + 0.950077 w_k1_DIS_s003
                                 + 1.06748 w_k1_FDX_s003
                                 + 0.809778 w_k1_IBM_s003
                                 + 1.17844 w_k1_INTC_s003
                                 + 0.959129 w_k1_KO_s003
                                 + 0.973275 w_k1_MAT_s003
                                 + 1.07957 w_k1_MSFT_s003
                                 + 0.920302 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.874133 w_k1_ADM_s003
                                 + 1.00222 w_k1_AXP_s003
                                 + 1.33878 w_k1_DIS_s003
                                 + 1.14037 w_k1_FDX_s003
                                 + 0.958194 w_k1_IBM_s003
                                 + 1.22932 w_k1_INTC_s003
                                 + 1.13694 w_k1_KO_s003 + 1.07983 w_k1_MAT_s003
                                 + 1.13809 w_k1_MSFT_s003 + 1.0814 w_k1_PG_s003
                                 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.7787 w_k1_ADM_s003
                                 + 0.877427 w_k1_AXP_s003
                                 + 0.890969 w_k1_DIS_s003
                                 + 0.8429 w_k1_FDX_s003 + 1.26286 w_k1_IBM_s003
                                 + 0.921272 w_k1_INTC_s003
                                 + 1.12805 w_k1_KO_s003 + 1.30404 w_k1_MAT_s003
                                 + 0.9705 w_k1_MSFT_s003
                                 + 0.960877 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.05415 w_k1_ADM_s003
                                 + 1.50777 w_k1_AXP_s003
                                 + 1.60764 w_k1_DIS_s003
                                 + 1.15735 w_k1_FDX_s003
                                 + 2.09673 w_k1_IBM_s003
                                 + 1.43518 w_k1_INTC_s003
                                 + 1.80863 w_k1_KO_s003 + 2.28954 w_k1_MAT_s003
                                 + 1.2876 w_k1_MSFT_s003 + 1.33618 w_k1_PG_s003
                                 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.670803 w_k1_ADM_s003
                                 + 0.671074 w_k1_AXP_s003
                                 + 0.757259 w_k1_DIS_s003
                                 + 0.838145 w_k1_FDX_s003
                                 + 0.672548 w_k1_IBM_s003
                                 + 1.06817 w_k1_INTC_s003
                                 + 0.792167 w_k1_KO_s003
                                 + 0.624452 w_k1_MAT_s003
                                 + 0.87813 w_k1_MSFT_s003
                                 + 0.859852 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.13056 w_k1_ADM_s003
                                 + 1.61473 w_k1_AXP_s003
                                 + 1.68618 w_k1_DIS_s003
                                 + 1.52504 w_k1_FDX_s003
                                 + 1.03112 w_k1_IBM_s003
                                 + 1.30024 w_k1_INTC_s003
                                 + 1.12861 w_k1_KO_s003 + 1.26142 w_k1_MAT_s003
                                 + 1.25313 w_k1_MSFT_s003
                                 + 1.05967 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.05136 w_k1_ADM_s003
                                 + 1.27467 w_k1_AXP_s003
                                 + 1.41094 w_k1_DIS_s003 + 1.134 w_k1_FDX_s003
                                 + 1.38902 w_k1_IBM_s003
                                 + 1.08485 w_k1_INTC_s003
                                 + 1.25111 w_k1_KO_s003 + 1.89103 w_k1_MAT_s003
                                 + 1.29225 w_k1_MSFT_s003
                                 + 1.23048 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.26778 w_k1_ADM_s004
                                 + 0.850824 w_k1_AXP_s004
                                 + 1.02346 w_k1_DIS_s004
                                 + 0.987578 w_k1_FDX_s004 + 1.302 w_k1_IBM_s004
                                 + 1.08795 w_k1_INTC_s004
                                 + 1.14512 w_k1_KO_s004
                                 + 0.995363 w_k1_MAT_s004
                                 + 1.05525 w_k1_MSFT_s004
                                 + 0.967267 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.935916 w_k1_ADM_s004
                                 + 0.77766 w_k1_AXP_s004
                                 + 0.736672 w_k1_DIS_s004
                                 + 0.847716 w_k1_FDX_s004
                                 + 0.964738 w_k1_IBM_s004
                                 + 0.979347 w_k1_INTC_s004
                                 + 0.850353 w_k1_KO_s004
                                 + 0.763899 w_k1_MAT_s004
                                 + 0.871685 w_k1_MSFT_s004
                                 + 0.913967 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.04019 w_k1_ADM_s004
                                 + 1.41373 w_k1_AXP_s004
                                 + 1.44536 w_k1_DIS_s004
                                 + 1.28242 w_k1_FDX_s004
                                 + 1.71298 w_k1_IBM_s004
                                 + 1.47096 w_k1_INTC_s004 + 1.5457 w_k1_KO_s004
                                 + 1.99283 w_k1_MAT_s004
                                 + 1.27912 w_k1_MSFT_s004
                                 + 1.31395 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.15331 w_k1_ADM_s004
                                 + 1.03667 w_k1_AXP_s004
                                 + 1.07078 w_k1_DIS_s004 + 1.2273 w_k1_FDX_s004
                                 + 0.86205 w_k1_IBM_s004
                                 + 1.24726 w_k1_INTC_s004
                                 + 1.06139 w_k1_KO_s004
                                 + 0.914407 w_k1_MAT_s004
                                 + 1.13176 w_k1_MSFT_s004
                                 + 1.06585 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.28243 w_k1_ADM_s004
                                 + 1.26759 w_k1_AXP_s004
                                 + 1.52723 w_k1_DIS_s004
                                 + 1.37247 w_k1_FDX_s004
                                 + 1.23301 w_k1_IBM_s004
                                 + 1.5197 w_k1_INTC_s004 + 1.25549 w_k1_KO_s004
                                 + 1.07333 w_k1_MAT_s004
                                 + 1.21025 w_k1_MSFT_s004 + 1.3164 w_k1_PG_s004
                                 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.887786 w_k1_ADM_s004
                                 + 0.779903 w_k1_AXP_s004
                                 + 0.589128 w_k1_DIS_s004
                                 + 0.748361 w_k1_FDX_s004
                                 + 0.966116 w_k1_IBM_s004
                                 + 0.938089 w_k1_INTC_s004
                                 + 0.877113 w_k1_KO_s004
                                 + 0.953989 w_k1_MAT_s004
                                 + 0.717178 w_k1_MSFT_s004
                                 + 0.835396 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.809228 w_k1_ADM_s004
                                 + 0.641134 w_k1_AXP_s004
                                 + 0.66022 w_k1_DIS_s004
                                 + 0.876701 w_k1_FDX_s004
                                 + 0.631295 w_k1_IBM_s004
                                 + 0.963445 w_k1_INTC_s004
                                 + 0.702516 w_k1_KO_s004
                                 + 0.605581 w_k1_MAT_s004
                                 + 0.858757 w_k1_MSFT_s004
                                 + 0.773338 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.33592 w_k1_ADM_s004
                                 + 0.893314 w_k1_AXP_s004
                                 + 0.758176 w_k1_DIS_s004
                                 + 1.05027 w_k1_FDX_s004
                                 + 1.07835 w_k1_IBM_s004
                                 + 0.926875 w_k1_INTC_s004
                                 + 0.895619 w_k1_KO_s004
                                 + 1.03642 w_k1_MAT_s004
                                 + 0.886987 w_k1_MSFT_s004
                                 + 1.02409 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.983342 w_k1_ADM_s005
                                 + 1.30082 w_k1_AXP_s005 + 1.1071 w_k1_DIS_s005
                                 + 1.06025 w_k1_FDX_s005
                                 + 1.32755 w_k1_IBM_s005
                                 + 1.05587 w_k1_INTC_s005
                                 + 1.06528 w_k1_KO_s005 + 1.39217 w_k1_MAT_s005
                                 + 0.974812 w_k1_MSFT_s005
                                 + 0.922538 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.38487 w_k1_ADM_s005
                                 + 1.86195 w_k1_AXP_s005
                                 + 1.45181 w_k1_DIS_s005
                                 + 1.48018 w_k1_FDX_s005
                                 + 1.51567 w_k1_IBM_s005
                                 + 1.64743 w_k1_INTC_s005
                                 + 1.46871 w_k1_KO_s005 + 1.85496 w_k1_MAT_s005
                                 + 1.44066 w_k1_MSFT_s005 + 1.2481 w_k1_PG_s005
                                 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.18835 w_k1_ADM_s005
                                 + 1.15487 w_k1_AXP_s005
                                 + 0.995783 w_k1_DIS_s005
                                 + 1.14384 w_k1_FDX_s005
                                 + 1.07638 w_k1_IBM_s005
                                 + 1.24833 w_k1_INTC_s005
                                 + 1.10058 w_k1_KO_s005 + 1.37689 w_k1_MAT_s005
                                 + 1.19354 w_k1_MSFT_s005
                                 + 1.11967 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.17096 w_k1_ADM_s005
                                 + 0.969192 w_k1_AXP_s005
                                 + 0.970956 w_k1_DIS_s005
                                 + 1.09634 w_k1_FDX_s005
                                 + 0.69386 w_k1_IBM_s005
                                 + 1.13186 w_k1_INTC_s005
                                 + 1.00021 w_k1_KO_s005 + 1.07182 w_k1_MAT_s005
                                 + 1.20644 w_k1_MSFT_s005 + 1.2039 w_k1_PG_s005
                                 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.920677 w_k1_ADM_s005
                                 + 1.00249 w_k1_AXP_s005
                                 + 1.05395 w_k1_DIS_s005
                                 + 0.977682 w_k1_FDX_s005
                                 + 0.796981 w_k1_IBM_s005
                                 + 1.13907 w_k1_INTC_s005
                                 + 1.03731 w_k1_KO_s005 + 1.04176 w_k1_MAT_s005
                                 + 1.10302 w_k1_MSFT_s005
                                 + 0.980677 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.47923 w_k1_ADM_s005
                                 + 1.21142 w_k1_AXP_s005 + 1.3324 w_k1_DIS_s005
                                 + 1.31803 w_k1_FDX_s005
                                 + 0.87124 w_k1_IBM_s005
                                 + 1.50483 w_k1_INTC_s005
                                 + 0.966482 w_k1_KO_s005
                                 + 0.57327 w_k1_MAT_s005
                                 + 1.38042 w_k1_MSFT_s005
                                 + 0.975324 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.888168 w_k1_ADM_s005
                                 + 0.885479 w_k1_AXP_s005
                                 + 0.660617 w_k1_DIS_s005
                                 + 0.745212 w_k1_FDX_s005
                                 + 0.896792 w_k1_IBM_s005
                                 + 0.967206 w_k1_INTC_s005
                                 + 0.802762 w_k1_KO_s005
                                 + 1.13891 w_k1_MAT_s005
                                 + 0.780511 w_k1_MSFT_s005
                                 + 0.895725 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.915427 w_k1_ADM_s005
                                 + 0.895877 w_k1_AXP_s005
                                 + 0.787185 w_k1_DIS_s005
                                 + 0.885165 w_k1_FDX_s005
                                 + 1.00596 w_k1_IBM_s005
                                 + 1.03308 w_k1_INTC_s005
                                 + 0.927892 w_k1_KO_s005
                                 + 1.14992 w_k1_MAT_s005
                                 + 0.934313 w_k1_MSFT_s005
                                 + 0.963396 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.16445 w_k1_ADM_s006
                                 + 1.30643 w_k1_AXP_s006
                                 + 1.09317 w_k1_DIS_s006
                                 + 1.15925 w_k1_FDX_s006
                                 + 0.949227 w_k1_IBM_s006
                                 + 1.24318 w_k1_INTC_s006
                                 + 0.990481 w_k1_KO_s006
                                 + 0.963183 w_k1_MAT_s006
                                 + 0.968284 w_k1_MSFT_s006
                                 + 1.10488 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.29581 w_k1_ADM_s006
                                 + 1.38505 w_k1_AXP_s006
                                 + 1.58834 w_k1_DIS_s006
                                 + 1.30374 w_k1_FDX_s006
                                 + 1.19795 w_k1_IBM_s006
                                 + 1.22806 w_k1_INTC_s006
                                 + 1.20613 w_k1_KO_s006 + 1.07823 w_k1_MAT_s006
                                 + 1.23499 w_k1_MSFT_s006
                                 + 1.28927 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.794905 w_k1_ADM_s006
                                 + 0.946956 w_k1_AXP_s006
                                 + 0.805705 w_k1_DIS_s006
                                 + 0.77688 w_k1_FDX_s006
                                 + 1.24662 w_k1_IBM_s006
                                 + 0.778884 w_k1_INTC_s006
                                 + 1.03507 w_k1_KO_s006 + 1.48916 w_k1_MAT_s006
                                 + 0.883653 w_k1_MSFT_s006
                                 + 0.952826 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.998715 w_k1_ADM_s006
                                 + 0.939909 w_k1_AXP_s006
                                 + 1.07757 w_k1_DIS_s006
                                 + 0.972161 w_k1_FDX_s006
                                 + 1.07825 w_k1_IBM_s006
                                 + 0.993004 w_k1_INTC_s006
                                 + 1.01414 w_k1_KO_s006 + 1.013 w_k1_MAT_s006
                                 + 1.07328 w_k1_MSFT_s006
                                 + 1.09524 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.882214 w_k1_ADM_s006
                                 + 0.824172 w_k1_AXP_s006
                                 + 0.603862 w_k1_DIS_s006
                                 + 0.777369 w_k1_FDX_s006
                                 + 0.782873 w_k1_IBM_s006
                                 + 0.919031 w_k1_INTC_s006
                                 + 0.649429 w_k1_KO_s006
                                 + 0.599486 w_k1_MAT_s006
                                 + 0.683084 w_k1_MSFT_s006
                                 + 0.846677 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.3056 w_k1_ADM_s006
                                 + 1.04361 w_k1_AXP_s006
                                 + 1.24641 w_k1_DIS_s006
                                 + 1.35344 w_k1_FDX_s006
                                 + 0.686784 w_k1_IBM_s006
                                 + 1.2569 w_k1_INTC_s006
                                 + 0.927911 w_k1_KO_s006
                                 + 0.465632 w_k1_MAT_s006
                                 + 1.16783 w_k1_MSFT_s006
                                 + 1.17706 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.965248 w_k1_ADM_s006
                                 + 0.878765 w_k1_AXP_s006
                                 + 0.889422 w_k1_DIS_s006
                                 + 0.939484 w_k1_FDX_s006
                                 + 0.757388 w_k1_IBM_s006
                                 + 0.943083 w_k1_INTC_s006
                                 + 0.842551 w_k1_KO_s006
                                 + 0.690723 w_k1_MAT_s006
                                 + 1.01842 w_k1_MSFT_s006
                                 + 0.956436 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.04913 w_k1_ADM_s006
                                 + 0.969845 w_k1_AXP_s006
                                 + 0.761583 w_k1_DIS_s006
                                 + 1.02323 w_k1_FDX_s006
                                 + 0.598191 w_k1_IBM_s006
                                 + 0.82797 w_k1_INTC_s006
                                 + 0.695669 w_k1_KO_s006
                                 + 0.583278 w_k1_MAT_s006
                                 + 0.858037 w_k1_MSFT_s006
                                 + 0.892683 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.55274 w_k1_ADM_s007
                                 + 1.2874 w_k1_AXP_s007 + 1.68496 w_k1_DIS_s007
                                 + 1.41603 w_k1_FDX_s007
                                 + 1.00385 w_k1_IBM_s007
                                 + 1.48127 w_k1_INTC_s007
                                 + 1.30026 w_k1_KO_s007
                                 + 0.975864 w_k1_MAT_s007
                                 + 1.53002 w_k1_MSFT_s007
                                 + 1.17519 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.30652 w_k1_ADM_s007
                                 + 1.22708 w_k1_AXP_s007
                                 + 1.42325 w_k1_DIS_s007 + 1.2081 w_k1_FDX_s007
                                 + 1.17554 w_k1_IBM_s007
                                 + 1.18854 w_k1_INTC_s007
                                 + 1.13769 w_k1_KO_s007
                                 + 0.988169 w_k1_MAT_s007
                                 + 1.25982 w_k1_MSFT_s007
                                 + 1.00668 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.27331 w_k1_ADM_s007
                                 + 1.49453 w_k1_AXP_s007
                                 + 1.11371 w_k1_DIS_s007
                                 + 1.10909 w_k1_FDX_s007
                                 + 1.39637 w_k1_IBM_s007
                                 + 1.31462 w_k1_INTC_s007
                                 + 1.08962 w_k1_KO_s007 + 1.21726 w_k1_MAT_s007
                                 + 1.22871 w_k1_MSFT_s007
                                 + 0.982366 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.19179 w_k1_ADM_s007
                                 + 1.03853 w_k1_AXP_s007
                                 + 1.17376 w_k1_DIS_s007
                                 + 1.10348 w_k1_FDX_s007
                                 + 0.699261 w_k1_IBM_s007
                                 + 0.949757 w_k1_INTC_s007
                                 + 0.861108 w_k1_KO_s007
                                 + 0.572769 w_k1_MAT_s007
                                 + 1.1491 w_k1_MSFT_s007
                                 + 0.964026 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.928535 w_k1_ADM_s007
                                 + 1.15523 w_k1_AXP_s007
                                 + 1.04271 w_k1_DIS_s007
                                 + 0.953892 w_k1_FDX_s007
                                 + 1.2149 w_k1_IBM_s007
                                 + 1.25269 w_k1_INTC_s007 + 1.1316 w_k1_KO_s007
                                 + 1.01304 w_k1_MAT_s007
                                 + 1.13181 w_k1_MSFT_s007
                                 + 1.07067 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.894588 w_k1_ADM_s007
                                 + 0.835836 w_k1_AXP_s007
                                 + 0.671085 w_k1_DIS_s007
                                 + 0.866329 w_k1_FDX_s007
                                 + 0.677814 w_k1_IBM_s007
                                 + 0.787679 w_k1_INTC_s007
                                 + 0.649427 w_k1_KO_s007
                                 + 0.564967 w_k1_MAT_s007
                                 + 0.910611 w_k1_MSFT_s007
                                 + 0.823565 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.02113 w_k1_ADM_s007
                                 + 0.969608 w_k1_AXP_s007
                                 + 0.70092 w_k1_DIS_s007
                                 + 0.780973 w_k1_FDX_s007
                                 + 1.09752 w_k1_IBM_s007
                                 + 1.06506 w_k1_INTC_s007
                                 + 0.943974 w_k1_KO_s007
                                 + 0.905211 w_k1_MAT_s007
                                 + 0.894351 w_k1_MSFT_s007
                                 + 0.827517 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.0427 w_k1_ADM_s007
                                 + 1.25019 w_k1_AXP_s007
                                 + 0.916218 w_k1_DIS_s007
                                 + 0.974397 w_k1_FDX_s007
                                 + 0.891785 w_k1_IBM_s007
                                 + 0.956097 w_k1_INTC_s007
                                 + 0.869081 w_k1_KO_s007
                                 + 0.924124 w_k1_MAT_s007
                                 + 0.960092 w_k1_MSFT_s007
                                 + 0.866741 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.777821 w_k1_ADM_s010
                                 + 0.72652 w_k1_AXP_s010
                                 + 0.886873 w_k1_DIS_s010
                                 + 0.869163 w_k1_FDX_s010
                                 + 0.753401 w_k1_IBM_s010
                                 + 0.850129 w_k1_INTC_s010
                                 + 0.923973 w_k1_KO_s010
                                 + 0.780029 w_k1_MAT_s010
                                 + 0.932973 w_k1_MSFT_s010
                                 + 0.967902 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.950865 w_k1_ADM_s010
                                 + 1.36984 w_k1_AXP_s010
                                 + 1.69283 w_k1_DIS_s010
                                 + 1.27692 w_k1_FDX_s010
                                 + 1.64323 w_k1_IBM_s010
                                 + 1.12574 w_k1_INTC_s010
                                 + 1.48223 w_k1_KO_s010 + 2.23299 w_k1_MAT_s010
                                 + 1.26873 w_k1_MSFT_s010
                                 + 1.33047 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.876985 w_k1_ADM_s010
                                 + 0.805371 w_k1_AXP_s010
                                 + 0.778152 w_k1_DIS_s010
                                 + 0.976218 w_k1_FDX_s010
                                 + 0.607915 w_k1_IBM_s010
                                 + 0.872773 w_k1_INTC_s010
                                 + 0.660106 w_k1_KO_s010
                                 + 0.490643 w_k1_MAT_s010
                                 + 0.91833 w_k1_MSFT_s010
                                 + 0.91644 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.781464 w_k1_ADM_s010
                                 + 0.759964 w_k1_AXP_s010
                                 + 0.583276 w_k1_DIS_s010
                                 + 0.71921 w_k1_FDX_s010
                                 + 0.840698 w_k1_IBM_s010
                                 + 0.760392 w_k1_INTC_s010
                                 + 0.787086 w_k1_KO_s010
                                 + 0.700967 w_k1_MAT_s010
                                 + 0.770939 w_k1_MSFT_s010
                                 + 0.804642 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.15283 w_k1_ADM_s010
                                 + 1.33796 w_k1_AXP_s010 + 1.5936 w_k1_DIS_s010
                                 + 1.34462 w_k1_FDX_s010
                                 + 0.850674 w_k1_IBM_s010
                                 + 1.11908 w_k1_INTC_s010
                                 + 1.13778 w_k1_KO_s010
                                 + 0.818313 w_k1_MAT_s010
                                 + 1.24034 w_k1_MSFT_s010
                                 + 1.17116 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.863269 w_k1_ADM_s010
                                 + 0.906192 w_k1_AXP_s010
                                 + 1.02032 w_k1_DIS_s010
                                 + 1.00151 w_k1_FDX_s010
                                 + 0.905127 w_k1_IBM_s010
                                 + 0.918137 w_k1_INTC_s010
                                 + 0.958765 w_k1_KO_s010
                                 + 0.777016 w_k1_MAT_s010
                                 + 0.851174 w_k1_MSFT_s010
                                 + 1.08712 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.93009 w_k1_ADM_s010
                                 + 1.08312 w_k1_AXP_s010
                                 + 1.06687 w_k1_DIS_s010
                                 + 1.00507 w_k1_FDX_s010
                                 + 1.37327 w_k1_IBM_s010
                                 + 0.90362 w_k1_INTC_s010
                                 + 1.29486 w_k1_KO_s010 + 1.6997 w_k1_MAT_s010
                                 + 0.966142 w_k1_MSFT_s010
                                 + 1.1112 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.990547 w_k1_ADM_s010
                                 + 1.03017 w_k1_AXP_s010
                                 + 1.07882 w_k1_DIS_s010
                                 + 1.11856 w_k1_FDX_s010
                                 + 0.789648 w_k1_IBM_s010
                                 + 0.876485 w_k1_INTC_s010
                                 + 0.78226 w_k1_KO_s010
                                 + 0.768371 w_k1_MAT_s010
                                 + 1.00682 w_k1_MSFT_s010
                                 + 0.924729 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.888119 w_k1_ADM_s011
                                 + 1.11323 w_k1_AXP_s011
                                 + 1.08842 w_k1_DIS_s011
                                 + 1.19494 w_k1_FDX_s011
                                 + 0.825176 w_k1_IBM_s011
                                 + 1.46342 w_k1_INTC_s011
                                 + 0.939808 w_k1_KO_s011
                                 + 0.674044 w_k1_MAT_s011
                                 + 1.14801 w_k1_MSFT_s011
                                 + 1.07134 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.695808 w_k1_ADM_s011
                                 + 1.39918 w_k1_AXP_s011
                                 + 1.07636 w_k1_DIS_s011
                                 + 1.03247 w_k1_FDX_s011
                                 + 1.43708 w_k1_IBM_s011
                                 + 1.19426 w_k1_INTC_s011
                                 + 1.07896 w_k1_KO_s011 + 1.69428 w_k1_MAT_s011
                                 + 0.998803 w_k1_MSFT_s011
                                 + 1.00481 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.28682 w_k1_ADM_s011
                                 + 1.21357 w_k1_AXP_s011
                                 + 1.29253 w_k1_DIS_s011 + 1.1564 w_k1_FDX_s011
                                 + 1.57861 w_k1_IBM_s011
                                 + 1.3293 w_k1_INTC_s011 + 1.508 w_k1_KO_s011
                                 + 1.18707 w_k1_MAT_s011
                                 + 0.983777 w_k1_MSFT_s011
                                 + 1.12262 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.13992 w_k1_ADM_s011
                                 + 1.29402 w_k1_AXP_s011
                                 + 1.70256 w_k1_DIS_s011
                                 + 1.38506 w_k1_FDX_s011
                                 + 1.21849 w_k1_IBM_s011
                                 + 1.56969 w_k1_INTC_s011
                                 + 1.39488 w_k1_KO_s011 + 0.82908 w_k1_MAT_s011
                                 + 1.2824 w_k1_MSFT_s011 + 1.03093 w_k1_PG_s011
                                 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.721555 w_k1_ADM_s011
                                 + 0.826926 w_k1_AXP_s011
                                 + 0.880771 w_k1_DIS_s011
                                 + 0.849681 w_k1_FDX_s011
                                 + 1.04035 w_k1_IBM_s011
                                 + 1.0134 w_k1_INTC_s011
                                 + 0.892184 w_k1_KO_s011
                                 + 0.802979 w_k1_MAT_s011
                                 + 0.788731 w_k1_MSFT_s011
                                 + 0.807595 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.15153 w_k1_ADM_s011
                                 + 1.08926 w_k1_AXP_s011 + 1.0059 w_k1_DIS_s011
                                 + 1.02186 w_k1_FDX_s011
                                 + 1.08702 w_k1_IBM_s011
                                 + 1.13364 w_k1_INTC_s011
                                 + 0.995814 w_k1_KO_s011
                                 + 0.693134 w_k1_MAT_s011
                                 + 0.961408 w_k1_MSFT_s011
                                 + 1.03007 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.927566 w_k1_ADM_s011
                                 + 0.732898 w_k1_AXP_s011
                                 + 0.638536 w_k1_DIS_s011
                                 + 0.811079 w_k1_FDX_s011
                                 + 0.733975 w_k1_IBM_s011
                                 + 0.989373 w_k1_INTC_s011
                                 + 0.712392 w_k1_KO_s011
                                 + 0.645034 w_k1_MAT_s011
                                 + 0.79664 w_k1_MSFT_s011
                                 + 0.856365 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.05933 w_k1_ADM_s011
                                 + 1.21309 w_k1_AXP_s011
                                 + 0.989064 w_k1_DIS_s011
                                 + 1.07326 w_k1_FDX_s011
                                 + 1.33811 w_k1_IBM_s011
                                 + 0.975785 w_k1_INTC_s011
                                 + 1.05756 w_k1_KO_s011 + 1.40333 w_k1_MAT_s011
                                 + 0.838598 w_k1_MSFT_s011
                                 + 1.02246 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.948497 w_k1_ADM_s012
                                 + 0.831357 w_k1_AXP_s012
                                 + 0.839094 w_k1_DIS_s012
                                 + 0.80566 w_k1_FDX_s012
                                 + 1.11165 w_k1_IBM_s012
                                 + 1.06823 w_k1_INTC_s012
                                 + 1.01433 w_k1_KO_s012 + 1.26483 w_k1_MAT_s012
                                 + 0.827622 w_k1_MSFT_s012
                                 + 1.01314 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.05919 w_k1_ADM_s012
                                 + 0.998907 w_k1_AXP_s012
                                 + 1.01984 w_k1_DIS_s012
                                 + 1.01409 w_k1_FDX_s012
                                 + 1.03455 w_k1_IBM_s012
                                 + 0.972242 w_k1_INTC_s012
                                 + 1.01394 w_k1_KO_s012 + 1.4347 w_k1_MAT_s012
                                 + 0.915734 w_k1_MSFT_s012 + 1.14 w_k1_PG_s012
                                 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.07913 w_k1_ADM_s012
                                 + 1.44176 w_k1_AXP_s012
                                 + 1.07074 w_k1_DIS_s012
                                 + 1.19304 w_k1_FDX_s012
                                 + 0.966823 w_k1_IBM_s012
                                 + 1.4674 w_k1_INTC_s012
                                 + 0.939953 w_k1_KO_s012
                                 + 1.36261 w_k1_MAT_s012
                                 + 1.16627 w_k1_MSFT_s012
                                 + 1.14257 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.16906 w_k1_ADM_s012
                                 + 1.10662 w_k1_AXP_s012
                                 + 1.39061 w_k1_DIS_s012
                                 + 0.956801 w_k1_FDX_s012
                                 + 1.30813 w_k1_IBM_s012
                                 + 1.13021 w_k1_INTC_s012
                                 + 1.46566 w_k1_KO_s012 + 1.28019 w_k1_MAT_s012
                                 + 1.07231 w_k1_MSFT_s012
                                 + 1.41796 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.80029 w_k1_ADM_s012
                                 + 0.972458 w_k1_AXP_s012
                                 + 0.694798 w_k1_DIS_s012
                                 + 0.823367 w_k1_FDX_s012
                                 + 1.02213 w_k1_IBM_s012
                                 + 0.85786 w_k1_INTC_s012
                                 + 1.00568 w_k1_KO_s012 + 1.538 w_k1_MAT_s012
                                 + 0.774885 w_k1_MSFT_s012
                                 + 1.06278 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.44206 w_k1_ADM_s012
                                 + 1.81389 w_k1_AXP_s012
                                 + 1.73544 w_k1_DIS_s012
                                 + 1.37154 w_k1_FDX_s012
                                 + 1.71407 w_k1_IBM_s012
                                 + 1.23417 w_k1_INTC_s012
                                 + 1.70371 w_k1_KO_s012 + 2.77149 w_k1_MAT_s012
                                 + 1.24522 w_k1_MSFT_s012
                                 + 1.60647 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.40907 w_k1_ADM_s012
                                 + 1.28632 w_k1_AXP_s012
                                 + 1.56321 w_k1_DIS_s012
                                 + 1.25328 w_k1_FDX_s012
                                 + 0.931348 w_k1_IBM_s012
                                 + 1.25661 w_k1_INTC_s012 + 1.3284 w_k1_KO_s012
                                 + 0.901263 w_k1_MAT_s012
                                 + 1.26524 w_k1_MSFT_s012
                                 + 1.30958 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.965482 w_k1_ADM_s012
                                 + 0.734315 w_k1_AXP_s012
                                 + 0.830354 w_k1_DIS_s012
                                 + 0.944101 w_k1_FDX_s012
                                 + 0.627218 w_k1_IBM_s012
                                 + 0.888553 w_k1_INTC_s012
                                 + 0.813052 w_k1_KO_s012
                                 + 0.584487 w_k1_MAT_s012
                                 + 0.859343 w_k1_MSFT_s012
                                 + 0.877855 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.13885 w_k1_ADM_s013
                                 + 1.52035 w_k1_AXP_s013
                                 + 1.52106 w_k1_DIS_s013
                                 + 1.33123 w_k1_FDX_s013
                                 + 1.49655 w_k1_IBM_s013
                                 + 1.18201 w_k1_INTC_s013
                                 + 1.29096 w_k1_KO_s013 + 2.08692 w_k1_MAT_s013
                                 + 1.19501 w_k1_MSFT_s013
                                 + 1.19794 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.987965 w_k1_ADM_s013
                                 + 1.08082 w_k1_AXP_s013
                                 + 1.28422 w_k1_DIS_s013
                                 + 1.03438 w_k1_FDX_s013
                                 + 1.65089 w_k1_IBM_s013
                                 + 0.943131 w_k1_INTC_s013
                                 + 1.3826 w_k1_KO_s013 + 2.76514 w_k1_MAT_s013
                                 + 0.996781 w_k1_MSFT_s013
                                 + 1.22744 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.784752 w_k1_ADM_s013
                                 + 0.825563 w_k1_AXP_s013
                                 + 0.846264 w_k1_DIS_s013
                                 + 0.851616 w_k1_FDX_s013
                                 + 1.10208 w_k1_IBM_s013
                                 + 1.00807 w_k1_INTC_s013
                                 + 0.911057 w_k1_KO_s013
                                 + 1.77015 w_k1_MAT_s013
                                 + 0.928915 w_k1_MSFT_s013
                                 + 0.874711 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.72616 w_k1_ADM_s013
                                 + 0.720262 w_k1_AXP_s013
                                 + 0.674311 w_k1_DIS_s013
                                 + 0.809792 w_k1_FDX_s013
                                 + 0.733117 w_k1_IBM_s013
                                 + 0.67701 w_k1_INTC_s013
                                 + 0.708901 w_k1_KO_s013
                                 + 1.04277 w_k1_MAT_s013
                                 + 0.792427 w_k1_MSFT_s013
                                 + 0.766305 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.22298 w_k1_ADM_s013
                                 + 1.03363 w_k1_AXP_s013
                                 + 1.16825 w_k1_DIS_s013
                                 + 1.00595 w_k1_FDX_s013
                                 + 1.30053 w_k1_IBM_s013
                                 + 0.983429 w_k1_INTC_s013
                                 + 1.20673 w_k1_KO_s013 + 1.77687 w_k1_MAT_s013
                                 + 1.0342 w_k1_MSFT_s013 + 1.12914 w_k1_PG_s013
                                 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.13135 w_k1_ADM_s013
                                 + 1.01241 w_k1_AXP_s013
                                 + 0.891535 w_k1_DIS_s013
                                 + 1.02811 w_k1_FDX_s013
                                 + 0.852077 w_k1_IBM_s013
                                 + 0.876378 w_k1_INTC_s013
                                 + 0.813981 w_k1_KO_s013
                                 + 0.896551 w_k1_MAT_s013
                                 + 0.871939 w_k1_MSFT_s013
                                 + 0.975953 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.29121 w_k1_ADM_s013
                                 + 1.04633 w_k1_AXP_s013
                                 + 1.19484 w_k1_DIS_s013
                                 + 1.20943 w_k1_FDX_s013
                                 + 0.802452 w_k1_IBM_s013
                                 + 1.10143 w_k1_INTC_s013
                                 + 0.899587 w_k1_KO_s013
                                 + 1.03802 w_k1_MAT_s013
                                 + 1.15866 w_k1_MSFT_s013
                                 + 1.16056 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.845798 w_k1_ADM_s013
                                 + 0.914973 w_k1_AXP_s013
                                 + 1.16937 w_k1_DIS_s013
                                 + 1.03357 w_k1_FDX_s013
                                 + 0.915257 w_k1_IBM_s013
                                 + 0.995311 w_k1_INTC_s013
                                 + 0.964101 w_k1_KO_s013
                                 + 1.64079 w_k1_MAT_s013
                                 + 1.13051 w_k1_MSFT_s013
                                 + 1.05474 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.824826 w_k1_ADM_s014
                                 + 0.847879 w_k1_AXP_s014
                                 + 1.06984 w_k1_DIS_s014
                                 + 0.951976 w_k1_FDX_s014
                                 + 0.939224 w_k1_IBM_s014
                                 + 0.804053 w_k1_INTC_s014
                                 + 0.866628 w_k1_KO_s014
                                 + 1.04353 w_k1_MAT_s014
                                 + 0.981524 w_k1_MSFT_s014
                                 + 0.976259 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.02107 w_k1_ADM_s014
                                 + 1.4249 w_k1_AXP_s014 + 1.8874 w_k1_DIS_s014
                                 + 1.5025 w_k1_FDX_s014 + 1.63152 w_k1_IBM_s014
                                 + 1.62797 w_k1_INTC_s014
                                 + 1.61823 w_k1_KO_s014 + 1.98804 w_k1_MAT_s014
                                 + 1.55396 w_k1_MSFT_s014
                                 + 1.42801 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.870334 w_k1_ADM_s014
                                 + 0.748989 w_k1_AXP_s014
                                 + 0.733381 w_k1_DIS_s014
                                 + 0.831662 w_k1_FDX_s014
                                 + 0.982747 w_k1_IBM_s014
                                 + 0.908541 w_k1_INTC_s014
                                 + 0.856177 w_k1_KO_s014
                                 + 0.816401 w_k1_MAT_s014
                                 + 0.830292 w_k1_MSFT_s014
                                 + 0.864614 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.03899 w_k1_ADM_s014
                                 + 0.889615 w_k1_AXP_s014
                                 + 1.09254 w_k1_DIS_s014
                                 + 1.23123 w_k1_FDX_s014
                                 + 0.72188 w_k1_IBM_s014
                                 + 1.11289 w_k1_INTC_s014
                                 + 0.860866 w_k1_KO_s014
                                 + 0.920648 w_k1_MAT_s014
                                 + 1.05454 w_k1_MSFT_s014
                                 + 1.10356 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.915847 w_k1_ADM_s014
                                 + 1.39111 w_k1_AXP_s014
                                 + 1.20003 w_k1_DIS_s014
                                 + 1.15846 w_k1_FDX_s014
                                 + 1.21635 w_k1_IBM_s014
                                 + 1.04804 w_k1_INTC_s014 + 1.1082 w_k1_KO_s014
                                 + 1.56706 w_k1_MAT_s014
                                 + 1.11964 w_k1_MSFT_s014
                                 + 1.03952 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.92207 w_k1_ADM_s014
                                 + 0.934733 w_k1_AXP_s014
                                 + 1.08511 w_k1_DIS_s014
                                 + 0.927692 w_k1_FDX_s014
                                 + 1.50586 w_k1_IBM_s014
                                 + 0.965521 w_k1_INTC_s014
                                 + 1.16781 w_k1_KO_s014 + 1.47539 w_k1_MAT_s014
                                 + 0.918471 w_k1_MSFT_s014
                                 + 1.21321 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.64644 w_k1_ADM_s014
                                 + 0.646715 w_k1_AXP_s014
                                 + 0.726386 w_k1_DIS_s014
                                 + 0.836298 w_k1_FDX_s014
                                 + 0.683727 w_k1_IBM_s014
                                 + 0.758346 w_k1_INTC_s014
                                 + 0.725227 w_k1_KO_s014
                                 + 0.594309 w_k1_MAT_s014
                                 + 0.75434 w_k1_MSFT_s014
                                 + 0.839933 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.834209 w_k1_ADM_s014
                                 + 1.00099 w_k1_AXP_s014
                                 + 1.36147 w_k1_DIS_s014
                                 + 1.25179 w_k1_FDX_s014
                                 + 1.02215 w_k1_IBM_s014
                                 + 1.05014 w_k1_INTC_s014
                                 + 1.10296 w_k1_KO_s014 + 1.09036 w_k1_MAT_s014
                                 + 1.20489 w_k1_MSFT_s014
                                 + 1.18976 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.713681 w_k1_ADM_s015
                                 + 0.654028 w_k1_AXP_s015
                                 + 0.592541 w_k1_DIS_s015
                                 + 0.721797 w_k1_FDX_s015
                                 + 0.992345 w_k1_IBM_s015
                                 + 0.806007 w_k1_INTC_s015
                                 + 0.915905 w_k1_KO_s015
                                 + 1.15245 w_k1_MAT_s015
                                 + 0.755398 w_k1_MSFT_s015
                                 + 0.922396 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.51948 w_k1_ADM_s015
                                 + 1.21997 w_k1_AXP_s015
                                 + 1.37475 w_k1_DIS_s015 + 1.3295 w_k1_FDX_s015
                                 + 1.11353 w_k1_IBM_s015
                                 + 1.07158 w_k1_INTC_s015
                                 + 1.16686 w_k1_KO_s015 + 1.25289 w_k1_MAT_s015
                                 + 1.26487 w_k1_MSFT_s015 + 1.178 w_k1_PG_s015
                                 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.147 w_k1_ADM_s015
                                 + 0.978617 w_k1_AXP_s015
                                 + 1.00119 w_k1_DIS_s015
                                 + 0.955325 w_k1_FDX_s015
                                 + 1.0446 w_k1_IBM_s015
                                 + 0.934629 w_k1_INTC_s015
                                 + 1.0361 w_k1_KO_s015 + 1.09391 w_k1_MAT_s015
                                 + 0.979029 w_k1_MSFT_s015
                                 + 1.02717 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.675346 w_k1_ADM_s015
                                 + 0.721987 w_k1_AXP_s015
                                 + 0.609472 w_k1_DIS_s015
                                 + 0.773663 w_k1_FDX_s015
                                 + 0.828888 w_k1_IBM_s015
                                 + 0.699441 w_k1_INTC_s015
                                 + 0.736574 w_k1_KO_s015
                                 + 1.09973 w_k1_MAT_s015
                                 + 0.70397 w_k1_MSFT_s015
                                 + 0.985197 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.701833 w_k1_ADM_s015
                                 + 0.742833 w_k1_AXP_s015
                                 + 0.853243 w_k1_DIS_s015
                                 + 0.857376 w_k1_FDX_s015
                                 + 1.14975 w_k1_IBM_s015
                                 + 0.983118 w_k1_INTC_s015
                                 + 1.06752 w_k1_KO_s015 + 1.21762 w_k1_MAT_s015
                                 + 1.12928 w_k1_MSFT_s015
                                 + 1.15903 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.919738 w_k1_ADM_s015
                                 + 0.755619 w_k1_AXP_s015
                                 + 0.703428 w_k1_DIS_s015
                                 + 0.905182 w_k1_FDX_s015
                                 + 0.655168 w_k1_IBM_s015
                                 + 1.0007 w_k1_INTC_s015
                                 + 0.678958 w_k1_KO_s015
                                 + 0.600902 w_k1_MAT_s015
                                 + 0.967399 w_k1_MSFT_s015
                                 + 1.02716 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.15401 w_k1_ADM_s015
                                 + 1.37731 w_k1_AXP_s015
                                 + 1.13884 w_k1_DIS_s015
                                 + 1.08237 w_k1_FDX_s015
                                 + 1.39892 w_k1_IBM_s015
                                 + 0.889867 w_k1_INTC_s015
                                 + 1.13342 w_k1_KO_s015 + 1.96716 w_k1_MAT_s015
                                 + 0.998968 w_k1_MSFT_s015
                                 + 1.20811 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.14412 w_k1_ADM_s015
                                 + 1.39382 w_k1_AXP_s015
                                 + 1.23359 w_k1_DIS_s015
                                 + 1.22219 w_k1_FDX_s015 + 1.0162 w_k1_IBM_s015
                                 + 1.10457 w_k1_INTC_s015
                                 + 1.16578 w_k1_KO_s015 + 1.50741 w_k1_MAT_s015
                                 + 1.04753 w_k1_MSFT_s015
                                 + 1.19352 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.0105 w_k1_ADM_s016
                                 + 0.916242 w_k1_AXP_s016
                                 + 1.06961 w_k1_DIS_s016
                                 + 0.920367 w_k1_FDX_s016
                                 + 1.12197 w_k1_IBM_s016
                                 + 0.925611 w_k1_INTC_s016
                                 + 1.04934 w_k1_KO_s016 + 1.15391 w_k1_MAT_s016
                                 + 0.939046 w_k1_MSFT_s016
                                 + 1.02734 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.13988 w_k1_ADM_s016
                                 + 1.43887 w_k1_AXP_s016
                                 + 0.911995 w_k1_DIS_s016
                                 + 1.08696 w_k1_FDX_s016
                                 + 0.982612 w_k1_IBM_s016
                                 + 1.02911 w_k1_INTC_s016
                                 + 0.952508 w_k1_KO_s016
                                 + 0.959355 w_k1_MAT_s016
                                 + 0.994341 w_k1_MSFT_s016
                                 + 1.13489 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.04714 w_k1_ADM_s016
                                 + 1.12788 w_k1_AXP_s016
                                 + 0.880799 w_k1_DIS_s016
                                 + 1.11139 w_k1_FDX_s016
                                 + 0.764511 w_k1_IBM_s016
                                 + 1.03193 w_k1_INTC_s016
                                 + 0.781176 w_k1_KO_s016
                                 + 0.823428 w_k1_MAT_s016
                                 + 0.964601 w_k1_MSFT_s016
                                 + 1.11325 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.38113 w_k1_ADM_s016
                                 + 1.60017 w_k1_AXP_s016
                                 + 1.57455 w_k1_DIS_s016
                                 + 1.37246 w_k1_FDX_s016
                                 + 1.24489 w_k1_IBM_s016
                                 + 1.07673 w_k1_INTC_s016
                                 + 1.14103 w_k1_KO_s016
                                 + 0.893388 w_k1_MAT_s016
                                 + 1.05335 w_k1_MSFT_s016
                                 + 1.25709 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.896895 w_k1_ADM_s016
                                 + 0.963019 w_k1_AXP_s016
                                 + 0.657897 w_k1_DIS_s016
                                 + 0.800717 w_k1_FDX_s016
                                 + 0.764231 w_k1_IBM_s016
                                 + 0.851527 w_k1_INTC_s016
                                 + 0.708104 w_k1_KO_s016
                                 + 0.590055 w_k1_MAT_s016
                                 + 0.766776 w_k1_MSFT_s016
                                 + 0.996274 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.876269 w_k1_ADM_s016
                                 + 0.701648 w_k1_AXP_s016
                                 + 1.01559 w_k1_DIS_s016
                                 + 0.828101 w_k1_FDX_s016
                                 + 0.881691 w_k1_IBM_s016
                                 + 0.936419 w_k1_INTC_s016
                                 + 0.955416 w_k1_KO_s016
                                 + 0.619153 w_k1_MAT_s016
                                 + 0.877599 w_k1_MSFT_s016
                                 + 0.995368 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.18184 w_k1_ADM_s016
                                 + 1.27387 w_k1_AXP_s016
                                 + 1.20795 w_k1_DIS_s016
                                 + 1.11788 w_k1_FDX_s016 + 1.5082 w_k1_IBM_s016
                                 + 1.02622 w_k1_INTC_s016
                                 + 1.03505 w_k1_KO_s016 + 1.28123 w_k1_MAT_s016
                                 + 0.998176 w_k1_MSFT_s016
                                 + 1.21471 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.06554 w_k1_ADM_s016
                                 + 1.38123 w_k1_AXP_s016
                                 + 1.17683 w_k1_DIS_s016
                                 + 1.19598 w_k1_FDX_s016
                                 + 0.750214 w_k1_IBM_s016
                                 + 1.415 w_k1_INTC_s016 + 0.917479 w_k1_KO_s016
                                 + 0.602029 w_k1_MAT_s016
                                 + 1.00834 w_k1_MSFT_s016 + 1.0653 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.3856 w_k1_ADM_s017
                                 + 1.47662 w_k1_AXP_s017
                                 + 1.03687 w_k1_DIS_s017
                                 + 1.26066 w_k1_FDX_s017
                                 + 0.729075 w_k1_IBM_s017
                                 + 0.992579 w_k1_INTC_s017
                                 + 0.824242 w_k1_KO_s017
                                 + 1.24228 w_k1_MAT_s017
                                 + 0.949349 w_k1_MSFT_s017
                                 + 0.968312 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.02225 w_k1_ADM_s017
                                 + 1.02151 w_k1_AXP_s017
                                 + 0.729695 w_k1_DIS_s017
                                 + 0.939758 w_k1_FDX_s017
                                 + 0.921341 w_k1_IBM_s017
                                 + 0.714737 w_k1_INTC_s017
                                 + 0.740732 w_k1_KO_s017
                                 + 0.911574 w_k1_MAT_s017
                                 + 0.738069 w_k1_MSFT_s017
                                 + 0.821484 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.942501 w_k1_ADM_s017
                                 + 0.684124 w_k1_AXP_s017
                                 + 0.767448 w_k1_DIS_s017
                                 + 0.778647 w_k1_FDX_s017
                                 + 0.859629 w_k1_IBM_s017
                                 + 0.743449 w_k1_INTC_s017
                                 + 0.900291 w_k1_KO_s017
                                 + 0.668446 w_k1_MAT_s017
                                 + 0.820467 w_k1_MSFT_s017
                                 + 0.804028 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.981012 w_k1_ADM_s017
                                 + 0.835485 w_k1_AXP_s017
                                 + 0.760245 w_k1_DIS_s017
                                 + 0.955031 w_k1_FDX_s017
                                 + 0.708441 w_k1_IBM_s017
                                 + 1.10811 w_k1_INTC_s017
                                 + 0.77374 w_k1_KO_s017
                                 + 0.455921 w_k1_MAT_s017
                                 + 0.839295 w_k1_MSFT_s017
                                 + 0.843809 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.28061 w_k1_ADM_s017
                                 + 1.59476 w_k1_AXP_s017 + 1.4302 w_k1_DIS_s017
                                 + 1.30559 w_k1_FDX_s017
                                 + 1.31949 w_k1_IBM_s017
                                 + 1.17889 w_k1_INTC_s017
                                 + 1.29547 w_k1_KO_s017 + 1.50529 w_k1_MAT_s017
                                 + 0.951766 w_k1_MSFT_s017
                                 + 1.07791 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.10696 w_k1_ADM_s017
                                 + 0.986211 w_k1_AXP_s017
                                 + 1.11928 w_k1_DIS_s017
                                 + 0.953443 w_k1_FDX_s017
                                 + 1.45096 w_k1_IBM_s017
                                 + 1.2445 w_k1_INTC_s017 + 1.42142 w_k1_KO_s017
                                 + 0.952712 w_k1_MAT_s017
                                 + 1.0302 w_k1_MSFT_s017 + 1.01664 w_k1_PG_s017
                                 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.16003 w_k1_ADM_s017
                                 + 1.00719 w_k1_AXP_s017
                                 + 1.08114 w_k1_DIS_s017
                                 + 1.05516 w_k1_FDX_s017
                                 + 0.951782 w_k1_IBM_s017
                                 + 1.17795 w_k1_INTC_s017 + 1.0238 w_k1_KO_s017
                                 + 0.660826 w_k1_MAT_s017
                                 + 1.06784 w_k1_MSFT_s017
                                 + 1.00179 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.13634 w_k1_ADM_s017
                                 + 1.12123 w_k1_AXP_s017
                                 + 1.70396 w_k1_DIS_s017
                                 + 1.17361 w_k1_FDX_s017
                                 + 1.28655 w_k1_IBM_s017
                                 + 1.15254 w_k1_INTC_s017
                                 + 1.46759 w_k1_KO_s017 + 1.26141 w_k1_MAT_s017
                                 + 1.23449 w_k1_MSFT_s017 + 1.1118 w_k1_PG_s017
                                 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.6226 w_k1_ADM_s100
                                 + 0.624393 w_k1_AXP_s100
                                 + 0.607943 w_k1_DIS_s100
                                 + 0.703011 w_k1_FDX_s100
                                 + 0.868982 w_k1_IBM_s100
                                 + 0.790078 w_k1_INTC_s100
                                 + 0.685076 w_k1_KO_s100
                                 + 0.704789 w_k1_MAT_s100
                                 + 0.707394 w_k1_MSFT_s100
                                 + 0.757662 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.00048 w_k1_ADM_s100
                                 + 1.43749 w_k1_AXP_s100
                                 + 1.38334 w_k1_DIS_s100
                                 + 1.25415 w_k1_FDX_s100 + 1.3424 w_k1_IBM_s100
                                 + 1.32418 w_k1_INTC_s100
                                 + 1.23715 w_k1_KO_s100 + 1.41262 w_k1_MAT_s100
                                 + 1.17172 w_k1_MSFT_s100
                                 + 1.19371 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.928331 w_k1_ADM_s100
                                 + 0.881817 w_k1_AXP_s100
                                 + 0.840993 w_k1_DIS_s100
                                 + 0.874616 w_k1_FDX_s100
                                 + 1.24377 w_k1_IBM_s100
                                 + 1.03636 w_k1_INTC_s100
                                 + 0.934267 w_k1_KO_s100
                                 + 1.26344 w_k1_MAT_s100
                                 + 0.862299 w_k1_MSFT_s100
                                 + 1.04665 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.11999 w_k1_ADM_s100
                                 + 1.10011 w_k1_AXP_s100
                                 + 1.26148 w_k1_DIS_s100
                                 + 1.14417 w_k1_FDX_s100
                                 + 0.985436 w_k1_IBM_s100
                                 + 1.24063 w_k1_INTC_s100
                                 + 1.02109 w_k1_KO_s100
                                 + 0.964574 w_k1_MAT_s100
                                 + 1.13429 w_k1_MSFT_s100
                                 + 0.992144 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.06542 w_k1_ADM_s100
                                 + 1.00777 w_k1_AXP_s100
                                 + 0.820504 w_k1_DIS_s100
                                 + 0.974445 w_k1_FDX_s100
                                 + 0.90425 w_k1_IBM_s100
                                 + 1.08031 w_k1_INTC_s100
                                 + 0.813846 w_k1_KO_s100
                                 + 1.09464 w_k1_MAT_s100
                                 + 1.00272 w_k1_MSFT_s100
                                 + 1.02207 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.01973 w_k1_ADM_s100
                                 + 1.27341 w_k1_AXP_s100
                                 + 1.04066 w_k1_DIS_s100
                                 + 0.914468 w_k1_FDX_s100
                                 + 1.63863 w_k1_IBM_s100
                                 + 1.22944 w_k1_INTC_s100
                                 + 1.19503 w_k1_KO_s100 + 1.69371 w_k1_MAT_s100
                                 + 0.992285 w_k1_MSFT_s100
                                 + 1.09445 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.939174 w_k1_ADM_s100
                                 + 0.847882 w_k1_AXP_s100
                                 + 0.717514 w_k1_DIS_s100
                                 + 0.894036 w_k1_FDX_s100
                                 + 0.753648 w_k1_IBM_s100
                                 + 0.805977 w_k1_INTC_s100
                                 + 0.677412 w_k1_KO_s100
                                 + 0.573914 w_k1_MAT_s100
                                 + 0.850175 w_k1_MSFT_s100
                                 + 0.892789 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.957433 w_k1_ADM_s100
                                 + 0.978425 w_k1_AXP_s100
                                 + 0.691752 w_k1_DIS_s100
                                 + 0.844484 w_k1_FDX_s100
                                 + 0.991267 w_k1_IBM_s100
                                 + 1.03671 w_k1_INTC_s100
                                 + 0.96195 w_k1_KO_s100
                                 + 0.912651 w_k1_MAT_s100
                                 + 0.926267 w_k1_MSFT_s100
                                 + 0.928949 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.828614 w_k1_ADM_s101
                                 + 0.659618 w_k1_AXP_s101
                                 + 0.717919 w_k1_DIS_s101
                                 + 0.690687 w_k1_FDX_s101
                                 + 0.719426 w_k1_IBM_s101
                                 + 0.81844 w_k1_INTC_s101
                                 + 0.780235 w_k1_KO_s101
                                 + 0.588915 w_k1_MAT_s101
                                 + 0.735441 w_k1_MSFT_s101
                                 + 0.73094 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.875344 w_k1_ADM_s101
                                 + 0.702736 w_k1_AXP_s101
                                 + 0.808239 w_k1_DIS_s101
                                 + 0.758392 w_k1_FDX_s101
                                 + 0.693184 w_k1_IBM_s101
                                 + 0.828754 w_k1_INTC_s101
                                 + 0.796691 w_k1_KO_s101
                                 + 0.669037 w_k1_MAT_s101
                                 + 0.836823 w_k1_MSFT_s101
                                 + 0.910892 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.0427 w_k1_ADM_s101
                                 + 1.46214 w_k1_AXP_s101
                                 + 1.51474 w_k1_DIS_s101
                                 + 1.18827 w_k1_FDX_s101
                                 + 1.52695 w_k1_IBM_s101
                                 + 1.20514 w_k1_INTC_s101
                                 + 1.61078 w_k1_KO_s101 + 1.52624 w_k1_MAT_s101
                                 + 1.144 w_k1_MSFT_s101 + 1.18126 w_k1_PG_s101
                                 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.14879 w_k1_ADM_s101
                                 + 0.845737 w_k1_AXP_s101
                                 + 1.05354 w_k1_DIS_s101
                                 + 1.06004 w_k1_FDX_s101
                                 + 0.63041 w_k1_IBM_s101
                                 + 1.18485 w_k1_INTC_s101
                                 + 0.815023 w_k1_KO_s101 + 0.5624 w_k1_MAT_s101
                                 + 0.934477 w_k1_MSFT_s101
                                 + 0.962801 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.09967 w_k1_ADM_s101
                                 + 1.33406 w_k1_AXP_s101
                                 + 1.74323 w_k1_DIS_s101
                                 + 1.32942 w_k1_FDX_s101
                                 + 0.980308 w_k1_IBM_s101
                                 + 1.28163 w_k1_INTC_s101
                                 + 1.33177 w_k1_KO_s101 + 0.99499 w_k1_MAT_s101
                                 + 1.24206 w_k1_MSFT_s101 + 1.2027 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.08833 w_k1_ADM_s101
                                 + 0.898988 w_k1_AXP_s101
                                 + 1.03124 w_k1_DIS_s101
                                 + 1.00966 w_k1_FDX_s101
                                 + 0.892007 w_k1_IBM_s101
                                 + 1.04155 w_k1_INTC_s101
                                 + 1.04314 w_k1_KO_s101
                                 + 0.686543 w_k1_MAT_s101
                                 + 0.938582 w_k1_MSFT_s101
                                 + 0.948278 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.86717 w_k1_ADM_s101
                                 + 1.06891 w_k1_AXP_s101
                                 + 1.04519 w_k1_DIS_s101
                                 + 0.948183 w_k1_FDX_s101
                                 + 1.24619 w_k1_IBM_s101
                                 + 1.09537 w_k1_INTC_s101
                                 + 1.22556 w_k1_KO_s101 + 1.23203 w_k1_MAT_s101
                                 + 0.957851 w_k1_MSFT_s101
                                 + 1.06619 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.0065 w_k1_ADM_s101
                                 + 1.14517 w_k1_AXP_s101
                                 + 0.99669 w_k1_DIS_s101
                                 + 0.913976 w_k1_FDX_s101
                                 + 0.840405 w_k1_IBM_s101
                                 + 0.952614 w_k1_INTC_s101
                                 + 0.955967 w_k1_KO_s101
                                 + 0.895547 w_k1_MAT_s101
                                 + 0.859996 w_k1_MSFT_s101
                                 + 0.982732 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.18942 w_k1_ADM_s102
                                 + 1.54061 w_k1_AXP_s102
                                 + 1.58029 w_k1_DIS_s102
                                 + 1.40304 w_k1_FDX_s102
                                 + 1.07241 w_k1_IBM_s102
                                 + 1.03587 w_k1_INTC_s102
                                 + 1.12508 w_k1_KO_s102 + 1.38919 w_k1_MAT_s102
                                 + 1.12263 w_k1_MSFT_s102
                                 + 1.31768 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.746138 w_k1_ADM_s102
                                 + 0.767844 w_k1_AXP_s102
                                 + 0.652075 w_k1_DIS_s102
                                 + 0.666285 w_k1_FDX_s102
                                 + 1.01698 w_k1_IBM_s102
                                 + 0.815506 w_k1_INTC_s102
                                 + 0.777673 w_k1_KO_s102
                                 + 0.504666 w_k1_MAT_s102
                                 + 0.66761 w_k1_MSFT_s102
                                 + 0.842726 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.855499 w_k1_ADM_s102
                                 + 1.1047 w_k1_AXP_s102 + 1.11304 w_k1_DIS_s102
                                 + 1.07398 w_k1_FDX_s102
                                 + 1.48803 w_k1_IBM_s102
                                 + 0.969688 w_k1_INTC_s102
                                 + 1.20229 w_k1_KO_s102 + 1.41987 w_k1_MAT_s102
                                 + 0.959142 w_k1_MSFT_s102
                                 + 1.18321 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.13018 w_k1_ADM_s102
                                 + 1.06495 w_k1_AXP_s102
                                 + 1.15878 w_k1_DIS_s102
                                 + 1.16526 w_k1_FDX_s102
                                 + 0.746142 w_k1_IBM_s102
                                 + 1.04221 w_k1_INTC_s102
                                 + 0.951733 w_k1_KO_s102
                                 + 0.696664 w_k1_MAT_s102
                                 + 1.14779 w_k1_MSFT_s102
                                 + 1.14042 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.984147 w_k1_ADM_s102
                                 + 1.14926 w_k1_AXP_s102
                                 + 1.05367 w_k1_DIS_s102
                                 + 1.04521 w_k1_FDX_s102
                                 + 1.03938 w_k1_IBM_s102
                                 + 0.827373 w_k1_INTC_s102
                                 + 0.97113 w_k1_KO_s102
                                 + 0.858437 w_k1_MAT_s102
                                 + 0.865057 w_k1_MSFT_s102
                                 + 1.06768 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.24319 w_k1_ADM_s102
                                 + 1.38744 w_k1_AXP_s102
                                 + 0.848353 w_k1_DIS_s102
                                 + 1.06281 w_k1_FDX_s102
                                 + 0.781314 w_k1_IBM_s102
                                 + 0.963312 w_k1_INTC_s102
                                 + 0.873199 w_k1_KO_s102
                                 + 0.787354 w_k1_MAT_s102
                                 + 0.958009 w_k1_MSFT_s102
                                 + 1.06293 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.824497 w_k1_ADM_s102
                                 + 0.721204 w_k1_AXP_s102
                                 + 0.759951 w_k1_DIS_s102
                                 + 0.779535 w_k1_FDX_s102
                                 + 0.756796 w_k1_IBM_s102
                                 + 0.74683 w_k1_INTC_s102
                                 + 0.894601 w_k1_KO_s102
                                 + 0.582944 w_k1_MAT_s102
                                 + 0.702793 w_k1_MSFT_s102
                                 + 0.908199 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.914802 w_k1_ADM_s102
                                 + 0.920073 w_k1_AXP_s102
                                 + 1.27591 w_k1_DIS_s102
                                 + 0.979035 w_k1_FDX_s102
                                 + 0.936142 w_k1_IBM_s102
                                 + 0.955342 w_k1_INTC_s102
                                 + 0.972801 w_k1_KO_s102
                                 + 0.901244 w_k1_MAT_s102
                                 + 1.00119 w_k1_MSFT_s102
                                 + 1.02857 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.02682 w_k1_ADM_s103
                                 + 1.06494 w_k1_AXP_s103
                                 + 1.07633 w_k1_DIS_s103
                                 + 0.865009 w_k1_FDX_s103
                                 + 1.68692 w_k1_IBM_s103
                                 + 0.666749 w_k1_INTC_s103
                                 + 1.14702 w_k1_KO_s103 + 1.55742 w_k1_MAT_s103
                                 + 0.701443 w_k1_MSFT_s103
                                 + 1.00251 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.820192 w_k1_ADM_s103
                                 + 0.951207 w_k1_AXP_s103
                                 + 1.01776 w_k1_DIS_s103
                                 + 0.884922 w_k1_FDX_s103
                                 + 1.06274 w_k1_IBM_s103
                                 + 0.813367 w_k1_INTC_s103
                                 + 0.896234 w_k1_KO_s103 + 1.3497 w_k1_MAT_s103
                                 + 0.921279 w_k1_MSFT_s103
                                 + 0.944609 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.883278 w_k1_ADM_s103
                                 + 0.696907 w_k1_AXP_s103
                                 + 0.67042 w_k1_DIS_s103
                                 + 0.800616 w_k1_FDX_s103
                                 + 0.65566 w_k1_IBM_s103
                                 + 0.684035 w_k1_INTC_s103
                                 + 0.771305 w_k1_KO_s103
                                 + 0.61085 w_k1_MAT_s103
                                 + 0.794102 w_k1_MSFT_s103
                                 + 0.896695 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.818069 w_k1_ADM_s103
                                 + 0.815034 w_k1_AXP_s103
                                 + 0.959437 w_k1_DIS_s103
                                 + 0.927036 w_k1_FDX_s103
                                 + 0.896199 w_k1_IBM_s103
                                 + 0.925769 w_k1_INTC_s103
                                 + 1.03969 w_k1_KO_s103 + 1.01875 w_k1_MAT_s103
                                 + 0.846807 w_k1_MSFT_s103
                                 + 0.994575 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.958121 w_k1_ADM_s103
                                 + 1.40811 w_k1_AXP_s103 + 1.2326 w_k1_DIS_s103
                                 + 1.02368 w_k1_FDX_s103
                                 + 1.43338 w_k1_IBM_s103
                                 + 1.07383 w_k1_INTC_s103
                                 + 1.27012 w_k1_KO_s103 + 1.59294 w_k1_MAT_s103
                                 + 0.946415 w_k1_MSFT_s103
                                 + 1.02219 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.895722 w_k1_ADM_s103
                                 + 1.30027 w_k1_AXP_s103
                                 + 1.22754 w_k1_DIS_s103
                                 + 1.03536 w_k1_FDX_s103
                                 + 1.15481 w_k1_IBM_s103
                                 + 0.963298 w_k1_INTC_s103
                                 + 1.07872 w_k1_KO_s103
                                 + 0.980294 w_k1_MAT_s103
                                 + 0.901341 w_k1_MSFT_s103
                                 + 1.00442 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.04606 w_k1_ADM_s103
                                 + 1.09664 w_k1_AXP_s103
                                 + 1.08558 w_k1_DIS_s103
                                 + 1.12607 w_k1_FDX_s103
                                 + 0.686347 w_k1_IBM_s103
                                 + 0.999163 w_k1_INTC_s103
                                 + 0.777627 w_k1_KO_s103
                                 + 0.741815 w_k1_MAT_s103
                                 + 0.961949 w_k1_MSFT_s103
                                 + 0.93059 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.25448 w_k1_ADM_s103
                                 + 1.58689 w_k1_AXP_s103
                                 + 1.69031 w_k1_DIS_s103
                                 + 1.21731 w_k1_FDX_s103
                                 + 1.13405 w_k1_IBM_s103
                                 + 1.08407 w_k1_INTC_s103
                                 + 1.33975 w_k1_KO_s103 + 1.48231 w_k1_MAT_s103
                                 + 1.14658 w_k1_MSFT_s103
                                 + 1.14063 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.08417 w_k1_ADM_s104
                                 + 1.2955 w_k1_AXP_s104 + 1.22001 w_k1_DIS_s104
                                 + 1.19966 w_k1_FDX_s104 + 1.407 w_k1_IBM_s104
                                 + 1.22765 w_k1_INTC_s104 + 1.2314 w_k1_KO_s104
                                 + 1.38703 w_k1_MAT_s104
                                 + 1.25545 w_k1_MSFT_s104
                                 + 1.16552 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.905235 w_k1_ADM_s104
                                 + 0.930421 w_k1_AXP_s104
                                 + 0.560044 w_k1_DIS_s104
                                 + 0.742989 w_k1_FDX_s104
                                 + 1.0597 w_k1_IBM_s104
                                 + 0.79528 w_k1_INTC_s104
                                 + 0.845935 w_k1_KO_s104
                                 + 1.14943 w_k1_MAT_s104
                                 + 0.732993 w_k1_MSFT_s104
                                 + 0.978411 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.20415 w_k1_ADM_s104
                                 + 1.06182 w_k1_AXP_s104
                                 + 1.12906 w_k1_DIS_s104
                                 + 0.959213 w_k1_FDX_s104
                                 + 1.51831 w_k1_IBM_s104
                                 + 0.991768 w_k1_INTC_s104
                                 + 1.27049 w_k1_KO_s104 + 1.20582 w_k1_MAT_s104
                                 + 1.05811 w_k1_MSFT_s104
                                 + 1.04859 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.331 w_k1_ADM_s104
                                 + 1.23214 w_k1_AXP_s104
                                 + 0.964015 w_k1_DIS_s104
                                 + 1.19599 w_k1_FDX_s104
                                 + 0.786028 w_k1_IBM_s104
                                 + 0.873265 w_k1_INTC_s104
                                 + 0.894363 w_k1_KO_s104
                                 + 0.940773 w_k1_MAT_s104
                                 + 0.963256 w_k1_MSFT_s104
                                 + 1.07377 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.957147 w_k1_ADM_s104
                                 + 0.852651 w_k1_AXP_s104
                                 + 0.969823 w_k1_DIS_s104
                                 + 1.04711 w_k1_FDX_s104
                                 + 0.672344 w_k1_IBM_s104
                                 + 1.08134 w_k1_INTC_s104
                                 + 0.902436 w_k1_KO_s104
                                 + 0.763865 w_k1_MAT_s104
                                 + 1.07912 w_k1_MSFT_s104
                                 + 0.979971 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.07371 w_k1_ADM_s104
                                 + 1.18198 w_k1_AXP_s104
                                 + 0.948003 w_k1_DIS_s104
                                 + 1.06126 w_k1_FDX_s104
                                 + 0.988801 w_k1_IBM_s104
                                 + 0.982547 w_k1_INTC_s104
                                 + 0.902579 w_k1_KO_s104
                                 + 1.14482 w_k1_MAT_s104
                                 + 1.11678 w_k1_MSFT_s104
                                 + 1.01397 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.3525 w_k1_ADM_s104
                                 + 1.52892 w_k1_AXP_s104
                                 + 1.63948 w_k1_DIS_s104
                                 + 1.36712 w_k1_FDX_s104
                                 + 1.13555 w_k1_IBM_s104
                                 + 1.10013 w_k1_INTC_s104
                                 + 1.24648 w_k1_KO_s104 + 1.36214 w_k1_MAT_s104
                                 + 1.42566 w_k1_MSFT_s104
                                 + 1.29575 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.02422 w_k1_ADM_s104
                                 + 0.814281 w_k1_AXP_s104
                                 + 0.74664 w_k1_DIS_s104
                                 + 0.866171 w_k1_FDX_s104
                                 + 0.893588 w_k1_IBM_s104
                                 + 0.977375 w_k1_INTC_s104
                                 + 0.801472 w_k1_KO_s104
                                 + 0.666851 w_k1_MAT_s104
                                 + 0.910127 w_k1_MSFT_s104
                                 + 0.913495 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.19826 w_k1_ADM_s105
                                 + 1.16632 w_k1_AXP_s105
                                 + 1.38002 w_k1_DIS_s105
                                 + 1.07242 w_k1_FDX_s105
                                 + 1.16874 w_k1_IBM_s105
                                 + 0.969003 w_k1_INTC_s105
                                 + 1.15778 w_k1_KO_s105 + 1.32 w_k1_MAT_s105
                                 + 0.993366 w_k1_MSFT_s105
                                 + 1.16146 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.22855 w_k1_ADM_s105
                                 + 1.15933 w_k1_AXP_s105 + 1.5917 w_k1_DIS_s105
                                 + 1.21531 w_k1_FDX_s105
                                 + 1.50695 w_k1_IBM_s105
                                 + 1.14241 w_k1_INTC_s105
                                 + 1.47488 w_k1_KO_s105 + 1.15227 w_k1_MAT_s105
                                 + 1.24329 w_k1_MSFT_s105
                                 + 1.25602 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.34102 w_k1_ADM_s105
                                 + 1.4282 w_k1_AXP_s105 + 1.32832 w_k1_DIS_s105
                                 + 1.05802 w_k1_FDX_s105
                                 + 1.84421 w_k1_IBM_s105
                                 + 0.758673 w_k1_INTC_s105
                                 + 1.15548 w_k1_KO_s105 + 1.66903 w_k1_MAT_s105
                                 + 0.916934 w_k1_MSFT_s105
                                 + 1.20472 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.12853 w_k1_ADM_s105
                                 + 1.31892 w_k1_AXP_s105
                                 + 1.24827 w_k1_DIS_s105
                                 + 1.20236 w_k1_FDX_s105
                                 + 0.924467 w_k1_IBM_s105
                                 + 1.01645 w_k1_INTC_s105
                                 + 1.13377 w_k1_KO_s105
                                 + 0.845677 w_k1_MAT_s105
                                 + 1.09299 w_k1_MSFT_s105
                                 + 1.03965 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.16774 w_k1_ADM_s105
                                 + 0.926071 w_k1_AXP_s105
                                 + 0.86326 w_k1_DIS_s105
                                 + 1.00695 w_k1_FDX_s105
                                 + 0.745179 w_k1_IBM_s105
                                 + 0.875227 w_k1_INTC_s105
                                 + 0.783691 w_k1_KO_s105
                                 + 0.77272 w_k1_MAT_s105
                                 + 0.907251 w_k1_MSFT_s105
                                 + 0.981795 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.981825 w_k1_ADM_s105
                                 + 0.845427 w_k1_AXP_s105
                                 + 1.12034 w_k1_DIS_s105
                                 + 0.913314 w_k1_FDX_s105
                                 + 1.54266 w_k1_IBM_s105
                                 + 0.856505 w_k1_INTC_s105
                                 + 1.26322 w_k1_KO_s105 + 1.64211 w_k1_MAT_s105
                                 + 0.937042 w_k1_MSFT_s105
                                 + 1.15993 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.931931 w_k1_ADM_s105
                                 + 0.799958 w_k1_AXP_s105
                                 + 0.726088 w_k1_DIS_s105
                                 + 0.817004 w_k1_FDX_s105
                                 + 0.789173 w_k1_IBM_s105
                                 + 0.694227 w_k1_INTC_s105
                                 + 0.707094 w_k1_KO_s105
                                 + 0.677491 w_k1_MAT_s105
                                 + 0.730998 w_k1_MSFT_s105
                                 + 0.931046 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.932546 w_k1_ADM_s105
                                 + 0.846701 w_k1_AXP_s105
                                 + 0.872556 w_k1_DIS_s105
                                 + 0.800074 w_k1_FDX_s105
                                 + 1.06592 w_k1_IBM_s105
                                 + 0.732402 w_k1_INTC_s105
                                 + 0.975004 w_k1_KO_s105
                                 + 0.896261 w_k1_MAT_s105
                                 + 0.759924 w_k1_MSFT_s105
                                 + 0.917968 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.959403 w_k1_ADM_s106
                                 + 0.707203 w_k1_AXP_s106
                                 + 0.707247 w_k1_DIS_s106
                                 + 0.722274 w_k1_FDX_s106
                                 + 1.17527 w_k1_IBM_s106
                                 + 0.729783 w_k1_INTC_s106
                                 + 0.998463 w_k1_KO_s106
                                 + 0.980456 w_k1_MAT_s106
                                 + 0.834561 w_k1_MSFT_s106
                                 + 0.968202 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.727332 w_k1_ADM_s106
                                 + 0.646243 w_k1_AXP_s106
                                 + 0.573154 w_k1_DIS_s106
                                 + 0.755477 w_k1_FDX_s106
                                 + 0.810845 w_k1_IBM_s106
                                 + 0.694742 w_k1_INTC_s106
                                 + 0.785429 w_k1_KO_s106
                                 + 0.892567 w_k1_MAT_s106
                                 + 0.769261 w_k1_MSFT_s106
                                 + 0.866033 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.11955 w_k1_ADM_s106
                                 + 1.1065 w_k1_AXP_s106
                                 + 0.974656 w_k1_DIS_s106
                                 + 0.99723 w_k1_FDX_s106
                                 + 1.46443 w_k1_IBM_s106
                                 + 0.823157 w_k1_INTC_s106
                                 + 1.12476 w_k1_KO_s106 + 1.51902 w_k1_MAT_s106
                                 + 1.0456 w_k1_MSFT_s106 + 1.13935 w_k1_PG_s106
                                 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.02132 w_k1_ADM_s106
                                 + 0.987309 w_k1_AXP_s106
                                 + 0.967768 w_k1_DIS_s106
                                 + 1.17243 w_k1_FDX_s106
                                 + 0.76426 w_k1_IBM_s106
                                 + 1.04243 w_k1_INTC_s106
                                 + 0.898037 w_k1_KO_s106
                                 + 0.687596 w_k1_MAT_s106
                                 + 1.19497 w_k1_MSFT_s106
                                 + 1.00983 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.13825 w_k1_ADM_s106
                                 + 1.44678 w_k1_AXP_s106
                                 + 1.64535 w_k1_DIS_s106
                                 + 1.34219 w_k1_FDX_s106
                                 + 1.42541 w_k1_IBM_s106
                                 + 1.16787 w_k1_INTC_s106
                                 + 1.54601 w_k1_KO_s106 + 1.30825 w_k1_MAT_s106
                                 + 1.47262 w_k1_MSFT_s106
                                 + 1.26559 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.44236 w_k1_ADM_s106
                                 + 1.22697 w_k1_AXP_s106
                                 + 1.38727 w_k1_DIS_s106
                                 + 1.22636 w_k1_FDX_s106 + 1.0883 w_k1_IBM_s106
                                 + 1.01607 w_k1_INTC_s106
                                 + 1.26649 w_k1_KO_s106 + 1.32619 w_k1_MAT_s106
                                 + 1.29208 w_k1_MSFT_s106
                                 + 1.24653 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.21416 w_k1_ADM_s106
                                 + 1.33563 w_k1_AXP_s106
                                 + 1.19777 w_k1_DIS_s106
                                 + 1.39969 w_k1_FDX_s106
                                 + 0.877873 w_k1_IBM_s106
                                 + 1.07837 w_k1_INTC_s106 + 1.0187 w_k1_KO_s106
                                 + 1.18173 w_k1_MAT_s106
                                 + 1.44958 w_k1_MSFT_s106
                                 + 1.19899 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.10506 w_k1_ADM_s106
                                 + 1.08303 w_k1_AXP_s106
                                 + 0.91704 w_k1_DIS_s106
                                 + 0.963648 w_k1_FDX_s106
                                 + 0.897863 w_k1_IBM_s106
                                 + 0.864789 w_k1_INTC_s106
                                 + 1.05449 w_k1_KO_s106
                                 + 0.975962 w_k1_MAT_s106
                                 + 1.08396 w_k1_MSFT_s106
                                 + 0.992353 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.821126 w_k1_ADM_s107
                                 + 0.832272 w_k1_AXP_s107
                                 + 1.03076 w_k1_DIS_s107
                                 + 1.01981 w_k1_FDX_s107
                                 + 0.917111 w_k1_IBM_s107
                                 + 1.08589 w_k1_INTC_s107
                                 + 1.21409 w_k1_KO_s107 + 1.18316 w_k1_MAT_s107
                                 + 1.08791 w_k1_MSFT_s107
                                 + 1.01026 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.00339 w_k1_ADM_s107
                                 + 1.11185 w_k1_AXP_s107
                                 + 1.17891 w_k1_DIS_s107
                                 + 0.972251 w_k1_FDX_s107
                                 + 1.29775 w_k1_IBM_s107
                                 + 1.18752 w_k1_INTC_s107
                                 + 1.35419 w_k1_KO_s107 + 1.46563 w_k1_MAT_s107
                                 + 1.01263 w_k1_MSFT_s107
                                 + 1.01533 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.40141 w_k1_ADM_s107
                                 + 1.32072 w_k1_AXP_s107
                                 + 1.57669 w_k1_DIS_s107
                                 + 1.29877 w_k1_FDX_s107
                                 + 1.10158 w_k1_IBM_s107
                                 + 1.19433 w_k1_INTC_s107 + 1.1663 w_k1_KO_s107
                                 + 1.3258 w_k1_MAT_s107
                                 + 1.36907 w_k1_MSFT_s107
                                 + 1.35199 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.793056 w_k1_ADM_s107
                                 + 0.67222 w_k1_AXP_s107 + 0.746 w_k1_DIS_s107
                                 + 0.818024 w_k1_FDX_s107
                                 + 0.728046 w_k1_IBM_s107
                                 + 0.851487 w_k1_INTC_s107
                                 + 0.907899 w_k1_KO_s107
                                 + 0.95763 w_k1_MAT_s107
                                 + 0.918638 w_k1_MSFT_s107
                                 + 0.907984 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.33972 w_k1_ADM_s107
                                 + 1.6361 w_k1_AXP_s107 + 1.67565 w_k1_DIS_s107
                                 + 1.45156 w_k1_FDX_s107
                                 + 1.48817 w_k1_IBM_s107
                                 + 1.32528 w_k1_INTC_s107
                                 + 1.55392 w_k1_KO_s107 + 1.83208 w_k1_MAT_s107
                                 + 1.31142 w_k1_MSFT_s107
                                 + 1.43753 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.893324 w_k1_ADM_s107
                                 + 0.968482 w_k1_AXP_s107
                                 + 0.864523 w_k1_DIS_s107
                                 + 0.887012 w_k1_FDX_s107
                                 + 0.924008 w_k1_IBM_s107
                                 + 1.00415 w_k1_INTC_s107
                                 + 0.98596 w_k1_KO_s107 + 1.29667 w_k1_MAT_s107
                                 + 0.994619 w_k1_MSFT_s107
                                 + 1.00516 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.33242 w_k1_ADM_s107
                                 + 1.35895 w_k1_AXP_s107
                                 + 1.41419 w_k1_DIS_s107
                                 + 1.31684 w_k1_FDX_s107 + 1.0056 w_k1_IBM_s107
                                 + 1.33558 w_k1_INTC_s107
                                 + 1.18111 w_k1_KO_s107 + 1.15162 w_k1_MAT_s107
                                 + 1.18724 w_k1_MSFT_s107
                                 + 1.11296 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.845408 w_k1_ADM_s107
                                 + 0.783871 w_k1_AXP_s107
                                 + 0.895928 w_k1_DIS_s107
                                 + 0.937781 w_k1_FDX_s107
                                 + 0.62415 w_k1_IBM_s107
                                 + 0.963261 w_k1_INTC_s107
                                 + 0.797827 w_k1_KO_s107
                                 + 0.833456 w_k1_MAT_s107
                                 + 1.03564 w_k1_MSFT_s107
                                 + 0.993097 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.06332 w_k1_ADM_s110
                                 + 0.918671 w_k1_AXP_s110
                                 + 0.908787 w_k1_DIS_s110
                                 + 0.923067 w_k1_FDX_s110
                                 + 0.967114 w_k1_IBM_s110
                                 + 0.888234 w_k1_INTC_s110
                                 + 0.917635 w_k1_KO_s110
                                 + 1.05484 w_k1_MAT_s110
                                 + 0.929781 w_k1_MSFT_s110
                                 + 0.961513 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.42104 w_k1_ADM_s110
                                 + 1.19862 w_k1_AXP_s110
                                 + 1.44322 w_k1_DIS_s110 + 1.3142 w_k1_FDX_s110
                                 + 0.956863 w_k1_IBM_s110
                                 + 1.25573 w_k1_INTC_s110
                                 + 1.08912 w_k1_KO_s110
                                 + 0.945113 w_k1_MAT_s110
                                 + 1.3237 w_k1_MSFT_s110 + 1.10774 w_k1_PG_s110
                                 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.11708 w_k1_ADM_s110
                                 + 0.991356 w_k1_AXP_s110
                                 + 0.802728 w_k1_DIS_s110
                                 + 0.917509 w_k1_FDX_s110
                                 + 1.26096 w_k1_IBM_s110
                                 + 0.98764 w_k1_INTC_s110
                                 + 0.856102 w_k1_KO_s110
                                 + 1.02055 w_k1_MAT_s110
                                 + 0.922178 w_k1_MSFT_s110
                                 + 0.957541 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.836762 w_k1_ADM_s110
                                 + 0.781925 w_k1_AXP_s110
                                 + 0.677125 w_k1_DIS_s110
                                 + 0.790795 w_k1_FDX_s110
                                 + 0.902931 w_k1_IBM_s110
                                 + 0.76648 w_k1_INTC_s110
                                 + 0.747639 w_k1_KO_s110
                                 + 0.818729 w_k1_MAT_s110
                                 + 0.7532 w_k1_MSFT_s110
                                 + 0.867179 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.22025 w_k1_ADM_s110
                                 + 1.10241 w_k1_AXP_s110 + 1.2443 w_k1_DIS_s110
                                 + 1.17842 w_k1_FDX_s110
                                 + 1.44969 w_k1_IBM_s110
                                 + 0.792013 w_k1_INTC_s110
                                 + 1.21644 w_k1_KO_s110 + 1.75802 w_k1_MAT_s110
                                 + 1.07655 w_k1_MSFT_s110
                                 + 1.15076 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.983937 w_k1_ADM_s110
                                 + 1.36383 w_k1_AXP_s110
                                 + 1.01163 w_k1_DIS_s110
                                 + 1.11539 w_k1_FDX_s110
                                 + 1.04848 w_k1_IBM_s110
                                 + 1.1056 w_k1_INTC_s110
                                 + 0.925414 w_k1_KO_s110
                                 + 1.63825 w_k1_MAT_s110
                                 + 1.00565 w_k1_MSFT_s110
                                 + 1.01804 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.01582 w_k1_ADM_s110
                                 + 0.665898 w_k1_AXP_s110
                                 + 0.591907 w_k1_DIS_s110
                                 + 0.807386 w_k1_FDX_s110
                                 + 0.666064 w_k1_IBM_s110
                                 + 0.723491 w_k1_INTC_s110
                                 + 0.710036 w_k1_KO_s110
                                 + 0.635235 w_k1_MAT_s110
                                 + 0.764913 w_k1_MSFT_s110
                                 + 0.75828 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.876913 w_k1_ADM_s110
                                 + 0.924546 w_k1_AXP_s110
                                 + 0.985342 w_k1_DIS_s110
                                 + 1.07719 w_k1_FDX_s110
                                 + 0.723797 w_k1_IBM_s110
                                 + 0.942912 w_k1_INTC_s110
                                 + 0.787173 w_k1_KO_s110
                                 + 0.895899 w_k1_MAT_s110
                                 + 1.10909 w_k1_MSFT_s110
                                 + 0.960902 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.54233 w_k1_ADM_s111
                                 + 1.70126 w_k1_AXP_s111
                                 + 2.16547 w_k1_DIS_s111
                                 + 1.58229 w_k1_FDX_s111
                                 + 1.63583 w_k1_IBM_s111
                                 + 1.42278 w_k1_INTC_s111
                                 + 1.82478 w_k1_KO_s111 + 1.62437 w_k1_MAT_s111
                                 + 1.56214 w_k1_MSFT_s111
                                 + 1.38481 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.00581 w_k1_ADM_s111
                                 + 1.041 w_k1_AXP_s111 + 1.02058 w_k1_DIS_s111
                                 + 1.02881 w_k1_FDX_s111
                                 + 0.786251 w_k1_IBM_s111
                                 + 1.12601 w_k1_INTC_s111
                                 + 0.955079 w_k1_KO_s111
                                 + 1.06664 w_k1_MAT_s111
                                 + 1.00549 w_k1_MSFT_s111
                                 + 0.963269 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.921925 w_k1_ADM_s111
                                 + 0.856984 w_k1_AXP_s111
                                 + 1.10462 w_k1_DIS_s111
                                 + 1.05928 w_k1_FDX_s111
                                 + 0.830607 w_k1_IBM_s111
                                 + 1.13078 w_k1_INTC_s111
                                 + 0.885074 w_k1_KO_s111
                                 + 0.836355 w_k1_MAT_s111
                                 + 1.04396 w_k1_MSFT_s111
                                 + 0.921799 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.636238 w_k1_ADM_s111
                                 + 0.558861 w_k1_AXP_s111
                                 + 0.541882 w_k1_DIS_s111
                                 + 0.613578 w_k1_FDX_s111
                                 + 0.968174 w_k1_IBM_s111
                                 + 0.845958 w_k1_INTC_s111
                                 + 0.840123 w_k1_KO_s111
                                 + 0.949112 w_k1_MAT_s111
                                 + 0.754689 w_k1_MSFT_s111
                                 + 0.82933 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.985732 w_k1_ADM_s111
                                 + 0.783045 w_k1_AXP_s111
                                 + 0.790482 w_k1_DIS_s111
                                 + 0.812525 w_k1_FDX_s111
                                 + 0.70317 w_k1_IBM_s111
                                 + 0.933331 w_k1_INTC_s111
                                 + 0.751442 w_k1_KO_s111
                                 + 0.758481 w_k1_MAT_s111
                                 + 0.887427 w_k1_MSFT_s111
                                 + 0.791903 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.892239 w_k1_ADM_s111
                                 + 0.882683 w_k1_AXP_s111
                                 + 0.897403 w_k1_DIS_s111
                                 + 0.830806 w_k1_FDX_s111
                                 + 1.32372 w_k1_IBM_s111
                                 + 1.08385 w_k1_INTC_s111
                                 + 1.07093 w_k1_KO_s111 + 1.27678 w_k1_MAT_s111
                                 + 0.935583 w_k1_MSFT_s111
                                 + 0.891049 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.49899 w_k1_ADM_s111
                                 + 1.56863 w_k1_AXP_s111
                                 + 1.20547 w_k1_DIS_s111
                                 + 1.25879 w_k1_FDX_s111
                                 + 0.795355 w_k1_IBM_s111
                                 + 1.13931 w_k1_INTC_s111
                                 + 0.929205 w_k1_KO_s111
                                 + 0.941995 w_k1_MAT_s111
                                 + 1.13952 w_k1_MSFT_s111
                                 + 1.03737 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.29591 w_k1_ADM_s111
                                 + 1.38792 w_k1_AXP_s111
                                 + 1.44273 w_k1_DIS_s111
                                 + 1.19945 w_k1_FDX_s111
                                 + 1.26436 w_k1_IBM_s111
                                 + 1.14078 w_k1_INTC_s111
                                 + 1.15728 w_k1_KO_s111 + 1.30048 w_k1_MAT_s111
                                 + 1.15419 w_k1_MSFT_s111
                                 + 1.00951 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.881236 w_k1_ADM_s112
                                 + 0.786643 w_k1_AXP_s112
                                 + 0.742763 w_k1_DIS_s112
                                 + 0.853059 w_k1_FDX_s112
                                 + 0.710643 w_k1_IBM_s112
                                 + 1.07944 w_k1_INTC_s112
                                 + 0.822008 w_k1_KO_s112
                                 + 0.585073 w_k1_MAT_s112
                                 + 0.914574 w_k1_MSFT_s112
                                 + 0.846919 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.867888 w_k1_ADM_s112
                                 + 1.31121 w_k1_AXP_s112
                                 + 1.15711 w_k1_DIS_s112
                                 + 1.08751 w_k1_FDX_s112
                                 + 1.59986 w_k1_IBM_s112
                                 + 1.13806 w_k1_INTC_s112
                                 + 1.48919 w_k1_KO_s112 + 1.98486 w_k1_MAT_s112
                                 + 1.12414 w_k1_MSFT_s112 + 1.1575 w_k1_PG_s112
                                 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.859465 w_k1_ADM_s112
                                 + 0.980111 w_k1_AXP_s112
                                 + 1.00063 w_k1_DIS_s112
                                 + 0.975538 w_k1_FDX_s112
                                 + 0.870372 w_k1_IBM_s112
                                 + 1.03882 w_k1_INTC_s112
                                 + 1.02188 w_k1_KO_s112 + 1.0062 w_k1_MAT_s112
                                 + 0.991001 w_k1_MSFT_s112
                                 + 1.01739 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.07265 w_k1_ADM_s112
                                 + 1.30493 w_k1_AXP_s112
                                 + 1.22794 w_k1_DIS_s112
                                 + 1.19034 w_k1_FDX_s112
                                 + 0.98076 w_k1_IBM_s112
                                 + 1.1777 w_k1_INTC_s112 + 1.1286 w_k1_KO_s112
                                 + 1.14919 w_k1_MAT_s112
                                 + 1.0547 w_k1_MSFT_s112 + 1.07518 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.738461 w_k1_ADM_s112
                                 + 0.874864 w_k1_AXP_s112
                                 + 0.714167 w_k1_DIS_s112
                                 + 0.7834 w_k1_FDX_s112 + 1.05307 w_k1_IBM_s112
                                 + 0.934462 w_k1_INTC_s112
                                 + 0.867094 w_k1_KO_s112
                                 + 0.991288 w_k1_MAT_s112
                                 + 0.832675 w_k1_MSFT_s112
                                 + 0.910828 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.00581 w_k1_ADM_s112
                                 + 0.923108 w_k1_AXP_s112
                                 + 1.15397 w_k1_DIS_s112
                                 + 0.904863 w_k1_FDX_s112
                                 + 1.38345 w_k1_IBM_s112
                                 + 1.26972 w_k1_INTC_s112
                                 + 1.12978 w_k1_KO_s112
                                 + 0.718462 w_k1_MAT_s112
                                 + 1.017 w_k1_MSFT_s112 + 1.1006 w_k1_PG_s112
                                 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.48215 w_k1_ADM_s112
                                 + 1.42784 w_k1_AXP_s112
                                 + 1.52965 w_k1_DIS_s112
                                 + 1.34854 w_k1_FDX_s112
                                 + 0.879591 w_k1_IBM_s112
                                 + 1.45345 w_k1_INTC_s112
                                 + 1.31325 w_k1_KO_s112 + 0.90785 w_k1_MAT_s112
                                 + 1.36325 w_k1_MSFT_s112
                                 + 1.16995 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.881656 w_k1_ADM_s112
                                 + 1.13752 w_k1_AXP_s112
                                 + 1.02407 w_k1_DIS_s112
                                 + 0.929635 w_k1_FDX_s112
                                 + 1.16839 w_k1_IBM_s112
                                 + 1.08882 w_k1_INTC_s112
                                 + 0.977194 w_k1_KO_s112
                                 + 0.992025 w_k1_MAT_s112
                                 + 0.848548 w_k1_MSFT_s112
                                 + 1.16503 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.09671 w_k1_ADM_s113
                                 + 1.54388 w_k1_AXP_s113
                                 + 1.14885 w_k1_DIS_s113
                                 + 1.17811 w_k1_FDX_s113
                                 + 0.768792 w_k1_IBM_s113
                                 + 1.29761 w_k1_INTC_s113
                                 + 1.02249 w_k1_KO_s113
                                 + 0.853597 w_k1_MAT_s113
                                 + 1.1971 w_k1_MSFT_s113 + 1.20347 w_k1_PG_s113
                                 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.05574 w_k1_ADM_s113
                                 + 1.03723 w_k1_AXP_s113
                                 + 1.10683 w_k1_DIS_s113
                                 + 1.02061 w_k1_FDX_s113
                                 + 1.14981 w_k1_IBM_s113
                                 + 1.33125 w_k1_INTC_s113
                                 + 1.08563 w_k1_KO_s113 + 1.12369 w_k1_MAT_s113
                                 + 1.16744 w_k1_MSFT_s113
                                 + 1.08501 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.811839 w_k1_ADM_s113
                                 + 0.982416 w_k1_AXP_s113
                                 + 1.04853 w_k1_DIS_s113
                                 + 0.846314 w_k1_FDX_s113
                                 + 1.72344 w_k1_IBM_s113
                                 + 0.835891 w_k1_INTC_s113
                                 + 1.23544 w_k1_KO_s113 + 1.28678 w_k1_MAT_s113
                                 + 0.817598 w_k1_MSFT_s113
                                 + 1.07217 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.2786 w_k1_ADM_s113
                                 + 1.39333 w_k1_AXP_s113
                                 + 1.48567 w_k1_DIS_s113
                                 + 1.27407 w_k1_FDX_s113
                                 + 1.01058 w_k1_IBM_s113
                                 + 1.2822 w_k1_INTC_s113 + 1.10465 w_k1_KO_s113
                                 + 1.06513 w_k1_MAT_s113
                                 + 1.40514 w_k1_MSFT_s113
                                 + 1.13252 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.15302 w_k1_ADM_s113
                                 + 1.05528 w_k1_AXP_s113
                                 + 0.907614 w_k1_DIS_s113
                                 + 0.874979 w_k1_FDX_s113
                                 + 1.22781 w_k1_IBM_s113
                                 + 0.877225 w_k1_INTC_s113
                                 + 0.95464 w_k1_KO_s113 + 1.03712 w_k1_MAT_s113
                                 + 0.854003 w_k1_MSFT_s113
                                 + 1.06121 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.0211 w_k1_ADM_s113
                                 + 0.951724 w_k1_AXP_s113
                                 + 0.954898 w_k1_DIS_s113
                                 + 0.994424 w_k1_FDX_s113
                                 + 0.858989 w_k1_IBM_s113
                                 + 1.00551 w_k1_INTC_s113
                                 + 0.933397 w_k1_KO_s113
                                 + 0.769805 w_k1_MAT_s113
                                 + 0.955483 w_k1_MSFT_s113
                                 + 0.932013 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.983169 w_k1_ADM_s113
                                 + 0.726974 w_k1_AXP_s113
                                 + 0.758685 w_k1_DIS_s113
                                 + 0.812329 w_k1_FDX_s113
                                 + 0.758593 w_k1_IBM_s113
                                 + 0.804654 w_k1_INTC_s113
                                 + 0.714391 w_k1_KO_s113
                                 + 0.688488 w_k1_MAT_s113
                                 + 0.788139 w_k1_MSFT_s113
                                 + 0.797693 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.15738 w_k1_ADM_s113
                                 + 1.48576 w_k1_AXP_s113
                                 + 1.51888 w_k1_DIS_s113
                                 + 1.31987 w_k1_FDX_s113 + 1.4146 w_k1_IBM_s113
                                 + 1.15604 w_k1_INTC_s113
                                 + 1.17297 w_k1_KO_s113 + 1.36332 w_k1_MAT_s113
                                 + 1.05266 w_k1_MSFT_s113
                                 + 1.14799 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.759222 w_k1_ADM_s114
                                 + 0.807029 w_k1_AXP_s114
                                 + 0.623797 w_k1_DIS_s114
                                 + 0.735811 w_k1_FDX_s114
                                 + 0.98785 w_k1_IBM_s114
                                 + 0.748868 w_k1_INTC_s114
                                 + 0.757503 w_k1_KO_s114
                                 + 0.928135 w_k1_MAT_s114
                                 + 0.927721 w_k1_MSFT_s114
                                 + 0.868073 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.852622 w_k1_ADM_s114
                                 + 1.34316 w_k1_AXP_s114
                                 + 1.07975 w_k1_DIS_s114
                                 + 1.13218 w_k1_FDX_s114
                                 + 1.10626 w_k1_IBM_s114
                                 + 1.25369 w_k1_INTC_s114
                                 + 1.07479 w_k1_KO_s114 + 1.07242 w_k1_MAT_s114
                                 + 1.27276 w_k1_MSFT_s114
                                 + 1.05341 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.00365 w_k1_ADM_s114
                                 + 1.0263 w_k1_AXP_s114
                                 + 0.950035 w_k1_DIS_s114
                                 + 0.964058 w_k1_FDX_s114
                                 + 1.58198 w_k1_IBM_s114
                                 + 0.99451 w_k1_INTC_s114
                                 + 1.35222 w_k1_KO_s114 + 1.68362 w_k1_MAT_s114
                                 + 1.02292 w_k1_MSFT_s114
                                 + 1.04126 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.01471 w_k1_ADM_s114
                                 + 1.01009 w_k1_AXP_s114
                                 + 0.89134 w_k1_DIS_s114
                                 + 1.07452 w_k1_FDX_s114
                                 + 0.671292 w_k1_IBM_s114
                                 + 0.952017 w_k1_INTC_s114
                                 + 0.837097 w_k1_KO_s114
                                 + 0.537737 w_k1_MAT_s114
                                 + 1.04029 w_k1_MSFT_s114
                                 + 0.851122 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.905863 w_k1_ADM_s114
                                 + 0.874618 w_k1_AXP_s114
                                 + 0.925106 w_k1_DIS_s114
                                 + 0.983433 w_k1_FDX_s114
                                 + 1.01871 w_k1_IBM_s114
                                 + 0.962767 w_k1_INTC_s114
                                 + 1.1258 w_k1_KO_s114 + 0.864615 w_k1_MAT_s114
                                 + 1.03992 w_k1_MSFT_s114
                                 + 1.09979 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.21997 w_k1_ADM_s114
                                 + 1.32144 w_k1_AXP_s114
                                 + 1.38627 w_k1_DIS_s114
                                 + 1.23387 w_k1_FDX_s114
                                 + 1.16899 w_k1_IBM_s114
                                 + 1.07421 w_k1_INTC_s114
                                 + 1.15042 w_k1_KO_s114 + 1.24219 w_k1_MAT_s114
                                 + 1.38977 w_k1_MSFT_s114
                                 + 1.12443 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.76102 w_k1_ADM_s114
                                 + 0.622372 w_k1_AXP_s114
                                 + 0.675828 w_k1_DIS_s114
                                 + 0.807013 w_k1_FDX_s114
                                 + 0.717815 w_k1_IBM_s114
                                 + 0.876196 w_k1_INTC_s114
                                 + 0.667534 w_k1_KO_s114
                                 + 0.393886 w_k1_MAT_s114
                                 + 0.901932 w_k1_MSFT_s114
                                 + 0.811588 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.862409 w_k1_ADM_s114
                                 + 1.00127 w_k1_AXP_s114
                                 + 0.673985 w_k1_DIS_s114
                                 + 0.856411 w_k1_FDX_s114
                                 + 0.903446 w_k1_IBM_s114
                                 + 0.848767 w_k1_INTC_s114
                                 + 0.796781 w_k1_KO_s114
                                 + 0.991431 w_k1_MAT_s114
                                 + 0.807702 w_k1_MSFT_s114
                                 + 0.889529 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.04249 w_k1_ADM_s115
                                 + 0.992405 w_k1_AXP_s115
                                 + 1.29562 w_k1_DIS_s115
                                 + 1.07895 w_k1_FDX_s115
                                 + 1.12036 w_k1_IBM_s115
                                 + 0.841924 w_k1_INTC_s115
                                 + 1.11424 w_k1_KO_s115
                                 + 0.983692 w_k1_MAT_s115
                                 + 1.08653 w_k1_MSFT_s115
                                 + 1.06479 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.75944 w_k1_ADM_s115
                                 + 1.45574 w_k1_AXP_s115
                                 + 1.81648 w_k1_DIS_s115 + 1.461 w_k1_FDX_s115
                                 + 1.08171 w_k1_IBM_s115
                                 + 1.09749 w_k1_INTC_s115
                                 + 1.40412 w_k1_KO_s115 + 1.14511 w_k1_MAT_s115
                                 + 1.39941 w_k1_MSFT_s115
                                 + 1.37919 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.974446 w_k1_ADM_s115
                                 + 0.942761 w_k1_AXP_s115
                                 + 1.02675 w_k1_DIS_s115
                                 + 0.866288 w_k1_FDX_s115
                                 + 1.51787 w_k1_IBM_s115
                                 + 0.830753 w_k1_INTC_s115
                                 + 1.2114 w_k1_KO_s115 + 1.42887 w_k1_MAT_s115
                                 + 0.925063 w_k1_MSFT_s115
                                 + 0.945036 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.23248 w_k1_ADM_s115
                                 + 1.25719 w_k1_AXP_s115 + 1.1208 w_k1_DIS_s115
                                 + 1.05984 w_k1_FDX_s115
                                 + 1.36014 w_k1_IBM_s115
                                 + 0.86949 w_k1_INTC_s115
                                 + 1.17078 w_k1_KO_s115 + 1.36117 w_k1_MAT_s115
                                 + 1.02425 w_k1_MSFT_s115
                                 + 1.07734 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.09559 w_k1_ADM_s115
                                 + 0.677812 w_k1_AXP_s115
                                 + 1.1162 w_k1_DIS_s115 + 1.03757 w_k1_FDX_s115
                                 + 0.672768 w_k1_IBM_s115
                                 + 0.834053 w_k1_INTC_s115
                                 + 0.889382 w_k1_KO_s115
                                 + 0.466566 w_k1_MAT_s115
                                 + 1.11366 w_k1_MSFT_s115
                                 + 0.964873 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.978216 w_k1_ADM_s115
                                 + 0.710736 w_k1_AXP_s115
                                 + 0.758099 w_k1_DIS_s115
                                 + 0.7969 w_k1_FDX_s115
                                 + 0.838332 w_k1_IBM_s115
                                 + 0.668 w_k1_INTC_s115 + 0.681615 w_k1_KO_s115
                                 + 0.492617 w_k1_MAT_s115
                                 + 0.779607 w_k1_MSFT_s115
                                 + 0.756948 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.0534 w_k1_ADM_s115
                                 + 1.11231 w_k1_AXP_s115
                                 + 0.905321 w_k1_DIS_s115
                                 + 0.992758 w_k1_FDX_s115
                                 + 0.91577 w_k1_IBM_s115
                                 + 0.736174 w_k1_INTC_s115
                                 + 0.923338 w_k1_KO_s115
                                 + 1.15664 w_k1_MAT_s115
                                 + 0.949427 w_k1_MSFT_s115
                                 + 0.999136 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.36842 w_k1_ADM_s115
                                 + 1.53061 w_k1_AXP_s115
                                 + 1.28374 w_k1_DIS_s115
                                 + 1.23815 w_k1_FDX_s115
                                 + 0.955626 w_k1_IBM_s115
                                 + 1.09535 w_k1_INTC_s115
                                 + 1.07673 w_k1_KO_s115 + 1.08205 w_k1_MAT_s115
                                 + 1.11509 w_k1_MSFT_s115 + 1.039 w_k1_PG_s115
                                 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.870852 w_k1_ADM_s116
                                 + 0.838601 w_k1_AXP_s116
                                 + 0.919574 w_k1_DIS_s116
                                 + 0.799037 w_k1_FDX_s116
                                 + 1.08623 w_k1_IBM_s116
                                 + 0.751784 w_k1_INTC_s116
                                 + 0.920402 w_k1_KO_s116
                                 + 1.03843 w_k1_MAT_s116
                                 + 0.90293 w_k1_MSFT_s116
                                 + 0.85833 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.25002 w_k1_ADM_s116
                                 + 1.16992 w_k1_AXP_s116
                                 + 1.12926 w_k1_DIS_s116
                                 + 1.14557 w_k1_FDX_s116
                                 + 0.969068 w_k1_IBM_s116
                                 + 1.10293 w_k1_INTC_s116 + 1.1048 w_k1_KO_s116
                                 + 0.772216 w_k1_MAT_s116
                                 + 1.17795 w_k1_MSFT_s116
                                 + 1.08724 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.728581 w_k1_ADM_s116
                                 + 0.79452 w_k1_AXP_s116
                                 + 0.687392 w_k1_DIS_s116
                                 + 0.809014 w_k1_FDX_s116
                                 + 0.766129 w_k1_IBM_s116
                                 + 0.909615 w_k1_INTC_s116
                                 + 0.715467 w_k1_KO_s116
                                 + 0.678932 w_k1_MAT_s116
                                 + 0.780696 w_k1_MSFT_s116
                                 + 0.846902 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.00961 w_k1_ADM_s116
                                 + 1.16672 w_k1_AXP_s116
                                 + 0.903083 w_k1_DIS_s116
                                 + 1.13651 w_k1_FDX_s116
                                 + 0.593078 w_k1_IBM_s116
                                 + 1.05439 w_k1_INTC_s116
                                 + 0.727431 w_k1_KO_s116
                                 + 0.502314 w_k1_MAT_s116
                                 + 0.897644 w_k1_MSFT_s116
                                 + 0.968971 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.905424 w_k1_ADM_s116
                                 + 0.979126 w_k1_AXP_s116
                                 + 0.949867 w_k1_DIS_s116
                                 + 1.0122 w_k1_FDX_s116 + 0.88076 w_k1_IBM_s116
                                 + 1.00897 w_k1_INTC_s116
                                 + 0.894392 w_k1_KO_s116
                                 + 0.847385 w_k1_MAT_s116
                                 + 0.9636 w_k1_MSFT_s116 + 1.09153 w_k1_PG_s116
                                 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.995869 w_k1_ADM_s116
                                 + 1.2126 w_k1_AXP_s116 + 1.00378 w_k1_DIS_s116
                                 + 0.946211 w_k1_FDX_s116
                                 + 1.33877 w_k1_IBM_s116
                                 + 1.02393 w_k1_INTC_s116
                                 + 1.01775 w_k1_KO_s116 + 1.36717 w_k1_MAT_s116
                                 + 0.894885 w_k1_MSFT_s116
                                 + 1.06929 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.16403 w_k1_ADM_s116
                                 + 1.21622 w_k1_AXP_s116 + 1.1834 w_k1_DIS_s116
                                 + 1.16043 w_k1_FDX_s116
                                 + 0.755046 w_k1_IBM_s116
                                 + 1.08719 w_k1_INTC_s116
                                 + 0.908125 w_k1_KO_s116
                                 + 0.722659 w_k1_MAT_s116
                                 + 1.07644 w_k1_MSFT_s116
                                 + 1.04368 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.04185 w_k1_ADM_s116
                                 + 1.54852 w_k1_AXP_s116
                                 + 1.59492 w_k1_DIS_s116
                                 + 1.28276 w_k1_FDX_s116
                                 + 1.41944 w_k1_IBM_s116
                                 + 1.48004 w_k1_INTC_s116
                                 + 1.29356 w_k1_KO_s116 + 1.3211 w_k1_MAT_s116
                                 + 1.40582 w_k1_MSFT_s116
                                 + 1.25349 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.815464 w_k1_ADM_s117
                                 + 0.75012 w_k1_AXP_s117
                                 + 0.622119 w_k1_DIS_s117
                                 + 0.746373 w_k1_FDX_s117
                                 + 0.708458 w_k1_IBM_s117
                                 + 0.83706 w_k1_INTC_s117
                                 + 0.634423 w_k1_KO_s117
                                 + 0.744818 w_k1_MAT_s117
                                 + 0.722239 w_k1_MSFT_s117
                                 + 0.774144 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.973222 w_k1_ADM_s117
                                 + 0.740432 w_k1_AXP_s117
                                 + 1.14294 w_k1_DIS_s117
                                 + 0.908169 w_k1_FDX_s117
                                 + 1.06206 w_k1_IBM_s117
                                 + 0.988407 w_k1_INTC_s117
                                 + 1.11308 w_k1_KO_s117 + 1.02794 w_k1_MAT_s117
                                 + 1.07638 w_k1_MSFT_s117
                                 + 1.03253 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.863939 w_k1_ADM_s117
                                 + 0.892277 w_k1_AXP_s117
                                 + 0.824907 w_k1_DIS_s117
                                 + 0.7882 w_k1_FDX_s117 + 1.18617 w_k1_IBM_s117
                                 + 0.849582 w_k1_INTC_s117
                                 + 0.899474 w_k1_KO_s117
                                 + 1.14791 w_k1_MAT_s117
                                 + 0.962252 w_k1_MSFT_s117
                                 + 0.856781 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.02033 w_k1_ADM_s117
                                 + 1.40673 w_k1_AXP_s117
                                 + 1.48014 w_k1_DIS_s117
                                 + 1.16859 w_k1_FDX_s117
                                 + 1.20325 w_k1_IBM_s117
                                 + 1.49477 w_k1_INTC_s117
                                 + 1.25014 w_k1_KO_s117 + 1.21532 w_k1_MAT_s117
                                 + 1.24248 w_k1_MSFT_s117
                                 + 1.09474 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.05514 w_k1_ADM_s117
                                 + 1.33362 w_k1_AXP_s117
                                 + 1.42947 w_k1_DIS_s117
                                 + 1.16201 w_k1_FDX_s117
                                 + 1.71095 w_k1_IBM_s117
                                 + 1.12783 w_k1_INTC_s117
                                 + 1.28887 w_k1_KO_s117 + 1.97386 w_k1_MAT_s117
                                 + 1.08978 w_k1_MSFT_s117
                                 + 1.00783 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.956117 w_k1_ADM_s117
                                 + 1.25478 w_k1_AXP_s117
                                 + 1.11408 w_k1_DIS_s117
                                 + 1.10876 w_k1_FDX_s117
                                 + 0.784164 w_k1_IBM_s117
                                 + 1.0015 w_k1_INTC_s117
                                 + 0.957528 w_k1_KO_s117
                                 + 1.10928 w_k1_MAT_s117
                                 + 1.09504 w_k1_MSFT_s117
                                 + 1.05095 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.794786 w_k1_ADM_s117
                                 + 0.75637 w_k1_AXP_s117
                                 + 0.831621 w_k1_DIS_s117
                                 + 0.928285 w_k1_FDX_s117
                                 + 0.742558 w_k1_IBM_s117
                                 + 0.882949 w_k1_INTC_s117
                                 + 0.837811 w_k1_KO_s117
                                 + 0.684834 w_k1_MAT_s117
                                 + 0.992224 w_k1_MSFT_s117
                                 + 0.858083 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.976048 w_k1_ADM_s117
                                 + 0.961443 w_k1_AXP_s117
                                 + 1.00765 w_k1_DIS_s117
                                 + 0.933344 w_k1_FDX_s117
                                 + 0.970895 w_k1_IBM_s117
                                 + 0.967912 w_k1_INTC_s117
                                 + 0.948366 w_k1_KO_s117
                                 + 1.12019 w_k1_MAT_s117
                                 + 0.953276 w_k1_MSFT_s117
                                 + 0.93979 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.043873482294955 w_k1_ADM_s000
                                 + 0.0259271559872316 w_k1_ADM_s001
                                 + 0.0356238710847426 w_k1_ADM_s002
                                 + 0.0488733983369309 w_k1_ADM_s003
                                 + 0.0420173358567834 w_k1_ADM_s004
                                 + 0.0182942975674885 w_k1_ADM_s005
                                 + 0.0520604044440569 w_k1_ADM_s006
                                 + 0.0445560356185024 w_k1_ADM_s007
                                 + 0.0150901024143404 w_k1_ADM_s010
                                 + 0.0161930329638967 w_k1_ADM_s011
                                 + 0.0169744939922013 w_k1_ADM_s012
                                 + 0.0209240182902562 w_k1_ADM_s013
                                 + 0.0170535095896087 w_k1_ADM_s014
                                 + 0.00477196232531599 w_k1_ADM_s015
                                 + 0.00806977740226732 w_k1_ADM_s016
                                 + 0.00537952321662429 w_k1_ADM_s017
                                 + 0.0555809272598718 w_k1_ADM_s100
                                 + 0.0142191431907444 w_k1_ADM_s101
                                 + 0.0109690644151162 w_k1_ADM_s102
                                 + 0.0221755815850043 w_k1_ADM_s103
                                 + 0.0226519000149319 w_k1_ADM_s104
                                 + 0.0443350887574981 w_k1_ADM_s105
                                 + 0.0261587461925413 w_k1_ADM_s106
                                 + 0.0248301256673167 w_k1_ADM_s107
                                 + 0.0283632322768276 w_k1_ADM_s110
                                 + 0.0499768595954326 w_k1_ADM_s111
                                 + 0.0377576795911011 w_k1_ADM_s112
                                 + 0.0480289575265493 w_k1_ADM_s113
                                 + 0.0724588563951829 w_k1_ADM_s114
                                 + 0.044456041405626 w_k1_ADM_s115
                                 + 0.0521602909175882 w_k1_ADM_s116
                                 + 0.0458407956061856 w_k1_ADM_s117
                                 + 0.0415833890422759 w_k1_AXP_s000
                                 + 0.0258063362027902 w_k1_AXP_s001
                                 + 0.0314948737536583 w_k1_AXP_s002
                                 + 0.0580771794827758 w_k1_AXP_s003
                                 + 0.0367319922095921 w_k1_AXP_s004
                                 + 0.0187063229236658 w_k1_AXP_s005
                                 + 0.0530859726885274 w_k1_AXP_s006
                                 + 0.0459113976307032 w_k1_AXP_s007
                                 + 0.0168282842359634 w_k1_AXP_s010
                                 + 0.0171465015127347 w_k1_AXP_s011
                                 + 0.0165926209987648 w_k1_AXP_s012
                                 + 0.0202152374370936 w_k1_AXP_s013
                                 + 0.0182817385762541 w_k1_AXP_s014
                                 + 0.00448638442447988 w_k1_AXP_s015
                                 + 0.00868717652628692 w_k1_AXP_s016
                                 + 0.0049262997536437 w_k1_AXP_s017
                                 + 0.0581066261639985 w_k1_AXP_s100
                                 + 0.0145358500169068 w_k1_AXP_s101
                                 + 0.0117983010931689 w_k1_AXP_s102
                                 + 0.0251410385396393 w_k1_AXP_s103
                                 + 0.0223547507926909 w_k1_AXP_s104
                                 + 0.0426732943952989 w_k1_AXP_s105
                                 + 0.0253208999931143 w_k1_AXP_s106
                                 + 0.0256646716914979 w_k1_AXP_s107
                                 + 0.0254386519509639 w_k1_AXP_s110
                                 + 0.0489524296734445 w_k1_AXP_s111
                                 + 0.041551270156392 w_k1_AXP_s112
                                 + 0.0484451371240777 w_k1_AXP_s113
                                 + 0.0754815103173798 w_k1_AXP_s114
                                 + 0.0406498684651893 w_k1_AXP_s115
                                 + 0.0605196464365245 w_k1_AXP_s116
                                 + 0.0520103731467635 w_k1_AXP_s117
                                 + 0.0411840766330664 w_k1_DIS_s000
                                 + 0.0301518179744768 w_k1_DIS_s001
                                 + 0.0331459521196276 w_k1_DIS_s002
                                 + 0.0639839086459269 w_k1_DIS_s003
                                 + 0.0377955380624477 w_k1_DIS_s004
                                 + 0.0167216001844362 w_k1_DIS_s005
                                 + 0.0517070507149745 w_k1_DIS_s006
                                 + 0.0418189171537845 w_k1_DIS_s007
                                 + 0.018035386322687 w_k1_DIS_s010
                                 + 0.0168144955212099 w_k1_DIS_s011
                                 + 0.0167712762665951 w_k1_DIS_s012
                                 + 0.021662818626219 w_k1_DIS_s013
                                 + 0.0213393917457342 w_k1_DIS_s014
                                 + 0.00436017820297137 w_k1_DIS_s015
                                 + 0.00790328902229472 w_k1_DIS_s016
                                 + 0.00499092602347336 w_k1_DIS_s017
                                 + 0.0546242442555587 w_k1_DIS_s100
                                 + 0.0157771172837857 w_k1_DIS_s101
                                 + 0.0118839847528473 w_k1_DIS_s102
                                 + 0.0259501451954112 w_k1_DIS_s103
                                 + 0.0205462594502536 w_k1_DIS_s104
                                 + 0.0445748398105887 w_k1_DIS_s105
                                 + 0.024084467904619 w_k1_DIS_s106
                                 + 0.0281663849065739 w_k1_DIS_s107
                                 + 0.025582423123958 w_k1_DIS_s110
                                 + 0.0504457923409787 w_k1_DIS_s111
                                 + 0.040255491318704 w_k1_DIS_s112
                                 + 0.047711419157306 w_k1_DIS_s113
                                 + 0.0723807627345868 w_k1_DIS_s114
                                 + 0.0423825417886257 w_k1_DIS_s115
                                 + 0.056735848438479 w_k1_DIS_s116
                                 + 0.0535556088402225 w_k1_DIS_s117
                                 + 0.0383032945653902 w_k1_FDX_s000
                                 + 0.0275486568617939 w_k1_FDX_s001
                                 + 0.0327499541641345 w_k1_FDX_s002
                                 + 0.0588311916408951 w_k1_FDX_s003
                                 + 0.0417326517433727 w_k1_FDX_s004
                                 + 0.0175838590398054 w_k1_FDX_s005
                                 + 0.0511415071048121 w_k1_FDX_s006
                                 + 0.04102617689066 w_k1_FDX_s007
                                 + 0.017008131113065 w_k1_FDX_s010
                                 + 0.0168955244597985 w_k1_FDX_s011
                                 + 0.0161009836076591 w_k1_FDX_s012
                                 + 0.0211707072184382 w_k1_FDX_s013
                                 + 0.02059826296202 w_k1_FDX_s014
                                 + 0.00454192889634334 w_k1_FDX_s015
                                 + 0.00773068059226885 w_k1_FDX_s016
                                 + 0.00492862728507572 w_k1_FDX_s017
                                 + 0.0557781948230542 w_k1_FDX_s100
                                 + 0.0142165034325523 w_k1_FDX_s101
                                 + 0.0113581803995167 w_k1_FDX_s102
                                 + 0.0228520960247353 w_k1_FDX_s103
                                 + 0.0215983353102917 w_k1_FDX_s104
                                 + 0.0406960041265717 w_k1_FDX_s105
                                 + 0.0246046846179645 w_k1_FDX_s106
                                 + 0.0261306307908442 w_k1_FDX_s107
                                 + 0.0266578541011799 w_k1_FDX_s110
                                 + 0.0468936414992054 w_k1_FDX_s111
                                 + 0.0391559869588185 w_k1_FDX_s112
                                 + 0.0453088962102924 w_k1_FDX_s113
                                 + 0.0756459967869621 w_k1_FDX_s114
                                 + 0.0397720142842924 w_k1_FDX_s115
                                 + 0.054835430084997 w_k1_FDX_s116
                                 + 0.0483149928417885 w_k1_FDX_s117
                                 + 0.0417331610358015 w_k1_IBM_s000
                                 + 0.027880827664659 w_k1_IBM_s001
                                 + 0.0309034350888736 w_k1_IBM_s002
                                 + 0.0593292602236918 w_k1_IBM_s003
                                 + 0.0396010923794858 w_k1_IBM_s004
                                 + 0.0174617377378013 w_k1_IBM_s005
                                 + 0.0484337594388001 w_k1_IBM_s006
                                 + 0.0412256603083345 w_k1_IBM_s007
                                 + 0.0170804347400845 w_k1_IBM_s010
                                 + 0.0180292418816079 w_k1_IBM_s011
                                 + 0.0153864789216915 w_k1_IBM_s012
                                 + 0.0207345134191488 w_k1_IBM_s013
                                 + 0.0207314465462821 w_k1_IBM_s014
                                 + 0.00458103902606123 w_k1_IBM_s015
                                 + 0.00778289265425642 w_k1_IBM_s016
                                 + 0.00504633224633273 w_k1_IBM_s017
                                 + 0.058856083047921 w_k1_IBM_s100
                                 + 0.0141086832167948 w_k1_IBM_s101
                                 + 0.0108382529096541 w_k1_IBM_s102
                                 + 0.0242242702106103 w_k1_IBM_s103
                                 + 0.0208201679139658 w_k1_IBM_s104
                                 + 0.0430527568818132 w_k1_IBM_s105
                                 + 0.0255473231221289 w_k1_IBM_s106
                                 + 0.0260226068633973 w_k1_IBM_s107
                                 + 0.0262847887473877 w_k1_IBM_s110
                                 + 0.0502732402054017 w_k1_IBM_s111
                                 + 0.0382200608045142 w_k1_IBM_s112
                                 + 0.0465487803897531 w_k1_IBM_s113
                                 + 0.078263459826037 w_k1_IBM_s114
                                 + 0.0422588996341872 w_k1_IBM_s115
                                 + 0.0574103267943344 w_k1_IBM_s116
                                 + 0.0489349011589097 w_k1_IBM_s117
                                 + 0.0389911937847459 w_k1_INTC_s000
                                 + 0.0307498936453798 w_k1_INTC_s001
                                 + 0.0343892771712002 w_k1_INTC_s002
                                 + 0.0629967019381706 w_k1_INTC_s003
                                 + 0.0449839293939359 w_k1_INTC_s004
                                 + 0.0199442370257971 w_k1_INTC_s005
                                 + 0.0515203918500681 w_k1_INTC_s006
                                 + 0.0444648445724742 w_k1_INTC_s007
                                 + 0.0151573716455178 w_k1_INTC_s010
                                 + 0.0194590325872438 w_k1_INTC_s011
                                 + 0.0168714085099392 w_k1_INTC_s012
                                 + 0.0195133048433639 w_k1_INTC_s013
                                 + 0.0196478000148823 w_k1_INTC_s014
                                 + 0.00438812937800535 w_k1_INTC_s015
                                 + 0.00749365433191488 w_k1_INTC_s016
                                 + 0.0051202544861192 w_k1_INTC_s017
                                 + 0.060600232271962 w_k1_INTC_s100
                                 + 0.0152261822688334 w_k1_INTC_s101
                                 + 0.0102101553349923 w_k1_INTC_s102
                                 + 0.0210540749528339 w_k1_INTC_s103
                                 + 0.0208592906639013 w_k1_INTC_s104
                                 + 0.0356606875399294 w_k1_INTC_s105
                                 + 0.0213779429951419 w_k1_INTC_s106
                                 + 0.0281101300692475 w_k1_INTC_s107
                                 + 0.0241113258779344 w_k1_INTC_s110
                                 + 0.0514445631271838 w_k1_INTC_s111
                                 + 0.0441929338395795 w_k1_INTC_s112
                                 + 0.0463734732584834 w_k1_INTC_s113
                                 + 0.0729933243231429 w_k1_INTC_s114
                                 + 0.032572950846507 w_k1_INTC_s115
                                 + 0.0577697628538807 w_k1_INTC_s116
                                 + 0.0523092925572563 w_k1_INTC_s117
                                 + 0.0381745854066578 w_k1_KO_s000
                                 + 0.027665864420956 w_k1_KO_s001
                                 + 0.0312504742276967 w_k1_KO_s002
                                 + 0.0630765336968124 w_k1_KO_s003
                                 + 0.0398401923973486 w_k1_KO_s004
                                 + 0.0172625839991817 w_k1_KO_s005
                                 + 0.0475912033948678 w_k1_KO_s006
                                 + 0.0396682426928236 w_k1_KO_s007
                                 + 0.0173911786184558 w_k1_KO_s010
                                 + 0.0167911006160396 w_k1_KO_s011
                                 + 0.0169199395689473 w_k1_KO_s012
                                 + 0.0198357452700619 w_k1_KO_s013
                                 + 0.0197187598064596 w_k1_KO_s014
                                 + 0.00444614467345053 w_k1_KO_s015
                                 + 0.00708068371400204 w_k1_KO_s016
                                 + 0.00516867059382929 w_k1_KO_s017
                                 + 0.0524920352270312 w_k1_KO_s100
                                 + 0.0156905985613616 w_k1_KO_s101
                                 + 0.0110387015331112 w_k1_KO_s102
                                 + 0.0241851805601761 w_k1_KO_s103
                                 + 0.0201367820285879 w_k1_KO_s104
                                 + 0.0418299382933141 w_k1_KO_s105
                                 + 0.0259309923974546 w_k1_KO_s106
                                 + 0.0298856524517939 w_k1_KO_s107
                                 + 0.0242014616932741 w_k1_KO_s110
                                 + 0.0482415536106559 w_k1_KO_s111
                                 + 0.0406355217730599 w_k1_KO_s112
                                 + 0.0435281072827649 w_k1_KO_s113
                                 + 0.0769887737422339 w_k1_KO_s114
                                 + 0.0399729933519271 w_k1_KO_s115
                                 + 0.0526443169721809 w_k1_KO_s116
                                 + 0.0488239558682444 w_k1_KO_s117
                                 + 0.0556842277450924 w_k1_MAT_s000
                                 + 0.021433542464443 w_k1_MAT_s001
                                 + 0.0292354732179354 w_k1_MAT_s002
                                 + 0.0666249670679755 w_k1_MAT_s003
                                 + 0.0380610860192206 w_k1_MAT_s004
                                 + 0.0203408205999338 w_k1_MAT_s005
                                 + 0.0468422002411826 w_k1_MAT_s006
                                 + 0.036067988646903 w_k1_MAT_s007
                                 + 0.0188158683788566 w_k1_MAT_s010
                                 + 0.0144980390107558 w_k1_MAT_s011
                                 + 0.0188030912832772 w_k1_MAT_s012
                                 + 0.0298668932261725 w_k1_MAT_s013
                                 + 0.0217283586591128 w_k1_MAT_s014
                                 + 0.00529153367849223 w_k1_MAT_s015
                                 + 0.00664236127669623 w_k1_MAT_s016
                                 + 0.00419736904525797 w_k1_MAT_s017
                                 + 0.0577400474921653 w_k1_MAT_s100
                                 + 0.0132247693585489 w_k1_MAT_s101
                                 + 0.00993843439948391 w_k1_MAT_s102
                                 + 0.0268510714180051 w_k1_MAT_s103
                                 + 0.0211865473760137 w_k1_MAT_s104
                                 + 0.0409212630853811 w_k1_MAT_s105
                                 + 0.0270143697791021 w_k1_MAT_s106
                                 + 0.0319820997837295 w_k1_MAT_s107
                                 + 0.028181070253941 w_k1_MAT_s110
                                 + 0.0521385209006096 w_k1_MAT_s111
                                 + 0.037630496481561 w_k1_MAT_s112
                                 + 0.0430875668162919 w_k1_MAT_s113
                                 + 0.0737193791824173 w_k1_MAT_s114
                                 + 0.0396749414739369 w_k1_MAT_s115
                                 + 0.0540215377123119 w_k1_MAT_s116
                                 + 0.0536109549332945 w_k1_MAT_s117
                                 + 0.0421874295980534 w_k1_MSFT_s000
                                 + 0.0280336732595162 w_k1_MSFT_s001
                                 + 0.0351328005733895 w_k1_MSFT_s002
                                 + 0.0614845528593479 w_k1_MSFT_s003
                                 + 0.04003569088699 w_k1_MSFT_s004
                                 + 0.0182614031784931 w_k1_MSFT_s005
                                 + 0.0491865138011082 w_k1_MSFT_s006
                                 + 0.0443071433096645 w_k1_MSFT_s007
                                 + 0.0161895932284823 w_k1_MSFT_s010
                                 + 0.0156571280129914 w_k1_MSFT_s011
                                 + 0.0154853779581882 w_k1_MSFT_s012
                                 + 0.0207116686661698 w_k1_MSFT_s013
                                 + 0.0199702368165269 w_k1_MSFT_s014
                                 + 0.00460990035512754 w_k1_MSFT_s015
                                 + 0.00700214593745221 w_k1_MSFT_s016
                                 + 0.00463762134275126 w_k1_MSFT_s017
                                 + 0.0552957928327786 w_k1_MSFT_s100
                                 + 0.0137030023622647 w_k1_MSFT_s101
                                 + 0.0102780148265545 w_k1_MSFT_s102
                                 + 0.0212125900764827 w_k1_MSFT_s103
                                 + 0.022017979784755 w_k1_MSFT_s104
                                 + 0.0377063114393727 w_k1_MSFT_s105
                                 + 0.0264382503200094 w_k1_MSFT_s106
                                 + 0.027264346332578 w_k1_MSFT_s107
                                 + 0.0260522351292114 w_k1_MSFT_s110
                                 + 0.0484579610631339 w_k1_MSFT_s111
                                 + 0.0391440335050789 w_k1_MSFT_s112
                                 + 0.0447263365414731 w_k1_MSFT_s113
                                 + 0.0814771151421829 w_k1_MSFT_s114
                                 + 0.0392188172645447 w_k1_MSFT_s115
                                 + 0.0547078243020492 w_k1_MSFT_s116
                                 + 0.0503919738354525 w_k1_MSFT_s117
                                 + 0.0397566081032355 w_k1_PG_s000
                                 + 0.0272805388980741 w_k1_PG_s001
                                 + 0.0331239758502999 w_k1_PG_s002
                                 + 0.0580673540486471 w_k1_PG_s003
                                 + 0.039523730747751 w_k1_PG_s004
                                 + 0.0174407683479848 w_k1_PG_s005
                                 + 0.0523286332028128 w_k1_PG_s006
                                 + 0.0385685686873962 w_k1_PG_s007
                                 + 0.0172695796516094 w_k1_PG_s010
                                 + 0.0160959969129838 w_k1_PG_s011
                                 + 0.0176180626459055 w_k1_PG_s012
                                 + 0.0211413460891295 w_k1_PG_s013
                                 + 0.0207967681747677 w_k1_PG_s014
                                 + 0.00497616762765932 w_k1_PG_s015
                                 + 0.0082553552100406 w_k1_PG_s016
                                 + 0.00462669626509524 w_k1_PG_s017
                                 + 0.0560117253852413 w_k1_PG_s100
                                 + 0.0143915651806325 w_k1_PG_s101
                                 + 0.0119862790183696 w_k1_PG_s102
                                 + 0.0231962705085043 w_k1_PG_s103
                                 + 0.0214795558386718 w_k1_PG_s104
                                 + 0.0423862812643094 w_k1_PG_s105
                                 + 0.0255891762642844 w_k1_PG_s106
                                 + 0.0267171802277537 w_k1_PG_s107
                                 + 0.0257635682286004 w_k1_PG_s110
                                 + 0.0448502188979492 w_k1_PG_s111
                                 + 0.0401168735538915 w_k1_PG_s112
                                 + 0.0452028519827404 w_k1_PG_s113
                                 + 0.0750313630550678 w_k1_PG_s114
                                 + 0.0389798327561546 w_k1_PG_s115
                                 + 0.0561386784919529 w_k1_PG_s116
                                 + 0.0472378120568974 w_k1_PG_s117 >= 1.09
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
