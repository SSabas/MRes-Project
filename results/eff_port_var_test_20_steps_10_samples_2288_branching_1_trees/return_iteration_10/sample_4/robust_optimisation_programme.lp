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
 w_k1_ADM_s0_return_constraint:  1.18177 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.945786 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.730708 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.14548 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.92519 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.03707 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.25743 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.935882 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.1345 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.39992 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.31393 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.04433 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.79252 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.3619 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.1311 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.799532 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.760442 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.00686 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.663253 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.857585 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.817686 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.854648 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.17152 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.885611 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.79204 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.804888 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.04941 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.919066 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.734303 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.917044 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.21017 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.52109 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.00142 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.789625 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.811665 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.910515 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.10955 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.06895 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.34942 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.888582 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.693412 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.11983 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.90771 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.12446 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.04827 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.681795 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.29919 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.18066 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.05211 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.04875 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.768651 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.29928 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.08578 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.76656 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.614286 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.2152 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.732579 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.34225 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.690166 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.823699 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.32208 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.820665 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.692009 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.758159 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.16835 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.0557 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.61185 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.955593 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.55232 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.56587 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.920047 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.683128 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.596203 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.881764 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.22507 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.15396 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.38315 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.898376 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.659004 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.14639 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.852221 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.15733 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.988743 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.720544 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.41844 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.2569 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.23618 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.954094 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.705428 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.57671 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.25291 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.838591 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.629135 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.10905 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.554659 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.30213 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.660989 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.866486 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.57166 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.842129 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.559679 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.804415 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.24163 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.922709 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.548183 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.949942 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.71529 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.60347 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.80994 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.531616 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.713056 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.870112 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.11484 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.38297 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.21234 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.930107 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.751248 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.07839 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.857056 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.13591 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.08455 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.912756 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.17243 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.04732 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.19047 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.9849 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.76425 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.4093 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.27501 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.865276 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.865515 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.15286 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.777575 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.10181 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.760003 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.919008 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.24865 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.882523 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.798741 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.834393 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.15775 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.93914 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.723574 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.07479 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.29675 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.25445 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.951312 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.7066 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.833056 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.852539 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.1627 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.22998 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.18207 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.972659 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.928232 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.08672 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.15291 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.906766 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.84917 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.565687 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.45011 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 2.01021 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.702453 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.977869 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.20266 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.08292 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.720916 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.196 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.471043 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.977749 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.693308 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.70296 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.740384 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.898469 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.58042 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.889242 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.433394 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.746913 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.959431 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.22167 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.585743 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.62615 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.58661 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 2.05953 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.72963 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.758378 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.540945 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.11584 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.15543 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.936117 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.16282 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.918395 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.772504 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.01992 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.953206 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.05795 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.943959 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.95537 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.10215 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.18965 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.996083 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.0145 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.963017 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.31559 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.10676 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.06898 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.919301 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.12204 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.949975 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.02526 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.796008 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.900463 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.995712 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.949191 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.736829 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.897021 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.07667 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.00591 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.784903 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.0137 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.43981 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.21756 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.980253 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.805641 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.875676 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.10256 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.13208 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.16306 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.24376 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.957366 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.821887 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.0962 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.967376 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.05445 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.787343 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.63513 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.30328 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.5289 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.943368 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.05052 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.970167 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.22326 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.902001 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.0069 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.69252 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.02882 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.681146 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.27801 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.718313 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.844661 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.41641 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.910343 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.486708 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.822589 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.08276 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.986783 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.593294 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.856146 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.42331 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.48474 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.78384 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.687782 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.674575 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.942223 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.03448 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.05479 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.19237 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.934189 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.916811 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.28912 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.1179 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.06352 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.743379 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.471236 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.73018 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 2.0776 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.639181 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.0334 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.12364 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.06767 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.721626 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.930062 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.443656 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.90756 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.714464 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.59223 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.636645 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.785027 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.71864 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.842649 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.360919 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.704342 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.876042 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.27378 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.442377 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.632402 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.47803 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.87263 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.713857 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.730694 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.417162 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.898631 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.04585 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.07272 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.19163 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.9304 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.797659 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.04268 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.923702 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.12717 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.929819 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.865057 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.16276 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.03996 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.04669 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.02515 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.803009 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.46036 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.1556 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.890852 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.874522 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.08265 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.818324 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.1836 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.783277 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.911265 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.10531 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.852243 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.65857 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.877952 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.05158 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.884677 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.743027 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.930737 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.26495 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.12892 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.975668 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.696741 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.807009 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.90981 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.02756 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.28928 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.10279 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.953813 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.851729 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.10459 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.907408 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.10076 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.939258 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.884949 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.31221 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.19367 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.06844 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.0826 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.920643 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.23003 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.07351 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.896087 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.837994 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.04891 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.85009 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.08639 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.778476 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.909257 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.29365 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.97531 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.784454 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.928064 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.16217 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.03644 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.776978 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.982883 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.18847 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.22833 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.916166 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.766818 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.73409 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.965346 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.973754 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.07204 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.0610000052454257 z_k1_s0000
                                 - 0.106000068328066 z_k1_s0001
                                 - 0.0559999781695942 z_k1_s0002
                                 - 0.0379999529365382 z_k1_s0003
                                 - 0.0349999487310289 z_k1_s0004
                                 - 0.105000147192826 z_k1_s0005
                                 - 0.132999744977962 z_k1_s0006
                                 - 0.0680000551915794 z_k1_s0007
                                 - 0.320999628850701 z_k1_s0010
                                 - 0.102000032457227 z_k1_s0011
                                 - 0.200999503018672 z_k1_s0012
                                 - 0.236999391768286 z_k1_s0013
                                 - 0.218000402428179 z_k1_s0014
                                 - 0.134000016345769 z_k1_s0015
                                 - 0.156000014581642 z_k1_s0016
                                 - 0.121000064797298 z_k1_s0017
                                 - 0.107000147760005 z_k1_s0020
                                 - 0.0459999886472956 z_k1_s0021
                                 - 0.0489999802003805 z_k1_s0022
                                 - 0.0670000004188862 z_k1_s0023
                                 - 0.080999701033295 z_k1_s0024
                                 - 0.0439999706452402 z_k1_s0025
                                 - 0.144999709899096 z_k1_s0026
                                 - 0.169999757674798 z_k1_s0027
                                 - 0.140999839854229 z_k1_s0030
                                 - 0.0510000205600313 z_k1_s0031
                                 - 0.0479999825035631 z_k1_s0032
                                 - 0.0409999633051293 z_k1_s0033
                                 - 0.145999903281676 z_k1_s0034
                                 - 0.150999966709123 z_k1_s0035
                                 - 0.0460000267325763 z_k1_s0036
                                 - 0.0719997997553646 z_k1_s0037
                                 - 0.0500000118630023 z_k1_s0040
                                 - 0.0730000938200398 z_k1_s0041
                                 - 0.136999701004382 z_k1_s0042
                                 - 0.0400000094904019 z_k1_s0043
                                 - 0.0840000369298565 z_k1_s0044
                                 - 0.136999701004382 z_k1_s0045
                                 - 0.159000267224517 z_k1_s0046
                                 - 0.170000125334271 z_k1_s0047
                                 - 0.0940000166868161 z_k1_s0050
                                 - 0.361000224801448 z_k1_s0051
                                 - 0.199999740610335 z_k1_s0052
                                 - 0.237000385623041 z_k1_s0053
                                 - 0.091000020577661 z_k1_s0054
                                 - 0.11399999074785 z_k1_s0055
                                 - 0.102000006311229 z_k1_s0056
                                 - 0.187000450470457 z_k1_s0057
                                 - 0.0990002589410316 z_k1_s0060
                                 - 0.0910002557336959 z_k1_s0061
                                 - 0.19500029742906 z_k1_s0062
                                 - 0.142000495430713 z_k1_s0063
                                 - 0.138000493827046 z_k1_s0064
                                 - 0.0640001133587865 z_k1_s0065
                                 - 0.102000479394035 z_k1_s0066
                                 - 0.0460000184421803 z_k1_s0067
                                 - 0.289001265646967 z_k1_s0070
                                 - 0.0820003307405547 z_k1_s0071
                                 - 0.121000388055807 z_k1_s0072
                                 - 0.068000212057786 z_k1_s0073
                                 - 0.132000754607586 z_k1_s0074
                                 - 0.076000279876511 z_k1_s0075
                                 - 0.0800003137858735 z_k1_s0076
                                 - 0.063000260771427 z_k1_s0077
                                 - 0.403000499632675 z_k1_s0100
                                 - 0.34000058968087 z_k1_s0101
                                 - 0.60299981209665 z_k1_s0102
                                 - 0.360000220696907 z_k1_s0103
                                 - 0.417000670244416 z_k1_s0104
                                 - 1.26300050264132 z_k1_s0105
                                 - 0.558998908259309 z_k1_s0106
                                 - 0.344000515884078 z_k1_s0107
                                 - 0.711998423027823 z_k1_s0110
                                 - 0.225999555805943 z_k1_s0111
                                 - 0.661997728602848 z_k1_s0112
                                 - 0.479999500489872 z_k1_s0113
                                 - 0.175999577979957 z_k1_s0114
                                 - 0.258999655826932 z_k1_s0115
                                 - 0.686999867312807 z_k1_s0116
                                 - 0.380997767228927 z_k1_s0117
                                 - 0.189999828810944 z_k1_s0120
                                 - 0.207000101120384 z_k1_s0121
                                 - 0.139999974918604 z_k1_s0122
                                 - 0.139000045782767 z_k1_s0123
                                 - 0.0790000143319305 z_k1_s0124
                                 - 0.208000030256221 z_k1_s0125
                                 - 0.344000731731535 z_k1_s0126
                                 - 0.169999769093992 z_k1_s0127
                                 - 0.320000042166428 z_k1_s0130
                                 - 0.353999941129154 z_k1_s0131
                                 - 0.288999652782962 z_k1_s0132
                                 - 0.563000222870249 z_k1_s0133
                                 - 0.980997716753689 z_k1_s0134
                                 - 1.14799923678743 z_k1_s0135
                                 - 0.767000639846189 z_k1_s0136
                                 - 0.694000465555789 z_k1_s0137
                                 - 0.161000412694785 z_k1_s0140
                                 - 0.144000163379267 z_k1_s0141
                                 - 0.134000541311109 z_k1_s0142
                                 - 0.128000003670515 z_k1_s0143
                                 - 0.124999989650118 z_k1_s0144
                                 - 0.0910000006188827 z_k1_s0145
                                 - 0.179999567224334 z_k1_s0146
                                 - 0.310999839715147 z_k1_s0147
                                 - 0.198999936107908 z_k1_s0150
                                 - 0.771000599048269 z_k1_s0151
                                 - 0.329000018198937 z_k1_s0152
                                 - 0.486999545439114 z_k1_s0153
                                 - 0.841999982376037 z_k1_s0154
                                 - 0.930998146900156 z_k1_s0155
                                 - 0.760997923496567 z_k1_s0156
                                 - 0.708998091820046 z_k1_s0157
                                 - 0.447000698999224 z_k1_s0160
                                 - 0.195000150649908 z_k1_s0161
                                 - 0.166000118511711 z_k1_s0162
                                 - 0.141000018082189 z_k1_s0163
                                 - 0.271999995992568 z_k1_s0164
                                 - 0.404000273180299 z_k1_s0165
                                 - 0.31300108865752 z_k1_s0166
                                 - 0.171000138597615 z_k1_s0167
                                 - 0.343000795185547 z_k1_s0170
                                 - 0.346000427428997 z_k1_s0171
                                 - 0.169000151256336 z_k1_s0172
                                 - 0.155000139952853 z_k1_s0173
                                 - 0.11599994558594 z_k1_s0174
                                 - 0.197999982177757 z_k1_s0175
                                 - 0.371000817792513 z_k1_s0176
                                 - 0.37499963645016 z_k1_s0177
                                 - 0.949000929443979 z_k1_s1000
                                 - 0.397000927544364 z_k1_s1001
                                 - 0.305000593077212 z_k1_s1002
                                 - 0.718998422523344 z_k1_s1003
                                 - 0.595001167942447 z_k1_s1004
                                 - 1.26999931427696 z_k1_s1005
                                 - 1.29300223817343 z_k1_s1006
                                 - 1.15500474382765 z_k1_s1007
                                 - 0.362998994316516 z_k1_s1010
                                 - 0.656998698060721 z_k1_s1011
                                 - 0.747999683659581 z_k1_s1012
                                 - 0.728996163114948 z_k1_s1013
                                 - 0.459997916474911 z_k1_s1014
                                 - 0.376998810109558 z_k1_s1015
                                 - 0.327999256384449 z_k1_s1016
                                 - 0.306999334245424 z_k1_s1017
                                 - 0.12000008114055 z_k1_s1020
                                 - 0.113000161657447 z_k1_s1021
                                 - 0.291999992841775 z_k1_s1022
                                 - 0.111000126205066 z_k1_s1023
                                 - 0.231000480145924 z_k1_s1024
                                 - 0.122000116592931 z_k1_s1025
                                 - 0.167999977196616 z_k1_s1026
                                 - 0.207000873118274 z_k1_s1027
                                 - 0.318999945184548 z_k1_s1030
                                 - 0.105000005247271 z_k1_s1031
                                 - 0.342999140840877 z_k1_s1032
                                 - 0.422999634696056 z_k1_s1033
                                 - 0.0860001022692162 z_k1_s1034
                                 - 0.189000047545096 z_k1_s1035
                                 - 0.110999994661398 z_k1_s1036
                                 - 0.328999292541277 z_k1_s1037
                                 - 0.708997660494365 z_k1_s1040
                                 - 0.320999309009488 z_k1_s1041
                                 - 0.659000260493734 z_k1_s1042
                                 - 0.232999521097804 z_k1_s1043
                                 - 0.510998616192042 z_k1_s1044
                                 - 0.242999383897103 z_k1_s1045
                                 - 0.831998690799876 z_k1_s1046
                                 - 0.31999943756931 z_k1_s1047
                                 - 0.264999964849685 z_k1_s1050
                                 - 0.503998468505851 z_k1_s1051
                                 - 0.599999535791947 z_k1_s1052
                                 - 0.30300003057979 z_k1_s1053
                                 - 0.811000480129852 z_k1_s1054
                                 - 0.901000206648754 z_k1_s1055
                                 - 0.334999978642042 z_k1_s1056
                                 - 0.357999954052404 z_k1_s1057
                                 - 0.32199989581927 z_k1_s1060
                                 - 0.395998428142626 z_k1_s1061
                                 - 0.955000956025295 z_k1_s1062
                                 - 0.220000040234214 z_k1_s1063
                                 - 0.402000046340716 z_k1_s1064
                                 - 0.526998880156364 z_k1_s1065
                                 - 0.2279998310665 z_k1_s1066
                                 - 0.686998433799729 z_k1_s1067
                                 - 0.224000037026449 z_k1_s1070
                                 - 0.200999951608555 z_k1_s1071
                                 - 0.650998746845627 z_k1_s1072
                                 - 0.681998370600183 z_k1_s1073
                                 - 0.17400009702233 z_k1_s1074
                                 - 0.179999916974826 z_k1_s1075
                                 - 0.637998582948558 z_k1_s1076
                                 - 0.574001611455263 z_k1_s1077
                                 - 0.486000027254109 z_k1_s1100
                                 - 0.730001128080406 z_k1_s1101
                                 - 0.337999712837876 z_k1_s1102
                                 - 0.20499971682306 z_k1_s1103
                                 - 0.350999247239697 z_k1_s1104
                                 - 0.825000132091783 z_k1_s1105
                                 - 0.319999608835955 z_k1_s1106
                                 - 0.22099973202483 z_k1_s1107
                                 - 0.175999898977632 z_k1_s1110
                                 - 0.247999341187212 z_k1_s1111
                                 - 0.253000662866986 z_k1_s1112
                                 - 0.215000021290703 z_k1_s1113
                                 - 0.113999965730933 z_k1_s1114
                                 - 0.35399875942884 z_k1_s1115
                                 - 0.477000320584199 z_k1_s1116
                                 - 0.121999925520863 z_k1_s1117
                                 - 0.149999969542801 z_k1_s1120
                                 - 0.456999976476344 z_k1_s1121
                                 - 0.289999798502866 z_k1_s1122
                                 - 0.183999775204657 z_k1_s1123
                                 - 0.678000546876899 z_k1_s1124
                                 - 0.476998424043323 z_k1_s1125
                                 - 0.598000644614004 z_k1_s1126
                                 - 0.221999881579406 z_k1_s1127
                                 - 0.36600131224877 z_k1_s1130
                                 - 1.11900244746903 z_k1_s1131
                                 - 0.34800096398159 z_k1_s1132
                                 - 1.18800518358074 z_k1_s1133
                                 - 0.849004640983475 z_k1_s1134
                                 - 0.400001255806867 z_k1_s1135
                                 - 0.385001048001129 z_k1_s1136
                                 - 0.290000885734904 z_k1_s1137
                                 - 0.346000120179272 z_k1_s1140
                                 - 0.505001886967016 z_k1_s1141
                                 - 0.262999992459958 z_k1_s1142
                                 - 0.94300056770249 z_k1_s1143
                                 - 0.346000120179272 z_k1_s1144
                                 - 0.351000168644295 z_k1_s1145
                                 - 0.432001480177712 z_k1_s1146
                                 - 0.574001653384246 z_k1_s1147
                                 - 1.58299929309716 z_k1_s1150
                                 - 0.761998863142331 z_k1_s1151
                                 - 0.822999159636775 z_k1_s1152
                                 - 1.06400269984099 z_k1_s1153
                                 - 0.78399924962354 z_k1_s1154
                                 - 1.54299777449611 z_k1_s1155
                                 - 0.794998994114776 z_k1_s1156
                                 - 1.62099938952005 z_k1_s1157
                                 - 0.717999962615921 z_k1_s1160
                                 - 0.637998667394494 z_k1_s1161
                                 - 0.734997322351377 z_k1_s1162
                                 - 1.256999942673 z_k1_s1163
                                 - 0.465999964367161 z_k1_s1164
                                 - 1.08699719032746 z_k1_s1165
                                 - 0.595000886712074 z_k1_s1166
                                 - 0.335000175841352 z_k1_s1167
                                 - 0.85399783249402 z_k1_s1170
                                 - 0.516000239175955 z_k1_s1171
                                 - 0.775001671786932 z_k1_s1172
                                 - 0.68799766289856 z_k1_s1173
                                 - 0.578000638519181 z_k1_s1174
                                 - 1.12099914527814 z_k1_s1175
                                 - 1.3330012818719 z_k1_s1176
                                 - 0.775001671786932 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.946811 w_k1_ADM_s000
                                 + 1.12049 w_k1_AXP_s000
                                 + 0.938586 w_k1_DIS_s000
                                 + 1.09554 w_k1_FDX_s000
                                 + 0.742783 w_k1_IBM_s000
                                 + 1.29401 w_k1_INTC_s000
                                 + 0.986741 w_k1_KO_s000
                                 + 0.767449 w_k1_MAT_s000
                                 + 1.10992 w_k1_MSFT_s000
                                 + 1.00065 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.678397 w_k1_ADM_s000
                                 + 0.756463 w_k1_AXP_s000
                                 + 0.734412 w_k1_DIS_s000
                                 + 0.749653 w_k1_FDX_s000
                                 + 0.824304 w_k1_IBM_s000
                                 + 0.964816 w_k1_INTC_s000
                                 + 0.932776 w_k1_KO_s000
                                 + 0.721533 w_k1_MAT_s000
                                 + 0.883277 w_k1_MSFT_s000
                                 + 0.857777 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.907757 w_k1_ADM_s000
                                 + 0.770695 w_k1_AXP_s000
                                 + 0.869637 w_k1_DIS_s000
                                 + 0.85706 w_k1_FDX_s000
                                 + 1.11061 w_k1_IBM_s000
                                 + 1.08343 w_k1_INTC_s000 + 1.2048 w_k1_KO_s000
                                 + 1.34122 w_k1_MAT_s000
                                 + 1.00712 w_k1_MSFT_s000
                                 + 0.967195 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.959614 w_k1_ADM_s000
                                 + 1.24261 w_k1_AXP_s000
                                 + 1.08225 w_k1_DIS_s000
                                 + 0.965306 w_k1_FDX_s000
                                 + 1.41796 w_k1_IBM_s000
                                 + 1.1594 w_k1_INTC_s000 + 1.45901 w_k1_KO_s000
                                 + 1.95202 w_k1_MAT_s000
                                 + 1.0344 w_k1_MSFT_s000 + 1.07002 w_k1_PG_s000
                                 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.33728 w_k1_ADM_s000
                                 + 1.8419 w_k1_AXP_s000 + 1.83106 w_k1_DIS_s000
                                 + 1.49243 w_k1_FDX_s000
                                 + 1.37409 w_k1_IBM_s000
                                 + 1.84206 w_k1_INTC_s000
                                 + 1.50234 w_k1_KO_s000 + 1.63227 w_k1_MAT_s000
                                 + 1.63644 w_k1_MSFT_s000
                                 + 1.36013 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.10685 w_k1_ADM_s000
                                 + 0.945797 w_k1_AXP_s000
                                 + 1.0237 w_k1_DIS_s000 + 1.02251 w_k1_FDX_s000
                                 + 0.966967 w_k1_IBM_s000
                                 + 0.915796 w_k1_INTC_s000
                                 + 1.13083 w_k1_KO_s000 + 1.14663 w_k1_MAT_s000
                                 + 1.11793 w_k1_MSFT_s000
                                 + 1.03626 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.22648 w_k1_ADM_s000
                                 + 1.13025 w_k1_AXP_s000 + 1.1993 w_k1_DIS_s000
                                 + 1.19225 w_k1_FDX_s000
                                 + 0.944353 w_k1_IBM_s000
                                 + 1.40297 w_k1_INTC_s000
                                 + 1.26608 w_k1_KO_s000 + 1.01518 w_k1_MAT_s000
                                 + 1.36356 w_k1_MSFT_s000
                                 + 1.23456 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.03189 w_k1_ADM_s000
                                 + 1.36887 w_k1_AXP_s000
                                 + 1.23582 w_k1_DIS_s000 + 1.2035 w_k1_FDX_s000
                                 + 1.28535 w_k1_IBM_s000
                                 + 1.36879 w_k1_INTC_s000
                                 + 1.24926 w_k1_KO_s000 + 1.68963 w_k1_MAT_s000
                                 + 1.16011 w_k1_MSFT_s000
                                 + 1.20385 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.26626 w_k1_ADM_s001
                                 + 1.18337 w_k1_AXP_s001
                                 + 1.17138 w_k1_DIS_s001
                                 + 1.08252 w_k1_FDX_s001
                                 + 0.911025 w_k1_IBM_s001
                                 + 0.978462 w_k1_INTC_s001
                                 + 1.18093 w_k1_KO_s001
                                 + 0.825895 w_k1_MAT_s001
                                 + 1.03309 w_k1_MSFT_s001
                                 + 0.968644 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.50779 w_k1_ADM_s001
                                 + 1.41448 w_k1_AXP_s001
                                 + 1.61401 w_k1_DIS_s001
                                 + 1.41035 w_k1_FDX_s001
                                 + 0.827708 w_k1_IBM_s001
                                 + 1.17091 w_k1_INTC_s001
                                 + 1.23205 w_k1_KO_s001 + 1.16191 w_k1_MAT_s001
                                 + 1.31214 w_k1_MSFT_s001
                                 + 1.09752 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.30169 w_k1_ADM_s001
                                 + 1.07066 w_k1_AXP_s001
                                 + 0.858847 w_k1_DIS_s001
                                 + 0.951393 w_k1_FDX_s001
                                 + 0.848838 w_k1_IBM_s001
                                 + 0.95253 w_k1_INTC_s001
                                 + 0.960241 w_k1_KO_s001
                                 + 0.99398 w_k1_MAT_s001
                                 + 0.929485 w_k1_MSFT_s001
                                 + 0.971316 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.759209 w_k1_ADM_s001
                                 + 0.764461 w_k1_AXP_s001
                                 + 0.614544 w_k1_DIS_s001
                                 + 0.674665 w_k1_FDX_s001
                                 + 1.06416 w_k1_IBM_s001
                                 + 0.757509 w_k1_INTC_s001
                                 + 0.902767 w_k1_KO_s001
                                 + 1.07204 w_k1_MAT_s001
                                 + 0.741654 w_k1_MSFT_s001
                                 + 0.819027 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.869074 w_k1_ADM_s001
                                 + 1.13166 w_k1_AXP_s001
                                 + 0.933732 w_k1_DIS_s001
                                 + 0.93855 w_k1_FDX_s001
                                 + 1.27472 w_k1_IBM_s001
                                 + 1.00049 w_k1_INTC_s001
                                 + 1.08257 w_k1_KO_s001 + 1.27105 w_k1_MAT_s001
                                 + 0.977854 w_k1_MSFT_s001
                                 + 1.06101 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.843646 w_k1_ADM_s001
                                 + 0.886087 w_k1_AXP_s001
                                 + 0.71286 w_k1_DIS_s001
                                 + 0.855644 w_k1_FDX_s001
                                 + 0.536201 w_k1_IBM_s001
                                 + 0.825046 w_k1_INTC_s001
                                 + 0.637587 w_k1_KO_s001
                                 + 0.825459 w_k1_MAT_s001
                                 + 0.88338 w_k1_MSFT_s001
                                 + 0.789963 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.37589 w_k1_ADM_s001
                                 + 1.64099 w_k1_AXP_s001
                                 + 1.24379 w_k1_DIS_s001
                                 + 1.14005 w_k1_FDX_s001
                                 + 1.58151 w_k1_IBM_s001
                                 + 1.02293 w_k1_INTC_s001
                                 + 1.30641 w_k1_KO_s001 + 2.30159 w_k1_MAT_s001
                                 + 1.10952 w_k1_MSFT_s001
                                 + 1.24357 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.80294 w_k1_ADM_s001
                                 + 1.77923 w_k1_AXP_s001
                                 + 2.00086 w_k1_DIS_s001
                                 + 1.46803 w_k1_FDX_s001
                                 + 1.42833 w_k1_IBM_s001
                                 + 1.02765 w_k1_INTC_s001
                                 + 1.62518 w_k1_KO_s001 + 1.84662 w_k1_MAT_s001
                                 + 1.403 w_k1_MSFT_s001 + 1.28769 w_k1_PG_s001
                                 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.689429 w_k1_ADM_s002
                                 + 0.706283 w_k1_AXP_s002
                                 + 0.567764 w_k1_DIS_s002
                                 + 0.728281 w_k1_FDX_s002
                                 + 0.720233 w_k1_IBM_s002
                                 + 0.749516 w_k1_INTC_s002
                                 + 0.707912 w_k1_KO_s002
                                 + 0.702213 w_k1_MAT_s002
                                 + 0.707436 w_k1_MSFT_s002
                                 + 0.77382 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.07443 w_k1_ADM_s002
                                 + 1.57074 w_k1_AXP_s002 + 1.7369 w_k1_DIS_s002
                                 + 1.26623 w_k1_FDX_s002
                                 + 1.51865 w_k1_IBM_s002
                                 + 1.41241 w_k1_INTC_s002
                                 + 1.46393 w_k1_KO_s002 + 1.98995 w_k1_MAT_s002
                                 + 1.28846 w_k1_MSFT_s002 + 1.1842 w_k1_PG_s002
                                 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.714858 w_k1_ADM_s002
                                 + 0.849044 w_k1_AXP_s002 + 1.126 w_k1_DIS_s002
                                 + 1.05846 w_k1_FDX_s002
                                 + 0.977938 w_k1_IBM_s002
                                 + 1.26878 w_k1_INTC_s002
                                 + 1.01167 w_k1_KO_s002
                                 + 0.831607 w_k1_MAT_s002
                                 + 1.08934 w_k1_MSFT_s002
                                 + 0.87683 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.0686 w_k1_ADM_s002
                                 + 1.43849 w_k1_AXP_s002 + 1.3911 w_k1_DIS_s002
                                 + 1.23182 w_k1_FDX_s002
                                 + 0.886149 w_k1_IBM_s002
                                 + 1.43434 w_k1_INTC_s002
                                 + 1.05458 w_k1_KO_s002
                                 + 0.932678 w_k1_MAT_s002
                                 + 1.16296 w_k1_MSFT_s002
                                 + 1.03981 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.09874 w_k1_ADM_s002
                                 + 0.892882 w_k1_AXP_s002
                                 + 0.64651 w_k1_DIS_s002
                                 + 0.695262 w_k1_FDX_s002
                                 + 1.03585 w_k1_IBM_s002
                                 + 0.987899 w_k1_INTC_s002
                                 + 0.836855 w_k1_KO_s002
                                 + 0.958414 w_k1_MAT_s002
                                 + 0.689983 w_k1_MSFT_s002
                                 + 0.80384 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.972098 w_k1_ADM_s002
                                 + 0.953798 w_k1_AXP_s002
                                 + 1.01495 w_k1_DIS_s002
                                 + 0.892567 w_k1_FDX_s002
                                 + 1.51012 w_k1_IBM_s002
                                 + 0.932802 w_k1_INTC_s002
                                 + 1.22685 w_k1_KO_s002 + 1.36674 w_k1_MAT_s002
                                 + 0.933643 w_k1_MSFT_s002
                                 + 0.921674 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.14476 w_k1_ADM_s002
                                 + 1.26511 w_k1_AXP_s002
                                 + 1.05249 w_k1_DIS_s002
                                 + 1.06933 w_k1_FDX_s002
                                 + 1.15229 w_k1_IBM_s002
                                 + 1.17954 w_k1_INTC_s002
                                 + 1.06111 w_k1_KO_s002 + 1.31219 w_k1_MAT_s002
                                 + 1.02525 w_k1_MSFT_s002
                                 + 1.05181 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.11374 w_k1_ADM_s002
                                 + 0.905984 w_k1_AXP_s002
                                 + 0.987081 w_k1_DIS_s002
                                 + 0.984155 w_k1_FDX_s002
                                 + 0.808677 w_k1_IBM_s002
                                 + 0.988419 w_k1_INTC_s002
                                 + 0.890997 w_k1_KO_s002
                                 + 0.587822 w_k1_MAT_s002
                                 + 0.970634 w_k1_MSFT_s002
                                 + 1.02089 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.664018 w_k1_ADM_s003
                                 + 0.925923 w_k1_AXP_s003
                                 + 0.765581 w_k1_DIS_s003
                                 + 0.821549 w_k1_FDX_s003
                                 + 0.87746 w_k1_IBM_s003
                                 + 0.761875 w_k1_INTC_s003
                                 + 0.700205 w_k1_KO_s003
                                 + 0.793901 w_k1_MAT_s003
                                 + 0.78274 w_k1_MSFT_s003
                                 + 0.919866 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.871613 w_k1_ADM_s003
                                 + 0.833658 w_k1_AXP_s003
                                 + 0.930289 w_k1_DIS_s003
                                 + 1.04722 w_k1_FDX_s003
                                 + 0.602805 w_k1_IBM_s003
                                 + 0.914988 w_k1_INTC_s003
                                 + 0.757735 w_k1_KO_s003
                                 + 0.487357 w_k1_MAT_s003
                                 + 0.95131 w_k1_MSFT_s003
                                 + 1.01318 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.24647 w_k1_ADM_s003
                                 + 1.51441 w_k1_AXP_s003
                                 + 0.959737 w_k1_DIS_s003
                                 + 1.29013 w_k1_FDX_s003
                                 + 0.838851 w_k1_IBM_s003
                                 + 0.837113 w_k1_INTC_s003
                                 + 0.819203 w_k1_KO_s003
                                 + 0.964049 w_k1_MAT_s003
                                 + 1.09065 w_k1_MSFT_s003 + 1.0468 w_k1_PG_s003
                                 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.43536 w_k1_ADM_s003
                                 + 1.75341 w_k1_AXP_s003
                                 + 2.23759 w_k1_DIS_s003
                                 + 1.71839 w_k1_FDX_s003
                                 + 1.27057 w_k1_IBM_s003
                                 + 1.35064 w_k1_INTC_s003
                                 + 1.43205 w_k1_KO_s003
                                 + 0.871159 w_k1_MAT_s003
                                 + 1.51572 w_k1_MSFT_s003
                                 + 1.19789 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.754114 w_k1_ADM_s003
                                 + 1.32333 w_k1_AXP_s003
                                 + 1.13085 w_k1_DIS_s003
                                 + 0.966573 w_k1_FDX_s003
                                 + 1.11374 w_k1_IBM_s003
                                 + 1.11538 w_k1_INTC_s003
                                 + 1.08277 w_k1_KO_s003 + 1.18841 w_k1_MAT_s003
                                 + 1.15924 w_k1_MSFT_s003 + 1.1179 w_k1_PG_s003
                                 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.853711 w_k1_ADM_s003
                                 + 1.4436 w_k1_AXP_s003 + 1.61856 w_k1_DIS_s003
                                 + 1.11228 w_k1_FDX_s003
                                 + 1.41798 w_k1_IBM_s003
                                 + 1.06703 w_k1_INTC_s003
                                 + 1.29192 w_k1_KO_s003 + 1.58612 w_k1_MAT_s003
                                 + 1.25891 w_k1_MSFT_s003
                                 + 1.14499 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.513212 w_k1_ADM_s003
                                 + 0.805371 w_k1_AXP_s003
                                 + 0.602384 w_k1_DIS_s003
                                 + 0.695854 w_k1_FDX_s003
                                 + 0.696958 w_k1_IBM_s003
                                 + 0.711353 w_k1_INTC_s003
                                 + 0.647742 w_k1_KO_s003
                                 + 0.610183 w_k1_MAT_s003
                                 + 0.786174 w_k1_MSFT_s003
                                 + 0.859303 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.871756 w_k1_ADM_s003
                                 + 0.91245 w_k1_AXP_s003
                                 + 1.22898 w_k1_DIS_s003
                                 + 1.16147 w_k1_FDX_s003
                                 + 0.815315 w_k1_IBM_s003
                                 + 1.10324 w_k1_INTC_s003
                                 + 0.828038 w_k1_KO_s003
                                 + 0.479299 w_k1_MAT_s003
                                 + 1.14263 w_k1_MSFT_s003
                                 + 1.02824 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.05257 w_k1_ADM_s004
                                 + 1.1234 w_k1_AXP_s004 + 1.24379 w_k1_DIS_s004
                                 + 1.11811 w_k1_FDX_s004
                                 + 1.52536 w_k1_IBM_s004
                                 + 1.65409 w_k1_INTC_s004
                                 + 1.28281 w_k1_KO_s004 + 1.30027 w_k1_MAT_s004
                                 + 1.08809 w_k1_MSFT_s004
                                 + 1.28861 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.21075 w_k1_ADM_s004
                                 + 1.29861 w_k1_AXP_s004 + 1.6402 w_k1_DIS_s004
                                 + 1.40388 w_k1_FDX_s004 + 1.2505 w_k1_IBM_s004
                                 + 1.62613 w_k1_INTC_s004
                                 + 1.33097 w_k1_KO_s004 + 1.19557 w_k1_MAT_s004
                                 + 1.70471 w_k1_MSFT_s004 + 1.2071 w_k1_PG_s004
                                 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.945757 w_k1_ADM_s004
                                 + 0.805936 w_k1_AXP_s004
                                 + 0.784658 w_k1_DIS_s004
                                 + 0.883828 w_k1_FDX_s004
                                 + 0.791665 w_k1_IBM_s004
                                 + 1.06983 w_k1_INTC_s004
                                 + 0.853295 w_k1_KO_s004
                                 + 0.883896 w_k1_MAT_s004
                                 + 1.05818 w_k1_MSFT_s004
                                 + 0.893807 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.750446 w_k1_ADM_s004
                                 + 0.713197 w_k1_AXP_s004
                                 + 0.555992 w_k1_DIS_s004
                                 + 0.644965 w_k1_FDX_s004
                                 + 1.20124 w_k1_IBM_s004
                                 + 0.89987 w_k1_INTC_s004
                                 + 0.882965 w_k1_KO_s004
                                 + 1.06221 w_k1_MAT_s004
                                 + 0.746499 w_k1_MSFT_s004
                                 + 0.836255 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.882418 w_k1_ADM_s004
                                 + 0.658605 w_k1_AXP_s004
                                 + 0.682087 w_k1_DIS_s004
                                 + 0.749544 w_k1_FDX_s004
                                 + 0.841552 w_k1_IBM_s004
                                 + 0.90863 w_k1_INTC_s004
                                 + 0.814712 w_k1_KO_s004
                                 + 0.991099 w_k1_MAT_s004
                                 + 0.889004 w_k1_MSFT_s004
                                 + 0.790131 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.930628 w_k1_ADM_s004
                                 + 1.02459 w_k1_AXP_s004
                                 + 1.05759 w_k1_DIS_s004
                                 + 1.03149 w_k1_FDX_s004
                                 + 1.07498 w_k1_IBM_s004
                                 + 1.30118 w_k1_INTC_s004
                                 + 1.06506 w_k1_KO_s004 + 1.21814 w_k1_MAT_s004
                                 + 1.18319 w_k1_MSFT_s004 + 1.1065 w_k1_PG_s004
                                 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.20559 w_k1_ADM_s004
                                 + 1.33615 w_k1_AXP_s004
                                 + 1.07184 w_k1_DIS_s004
                                 + 1.23123 w_k1_FDX_s004
                                 + 0.827731 w_k1_IBM_s004
                                 + 1.45334 w_k1_INTC_s004
                                 + 0.889388 w_k1_KO_s004
                                 + 0.739073 w_k1_MAT_s004
                                 + 1.15414 w_k1_MSFT_s004
                                 + 1.07257 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.02868 w_k1_ADM_s004
                                 + 0.914992 w_k1_AXP_s004
                                 + 0.823479 w_k1_DIS_s004
                                 + 0.892532 w_k1_FDX_s004
                                 + 1.18179 w_k1_IBM_s004
                                 + 1.0288 w_k1_INTC_s004 + 1.06348 w_k1_KO_s004
                                 + 1.44699 w_k1_MAT_s004
                                 + 1.04789 w_k1_MSFT_s004
                                 + 1.01161 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.841292 w_k1_ADM_s005
                                 + 1.32257 w_k1_AXP_s005
                                 + 1.66444 w_k1_DIS_s005
                                 + 1.17155 w_k1_FDX_s005 + 1.6726 w_k1_IBM_s005
                                 + 1.24731 w_k1_INTC_s005
                                 + 1.45233 w_k1_KO_s005 + 1.90674 w_k1_MAT_s005
                                 + 1.20858 w_k1_MSFT_s005 + 1.1315 w_k1_PG_s005
                                 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.32065 w_k1_ADM_s005
                                 + 0.957771 w_k1_AXP_s005
                                 + 1.01665 w_k1_DIS_s005
                                 + 1.03925 w_k1_FDX_s005
                                 + 0.980029 w_k1_IBM_s005
                                 + 0.903002 w_k1_INTC_s005
                                 + 0.912982 w_k1_KO_s005
                                 + 1.00337 w_k1_MAT_s005
                                 + 0.877025 w_k1_MSFT_s005
                                 + 1.00985 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.20723 w_k1_ADM_s005
                                 + 0.991812 w_k1_AXP_s005
                                 + 1.01382 w_k1_DIS_s005
                                 + 1.00201 w_k1_FDX_s005
                                 + 1.34003 w_k1_IBM_s005
                                 + 1.05364 w_k1_INTC_s005
                                 + 1.00259 w_k1_KO_s005
                                 + 0.996447 w_k1_MAT_s005
                                 + 0.881552 w_k1_MSFT_s005
                                 + 0.932079 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.27674 w_k1_ADM_s005
                                 + 1.53835 w_k1_AXP_s005 + 1.2099 w_k1_DIS_s005
                                 + 1.31218 w_k1_FDX_s005
                                 + 1.15262 w_k1_IBM_s005
                                 + 1.11041 w_k1_INTC_s005
                                 + 0.993451 w_k1_KO_s005
                                 + 1.30083 w_k1_MAT_s005
                                 + 1.15883 w_k1_MSFT_s005 + 1.0977 w_k1_PG_s005
                                 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.00343 w_k1_ADM_s005
                                 + 0.742087 w_k1_AXP_s005
                                 + 0.937059 w_k1_DIS_s005
                                 + 0.91933 w_k1_FDX_s005
                                 + 0.822259 w_k1_IBM_s005
                                 + 0.9031 w_k1_INTC_s005
                                 + 0.843173 w_k1_KO_s005
                                 + 0.70869 w_k1_MAT_s005
                                 + 0.866272 w_k1_MSFT_s005
                                 + 0.856175 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.06846 w_k1_ADM_s005
                                 + 0.899543 w_k1_AXP_s005
                                 + 0.897687 w_k1_DIS_s005
                                 + 1.03803 w_k1_FDX_s005
                                 + 0.724474 w_k1_IBM_s005
                                 + 0.965364 w_k1_INTC_s005
                                 + 0.702052 w_k1_KO_s005
                                 + 0.706121 w_k1_MAT_s005
                                 + 0.842702 w_k1_MSFT_s005
                                 + 0.83145 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.702568 w_k1_ADM_s005
                                 + 0.602266 w_k1_AXP_s005
                                 + 0.486541 w_k1_DIS_s005
                                 + 0.686619 w_k1_FDX_s005
                                 + 0.67785 w_k1_IBM_s005
                                 + 0.794637 w_k1_INTC_s005
                                 + 0.65209 w_k1_KO_s005
                                 + 0.705625 w_k1_MAT_s005
                                 + 0.747066 w_k1_MSFT_s005
                                 + 0.777514 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.07397 w_k1_ADM_s005
                                 + 0.760746 w_k1_AXP_s005
                                 + 0.68192 w_k1_DIS_s005
                                 + 0.832558 w_k1_FDX_s005
                                 + 0.979347 w_k1_IBM_s005
                                 + 0.813715 w_k1_INTC_s005
                                 + 0.732701 w_k1_KO_s005
                                 + 0.818772 w_k1_MAT_s005
                                 + 0.731555 w_k1_MSFT_s005
                                 + 0.820051 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.85271 w_k1_ADM_s006
                                 + 1.59351 w_k1_AXP_s006
                                 + 1.56933 w_k1_DIS_s006
                                 + 1.25827 w_k1_FDX_s006
                                 + 1.50212 w_k1_IBM_s006
                                 + 1.17991 w_k1_INTC_s006
                                 + 1.32813 w_k1_KO_s006 + 1.43371 w_k1_MAT_s006
                                 + 1.21113 w_k1_MSFT_s006
                                 + 1.19458 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.04432 w_k1_ADM_s006
                                 + 0.914416 w_k1_AXP_s006
                                 + 1.12417 w_k1_DIS_s006 + 1.1254 w_k1_FDX_s006
                                 + 0.70036 w_k1_IBM_s006
                                 + 0.989124 w_k1_INTC_s006
                                 + 0.862994 w_k1_KO_s006
                                 + 0.697464 w_k1_MAT_s006
                                 + 1.19298 w_k1_MSFT_s006
                                 + 1.23329 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.09292 w_k1_ADM_s006
                                 + 0.937186 w_k1_AXP_s006
                                 + 0.798414 w_k1_DIS_s006
                                 + 0.932617 w_k1_FDX_s006
                                 + 0.871997 w_k1_IBM_s006
                                 + 0.835413 w_k1_INTC_s006
                                 + 0.768039 w_k1_KO_s006
                                 + 0.787471 w_k1_MAT_s006
                                 + 0.867878 w_k1_MSFT_s006
                                 + 0.925368 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.0412 w_k1_ADM_s006
                                 + 1.40441 w_k1_AXP_s006 + 1.4791 w_k1_DIS_s006
                                 + 1.25627 w_k1_FDX_s006
                                 + 1.13874 w_k1_IBM_s006
                                 + 1.01604 w_k1_INTC_s006
                                 + 1.12523 w_k1_KO_s006 + 1.19235 w_k1_MAT_s006
                                 + 1.25398 w_k1_MSFT_s006
                                 + 1.13618 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.951113 w_k1_ADM_s006
                                 + 1.22878 w_k1_AXP_s006
                                 + 1.05386 w_k1_DIS_s006
                                 + 0.998718 w_k1_FDX_s006
                                 + 1.03259 w_k1_IBM_s006
                                 + 0.857536 w_k1_INTC_s006
                                 + 0.928478 w_k1_KO_s006
                                 + 1.24981 w_k1_MAT_s006
                                 + 0.9701 w_k1_MSFT_s006
                                 + 0.953324 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.33146 w_k1_ADM_s006
                                 + 1.26103 w_k1_AXP_s006
                                 + 1.87625 w_k1_DIS_s006
                                 + 1.40108 w_k1_FDX_s006
                                 + 0.926476 w_k1_IBM_s006
                                 + 1.28931 w_k1_INTC_s006
                                 + 1.24272 w_k1_KO_s006
                                 + 0.659903 w_k1_MAT_s006
                                 + 1.39622 w_k1_MSFT_s006
                                 + 1.08418 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.922043 w_k1_ADM_s006
                                 + 0.967394 w_k1_AXP_s006
                                 + 1.18507 w_k1_DIS_s006
                                 + 0.976601 w_k1_FDX_s006
                                 + 1.26919 w_k1_IBM_s006
                                 + 1.00608 w_k1_INTC_s006
                                 + 1.05863 w_k1_KO_s006 + 1.41343 w_k1_MAT_s006
                                 + 0.97385 w_k1_MSFT_s006
                                 + 1.07435 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.706049 w_k1_ADM_s006
                                 + 0.549843 w_k1_AXP_s006
                                 + 0.529995 w_k1_DIS_s006
                                 + 0.683276 w_k1_FDX_s006
                                 + 0.587382 w_k1_IBM_s006
                                 + 0.661781 w_k1_INTC_s006
                                 + 0.511717 w_k1_KO_s006
                                 + 0.473744 w_k1_MAT_s006
                                 + 0.693048 w_k1_MSFT_s006
                                 + 0.743906 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.884946 w_k1_ADM_s007
                                 + 1.07048 w_k1_AXP_s007
                                 + 0.970379 w_k1_DIS_s007
                                 + 0.933059 w_k1_FDX_s007
                                 + 1.08888 w_k1_IBM_s007
                                 + 0.93629 w_k1_INTC_s007
                                 + 1.04843 w_k1_KO_s007 + 1.53662 w_k1_MAT_s007
                                 + 0.904279 w_k1_MSFT_s007 + 1.002 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.24353 w_k1_ADM_s007
                                 + 1.10572 w_k1_AXP_s007
                                 + 1.13719 w_k1_DIS_s007
                                 + 0.954446 w_k1_FDX_s007
                                 + 1.65753 w_k1_IBM_s007
                                 + 1.2321 w_k1_INTC_s007 + 1.38154 w_k1_KO_s007
                                 + 1.57505 w_k1_MAT_s007
                                 + 0.985958 w_k1_MSFT_s007
                                 + 1.10329 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.3584 w_k1_ADM_s007
                                 + 1.66936 w_k1_AXP_s007 + 1.6098 w_k1_DIS_s007
                                 + 1.33741 w_k1_FDX_s007
                                 + 1.33907 w_k1_IBM_s007
                                 + 1.33115 w_k1_INTC_s007
                                 + 1.27686 w_k1_KO_s007 + 1.85593 w_k1_MAT_s007
                                 + 1.2773 w_k1_MSFT_s007 + 1.09402 w_k1_PG_s007
                                 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.748699 w_k1_ADM_s007
                                 + 0.790011 w_k1_AXP_s007
                                 + 0.760559 w_k1_DIS_s007
                                 + 0.743977 w_k1_FDX_s007
                                 + 0.854996 w_k1_IBM_s007
                                 + 0.831108 w_k1_INTC_s007
                                 + 0.795103 w_k1_KO_s007
                                 + 0.944911 w_k1_MAT_s007
                                 + 0.679971 w_k1_MSFT_s007
                                 + 0.888855 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.946892 w_k1_ADM_s007
                                 + 0.895809 w_k1_AXP_s007
                                 + 0.731051 w_k1_DIS_s007
                                 + 0.946522 w_k1_FDX_s007
                                 + 0.653422 w_k1_IBM_s007
                                 + 0.887745 w_k1_INTC_s007
                                 + 0.662014 w_k1_KO_s007
                                 + 0.75268 w_k1_MAT_s007
                                 + 0.799968 w_k1_MSFT_s007
                                 + 0.819541 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.898065 w_k1_ADM_s007
                                 + 0.724749 w_k1_AXP_s007
                                 + 0.848169 w_k1_DIS_s007
                                 + 0.812152 w_k1_FDX_s007
                                 + 0.885357 w_k1_IBM_s007
                                 + 0.909377 w_k1_INTC_s007
                                 + 0.899837 w_k1_KO_s007
                                 + 0.705035 w_k1_MAT_s007
                                 + 0.891796 w_k1_MSFT_s007
                                 + 0.873975 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.49565 w_k1_ADM_s007
                                 + 1.3771 w_k1_AXP_s007 + 1.34947 w_k1_DIS_s007
                                 + 1.28671 w_k1_FDX_s007
                                 + 0.912652 w_k1_IBM_s007
                                 + 0.969221 w_k1_INTC_s007
                                 + 1.01162 w_k1_KO_s007 + 1.04656 w_k1_MAT_s007
                                 + 1.10441 w_k1_MSFT_s007
                                 + 1.22449 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.82218 w_k1_ADM_s007
                                 + 0.981692 w_k1_AXP_s007
                                 + 0.548899 w_k1_DIS_s007
                                 + 0.757255 w_k1_FDX_s007
                                 + 1.06204 w_k1_IBM_s007
                                 + 0.827092 w_k1_INTC_s007
                                 + 0.893527 w_k1_KO_s007
                                 + 1.27859 w_k1_MAT_s007
                                 + 0.715839 w_k1_MSFT_s007
                                 + 0.7983 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.785265 w_k1_ADM_s010
                                 + 0.576598 w_k1_AXP_s010
                                 + 0.495568 w_k1_DIS_s010
                                 + 0.691393 w_k1_FDX_s010
                                 + 0.663421 w_k1_IBM_s010
                                 + 0.835727 w_k1_INTC_s010
                                 + 0.602261 w_k1_KO_s010
                                 + 0.513713 w_k1_MAT_s010
                                 + 0.775133 w_k1_MSFT_s010
                                 + 0.701948 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.850107 w_k1_ADM_s010
                                 + 1.05418 w_k1_AXP_s010
                                 + 1.29585 w_k1_DIS_s010
                                 + 1.03861 w_k1_FDX_s010 + 1.4675 w_k1_IBM_s010
                                 + 0.997203 w_k1_INTC_s010
                                 + 1.39011 w_k1_KO_s010 + 1.54796 w_k1_MAT_s010
                                 + 1.07293 w_k1_MSFT_s010 + 1.1377 w_k1_PG_s010
                                 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.908251 w_k1_ADM_s010
                                 + 1.15302 w_k1_AXP_s010
                                 + 1.09856 w_k1_DIS_s010
                                 + 1.15106 w_k1_FDX_s010
                                 + 0.934788 w_k1_IBM_s010
                                 + 1.33621 w_k1_INTC_s010
                                 + 1.07025 w_k1_KO_s010 + 1.15421 w_k1_MAT_s010
                                 + 1.18216 w_k1_MSFT_s010
                                 + 0.929322 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.898524 w_k1_ADM_s010
                                 + 0.807351 w_k1_AXP_s010
                                 + 1.22844 w_k1_DIS_s010
                                 + 0.891743 w_k1_FDX_s010
                                 + 1.14415 w_k1_IBM_s010
                                 + 0.981094 w_k1_INTC_s010
                                 + 1.19532 w_k1_KO_s010
                                 + 0.925288 w_k1_MAT_s010
                                 + 1.04623 w_k1_MSFT_s010
                                 + 0.927929 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.889835 w_k1_ADM_s010
                                 + 0.864786 w_k1_AXP_s010
                                 + 0.667398 w_k1_DIS_s010
                                 + 0.731774 w_k1_FDX_s010
                                 + 1.0962 w_k1_IBM_s010
                                 + 0.813854 w_k1_INTC_s010
                                 + 0.840784 w_k1_KO_s010
                                 + 1.14843 w_k1_MAT_s010
                                 + 0.746958 w_k1_MSFT_s010
                                 + 0.803991 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.740761 w_k1_ADM_s010
                                 + 0.743292 w_k1_AXP_s010
                                 + 0.793036 w_k1_DIS_s010
                                 + 0.892038 w_k1_FDX_s010
                                 + 0.812669 w_k1_IBM_s010
                                 + 0.992198 w_k1_INTC_s010
                                 + 0.880442 w_k1_KO_s010
                                 + 0.607663 w_k1_MAT_s010
                                 + 0.964799 w_k1_MSFT_s010
                                 + 0.810959 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.37791 w_k1_ADM_s010
                                 + 1.6064 w_k1_AXP_s010 + 1.84839 w_k1_DIS_s010
                                 + 1.54044 w_k1_FDX_s010
                                 + 1.19762 w_k1_IBM_s010
                                 + 1.4151 w_k1_INTC_s010 + 1.42862 w_k1_KO_s010
                                 + 1.23964 w_k1_MAT_s010
                                 + 1.3333 w_k1_MSFT_s010 + 1.11753 w_k1_PG_s010
                                 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.07627 w_k1_ADM_s010
                                 + 1.46771 w_k1_AXP_s010
                                 + 1.54154 w_k1_DIS_s010
                                 + 1.23492 w_k1_FDX_s010 + 1.6933 w_k1_IBM_s010
                                 + 1.42284 w_k1_INTC_s010
                                 + 1.45897 w_k1_KO_s010 + 1.41391 w_k1_MAT_s010
                                 + 1.23842 w_k1_MSFT_s010
                                 + 1.07575 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.11454 w_k1_ADM_s011
                                 + 1.21372 w_k1_AXP_s011
                                 + 1.01069 w_k1_DIS_s011 + 1.029 w_k1_FDX_s011
                                 + 1.19401 w_k1_IBM_s011
                                 + 0.878525 w_k1_INTC_s011
                                 + 1.09036 w_k1_KO_s011 + 1.39614 w_k1_MAT_s011
                                 + 1.05731 w_k1_MSFT_s011
                                 + 1.11562 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.31201 w_k1_ADM_s011
                                 + 0.859134 w_k1_AXP_s011
                                 + 1.0125 w_k1_DIS_s011 + 1.034 w_k1_FDX_s011
                                 + 0.783152 w_k1_IBM_s011
                                 + 0.861572 w_k1_INTC_s011
                                 + 0.896583 w_k1_KO_s011
                                 + 0.78021 w_k1_MAT_s011
                                 + 1.03461 w_k1_MSFT_s011
                                 + 0.999709 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.930131 w_k1_ADM_s011
                                 + 0.977106 w_k1_AXP_s011
                                 + 0.818719 w_k1_DIS_s011
                                 + 0.910559 w_k1_FDX_s011
                                 + 0.742394 w_k1_IBM_s011
                                 + 0.919061 w_k1_INTC_s011
                                 + 0.774833 w_k1_KO_s011
                                 + 0.96182 w_k1_MAT_s011
                                 + 0.968395 w_k1_MSFT_s011
                                 + 0.964093 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.34715 w_k1_ADM_s011
                                 + 1.45804 w_k1_AXP_s011
                                 + 1.65329 w_k1_DIS_s011
                                 + 1.27611 w_k1_FDX_s011
                                 + 1.06254 w_k1_IBM_s011
                                 + 0.954851 w_k1_INTC_s011
                                 + 1.2265 w_k1_KO_s011 + 1.23495 w_k1_MAT_s011
                                 + 1.24393 w_k1_MSFT_s011
                                 + 1.16258 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.23716 w_k1_ADM_s011
                                 + 1.22759 w_k1_AXP_s011
                                 + 1.17418 w_k1_DIS_s011
                                 + 1.26528 w_k1_FDX_s011
                                 + 0.893868 w_k1_IBM_s011
                                 + 0.874963 w_k1_INTC_s011
                                 + 0.924064 w_k1_KO_s011
                                 + 1.12514 w_k1_MAT_s011
                                 + 1.00562 w_k1_MSFT_s011 + 1.0606 w_k1_PG_s011
                                 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.07805 w_k1_ADM_s011
                                 + 0.898818 w_k1_AXP_s011
                                 + 1.34089 w_k1_DIS_s011
                                 + 1.05384 w_k1_FDX_s011
                                 + 0.895087 w_k1_IBM_s011
                                 + 0.963701 w_k1_INTC_s011
                                 + 0.961577 w_k1_KO_s011
                                 + 0.820966 w_k1_MAT_s011
                                 + 1.1926 w_k1_MSFT_s011 + 1.08325 w_k1_PG_s011
                                 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.03886 w_k1_ADM_s011
                                 + 0.755745 w_k1_AXP_s011
                                 + 0.608731 w_k1_DIS_s011
                                 + 0.762101 w_k1_FDX_s011
                                 + 1.01586 w_k1_IBM_s011
                                 + 0.676443 w_k1_INTC_s011
                                 + 0.846579 w_k1_KO_s011 + 1.0563 w_k1_MAT_s011
                                 + 0.795987 w_k1_MSFT_s011
                                 + 0.902319 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.12517 w_k1_ADM_s011
                                 + 1.40714 w_k1_AXP_s011 + 1.7358 w_k1_DIS_s011
                                 + 1.13949 w_k1_FDX_s011
                                 + 1.50634 w_k1_IBM_s011
                                 + 0.948506 w_k1_INTC_s011
                                 + 1.28693 w_k1_KO_s011 + 1.79972 w_k1_MAT_s011
                                 + 1.31398 w_k1_MSFT_s011
                                 + 1.26234 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.969651 w_k1_ADM_s012
                                 + 1.06291 w_k1_AXP_s012
                                 + 0.811074 w_k1_DIS_s012
                                 + 1.05618 w_k1_FDX_s012
                                 + 0.659258 w_k1_IBM_s012
                                 + 1.06629 w_k1_INTC_s012
                                 + 0.58118 w_k1_KO_s012
                                 + 0.645884 w_k1_MAT_s012
                                 + 0.891706 w_k1_MSFT_s012
                                 + 0.934398 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.1285 w_k1_ADM_s012
                                 + 1.3476 w_k1_AXP_s012 + 1.70941 w_k1_DIS_s012
                                 + 1.23423 w_k1_FDX_s012
                                 + 1.14496 w_k1_IBM_s012
                                 + 1.17991 w_k1_INTC_s012
                                 + 1.14618 w_k1_KO_s012 + 1.14303 w_k1_MAT_s012
                                 + 1.22488 w_k1_MSFT_s012
                                 + 1.07533 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.976175 w_k1_ADM_s012
                                 + 1.12943 w_k1_AXP_s012 + 1.2674 w_k1_DIS_s012
                                 + 0.983778 w_k1_FDX_s012
                                 + 1.42787 w_k1_IBM_s012
                                 + 0.909618 w_k1_INTC_s012
                                 + 0.975007 w_k1_KO_s012
                                 + 0.983371 w_k1_MAT_s012
                                 + 0.97557 w_k1_MSFT_s012
                                 + 1.00079 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.12194 w_k1_ADM_s012
                                 + 1.09616 w_k1_AXP_s012
                                 + 0.828702 w_k1_DIS_s012
                                 + 0.975001 w_k1_FDX_s012
                                 + 1.24861 w_k1_IBM_s012
                                 + 1.09232 w_k1_INTC_s012
                                 + 0.955408 w_k1_KO_s012
                                 + 1.06227 w_k1_MAT_s012
                                 + 0.914588 w_k1_MSFT_s012
                                 + 0.992165 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.39412 w_k1_ADM_s012
                                 + 1.55982 w_k1_AXP_s012
                                 + 1.00136 w_k1_DIS_s012
                                 + 1.08318 w_k1_FDX_s012
                                 + 1.06151 w_k1_IBM_s012
                                 + 0.785102 w_k1_INTC_s012
                                 + 0.906931 w_k1_KO_s012
                                 + 1.39521 w_k1_MAT_s012
                                 + 0.868951 w_k1_MSFT_s012
                                 + 0.966335 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.991143 w_k1_ADM_s012
                                 + 0.79845 w_k1_AXP_s012
                                 + 1.18176 w_k1_DIS_s012
                                 + 1.01523 w_k1_FDX_s012
                                 + 0.990813 w_k1_IBM_s012
                                 + 0.993197 w_k1_INTC_s012
                                 + 0.958669 w_k1_KO_s012
                                 + 0.635806 w_k1_MAT_s012
                                 + 1.01305 w_k1_MSFT_s012
                                 + 1.08186 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.690147 w_k1_ADM_s012
                                 + 0.749362 w_k1_AXP_s012
                                 + 0.727072 w_k1_DIS_s012
                                 + 0.829248 w_k1_FDX_s012
                                 + 0.816339 w_k1_IBM_s012
                                 + 0.961064 w_k1_INTC_s012
                                 + 0.732765 w_k1_KO_s012
                                 + 0.636733 w_k1_MAT_s012
                                 + 0.84361 w_k1_MSFT_s012
                                 + 0.788286 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.25827 w_k1_ADM_s012
                                 + 1.39167 w_k1_AXP_s012
                                 + 1.35503 w_k1_DIS_s012
                                 + 1.35002 w_k1_FDX_s012
                                 + 1.26595 w_k1_IBM_s012
                                 + 1.07715 w_k1_INTC_s012
                                 + 0.997036 w_k1_KO_s012
                                 + 1.35072 w_k1_MAT_s012
                                 + 1.0887 w_k1_MSFT_s012 + 1.12967 w_k1_PG_s012
                                 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.02395 w_k1_ADM_s013
                                 + 0.681962 w_k1_AXP_s013
                                 + 0.686883 w_k1_DIS_s013
                                 + 0.878028 w_k1_FDX_s013
                                 + 0.742106 w_k1_IBM_s013
                                 + 0.921809 w_k1_INTC_s013
                                 + 0.845055 w_k1_KO_s013
                                 + 0.763768 w_k1_MAT_s013
                                 + 0.843161 w_k1_MSFT_s013
                                 + 0.958131 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.835508 w_k1_ADM_s013
                                 + 0.888504 w_k1_AXP_s013
                                 + 0.735989 w_k1_DIS_s013
                                 + 0.885832 w_k1_FDX_s013
                                 + 0.69723 w_k1_IBM_s013
                                 + 0.998499 w_k1_INTC_s013
                                 + 0.75139 w_k1_KO_s013
                                 + 0.669675 w_k1_MAT_s013
                                 + 0.870742 w_k1_MSFT_s013
                                 + 0.837163 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.71333 w_k1_ADM_s013
                                 + 0.577338 w_k1_AXP_s013
                                 + 0.539045 w_k1_DIS_s013
                                 + 0.733371 w_k1_FDX_s013
                                 + 0.627284 w_k1_IBM_s013
                                 + 0.858696 w_k1_INTC_s013
                                 + 0.60852 w_k1_KO_s013
                                 + 0.563457 w_k1_MAT_s013
                                 + 0.740004 w_k1_MSFT_s013
                                 + 0.75463 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.984594 w_k1_ADM_s013
                                 + 0.953868 w_k1_AXP_s013
                                 + 0.964422 w_k1_DIS_s013
                                 + 1.02573 w_k1_FDX_s013
                                 + 0.793082 w_k1_IBM_s013
                                 + 0.968372 w_k1_INTC_s013
                                 + 0.906788 w_k1_KO_s013
                                 + 0.996943 w_k1_MAT_s013
                                 + 1.05518 w_k1_MSFT_s013
                                 + 0.971514 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.04443 w_k1_ADM_s013
                                 + 0.981108 w_k1_AXP_s013
                                 + 1.2646 w_k1_DIS_s013 + 1.10368 w_k1_FDX_s013
                                 + 1.02371 w_k1_IBM_s013
                                 + 1.00398 w_k1_INTC_s013 + 1.1357 w_k1_KO_s013
                                 + 1.08177 w_k1_MAT_s013
                                 + 1.14481 w_k1_MSFT_s013
                                 + 1.12262 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.3437 w_k1_ADM_s013
                                 + 1.09275 w_k1_AXP_s013
                                 + 1.58834 w_k1_DIS_s013
                                 + 1.31481 w_k1_FDX_s013
                                 + 1.41305 w_k1_IBM_s013
                                 + 1.23078 w_k1_INTC_s013
                                 + 1.44788 w_k1_KO_s013 + 1.33794 w_k1_MAT_s013
                                 + 1.42467 w_k1_MSFT_s013 + 1.2251 w_k1_PG_s013
                                 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.871208 w_k1_ADM_s013
                                 + 0.744473 w_k1_AXP_s013
                                 + 0.68457 w_k1_DIS_s013
                                 + 0.766004 w_k1_FDX_s013
                                 + 0.988131 w_k1_IBM_s013
                                 + 0.887531 w_k1_INTC_s013
                                 + 0.771276 w_k1_KO_s013
                                 + 0.910781 w_k1_MAT_s013
                                 + 0.778273 w_k1_MSFT_s013
                                 + 0.802619 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.10562 w_k1_ADM_s013
                                 + 1.29222 w_k1_AXP_s013
                                 + 0.917375 w_k1_DIS_s013
                                 + 0.975653 w_k1_FDX_s013
                                 + 1.18204 w_k1_IBM_s013
                                 + 0.939898 w_k1_INTC_s013
                                 + 1.05493 w_k1_KO_s013 + 1.38513 w_k1_MAT_s013
                                 + 0.911441 w_k1_MSFT_s013
                                 + 1.10903 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.30769 w_k1_ADM_s014
                                 + 1.48233 w_k1_AXP_s014
                                 + 1.55174 w_k1_DIS_s014
                                 + 1.44947 w_k1_FDX_s014
                                 + 1.09204 w_k1_IBM_s014
                                 + 1.13738 w_k1_INTC_s014
                                 + 1.19365 w_k1_KO_s014 + 1.11603 w_k1_MAT_s014
                                 + 1.12457 w_k1_MSFT_s014
                                 + 1.16693 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.06232 w_k1_ADM_s014
                                 + 0.690375 w_k1_AXP_s014
                                 + 0.964326 w_k1_DIS_s014
                                 + 0.973476 w_k1_FDX_s014
                                 + 0.874724 w_k1_IBM_s014
                                 + 0.86402 w_k1_INTC_s014
                                 + 0.870958 w_k1_KO_s014
                                 + 0.631175 w_k1_MAT_s014
                                 + 0.931245 w_k1_MSFT_s014
                                 + 0.86274 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.14377 w_k1_ADM_s014
                                 + 0.968664 w_k1_AXP_s014
                                 + 1.01257 w_k1_DIS_s014
                                 + 1.03911 w_k1_FDX_s014
                                 + 0.704494 w_k1_IBM_s014
                                 + 0.807966 w_k1_INTC_s014
                                 + 0.858771 w_k1_KO_s014
                                 + 0.73476 w_k1_MAT_s014
                                 + 0.895004 w_k1_MSFT_s014
                                 + 0.887557 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.52181 w_k1_ADM_s014
                                 + 1.31434 w_k1_AXP_s014
                                 + 1.31425 w_k1_DIS_s014
                                 + 1.20743 w_k1_FDX_s014
                                 + 1.55865 w_k1_IBM_s014
                                 + 1.02087 w_k1_INTC_s014
                                 + 1.32798 w_k1_KO_s014 + 1.24794 w_k1_MAT_s014
                                 + 1.15622 w_k1_MSFT_s014
                                 + 1.07909 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.22658 w_k1_ADM_s014
                                 + 0.94065 w_k1_AXP_s014
                                 + 1.39441 w_k1_DIS_s014
                                 + 1.20331 w_k1_FDX_s014
                                 + 0.92699 w_k1_IBM_s014
                                 + 1.05766 w_k1_INTC_s014
                                 + 1.11319 w_k1_KO_s014
                                 + 0.772043 w_k1_MAT_s014
                                 + 1.32177 w_k1_MSFT_s014 + 1.0843 w_k1_PG_s014
                                 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.769662 w_k1_ADM_s014
                                 + 1.01545 w_k1_AXP_s014
                                 + 1.20203 w_k1_DIS_s014
                                 + 0.876736 w_k1_FDX_s014
                                 + 1.47532 w_k1_IBM_s014
                                 + 0.891211 w_k1_INTC_s014
                                 + 1.32658 w_k1_KO_s014 + 1.97706 w_k1_MAT_s014
                                 + 1.04497 w_k1_MSFT_s014
                                 + 1.04794 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.837786 w_k1_ADM_s014
                                 + 0.685415 w_k1_AXP_s014
                                 + 0.616222 w_k1_DIS_s014
                                 + 0.81498 w_k1_FDX_s014
                                 + 0.664582 w_k1_IBM_s014
                                 + 0.750217 w_k1_INTC_s014
                                 + 0.759935 w_k1_KO_s014
                                 + 0.596692 w_k1_MAT_s014
                                 + 0.779419 w_k1_MSFT_s014
                                 + 0.741067 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.851608 w_k1_ADM_s014
                                 + 0.900542 w_k1_AXP_s014
                                 + 0.784256 w_k1_DIS_s014
                                 + 0.893328 w_k1_FDX_s014
                                 + 1.05605 w_k1_IBM_s014
                                 + 0.823484 w_k1_INTC_s014
                                 + 0.890024 w_k1_KO_s014
                                 + 1.09864 w_k1_MAT_s014
                                 + 0.965442 w_k1_MSFT_s014
                                 + 0.885426 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.14435 w_k1_ADM_s015
                                 + 1.05094 w_k1_AXP_s015
                                 + 1.14979 w_k1_DIS_s015
                                 + 1.04272 w_k1_FDX_s015
                                 + 1.87415 w_k1_IBM_s015
                                 + 1.11759 w_k1_INTC_s015
                                 + 1.55694 w_k1_KO_s015 + 2.22635 w_k1_MAT_s015
                                 + 1.10638 w_k1_MSFT_s015
                                 + 1.25436 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.905139 w_k1_ADM_s015
                                 + 0.724566 w_k1_AXP_s015
                                 + 0.573642 w_k1_DIS_s015
                                 + 0.743964 w_k1_FDX_s015
                                 + 0.699239 w_k1_IBM_s015
                                 + 0.799002 w_k1_INTC_s015
                                 + 0.734814 w_k1_KO_s015
                                 + 0.670441 w_k1_MAT_s015
                                 + 0.703794 w_k1_MSFT_s015
                                 + 0.809861 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.19695 w_k1_ADM_s015
                                 + 0.736605 w_k1_AXP_s015
                                 + 0.707462 w_k1_DIS_s015
                                 + 0.777822 w_k1_FDX_s015
                                 + 1.37179 w_k1_IBM_s015
                                 + 0.856103 w_k1_INTC_s015
                                 + 1.04539 w_k1_KO_s015 + 1.41091 w_k1_MAT_s015
                                 + 0.77655 w_k1_MSFT_s015
                                 + 0.967293 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.12137 w_k1_ADM_s015
                                 + 1.15914 w_k1_AXP_s015
                                 + 1.07498 w_k1_DIS_s015 + 1.1233 w_k1_FDX_s015
                                 + 1.34622 w_k1_IBM_s015
                                 + 1.26567 w_k1_INTC_s015
                                 + 1.31654 w_k1_KO_s015 + 1.48811 w_k1_MAT_s015
                                 + 1.03202 w_k1_MSFT_s015
                                 + 1.32481 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.50245 w_k1_ADM_s015
                                 + 1.17426 w_k1_AXP_s015
                                 + 1.31105 w_k1_DIS_s015
                                 + 1.33814 w_k1_FDX_s015
                                 + 0.841719 w_k1_IBM_s015
                                 + 1.34168 w_k1_INTC_s015
                                 + 1.12568 w_k1_KO_s015
                                 + 0.886702 w_k1_MAT_s015
                                 + 1.17483 w_k1_MSFT_s015
                                 + 1.27935 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.894918 w_k1_ADM_s015
                                 + 0.728403 w_k1_AXP_s015
                                 + 0.816233 w_k1_DIS_s015
                                 + 0.83598 w_k1_FDX_s015
                                 + 0.996765 w_k1_IBM_s015
                                 + 0.937777 w_k1_INTC_s015
                                 + 0.953851 w_k1_KO_s015
                                 + 1.01874 w_k1_MAT_s015
                                 + 0.795332 w_k1_MSFT_s015
                                 + 1.02524 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.16235 w_k1_ADM_s015
                                 + 0.864857 w_k1_AXP_s015
                                 + 0.893352 w_k1_DIS_s015
                                 + 0.934207 w_k1_FDX_s015
                                 + 0.769792 w_k1_IBM_s015
                                 + 1.00076 w_k1_INTC_s015
                                 + 0.951785 w_k1_KO_s015
                                 + 0.861968 w_k1_MAT_s015
                                 + 0.79114 w_k1_MSFT_s015
                                 + 1.00064 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.37294 w_k1_ADM_s015
                                 + 1.43533 w_k1_AXP_s015 + 1.8334 w_k1_DIS_s015
                                 + 1.45378 w_k1_FDX_s015
                                 + 1.23889 w_k1_IBM_s015
                                 + 1.4727 w_k1_INTC_s015 + 1.48252 w_k1_KO_s015
                                 + 1.29954 w_k1_MAT_s015
                                 + 1.33288 w_k1_MSFT_s015
                                 + 1.11779 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.01227 w_k1_ADM_s016
                                 + 1.22276 w_k1_AXP_s016
                                 + 1.35095 w_k1_DIS_s016
                                 + 1.22338 w_k1_FDX_s016
                                 + 1.14967 w_k1_IBM_s016
                                 + 1.19433 w_k1_INTC_s016
                                 + 1.16321 w_k1_KO_s016 + 1.66173 w_k1_MAT_s016
                                 + 1.14577 w_k1_MSFT_s016
                                 + 1.11683 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.886087 w_k1_ADM_s016
                                 + 1.05117 w_k1_AXP_s016
                                 + 1.30287 w_k1_DIS_s016
                                 + 1.23721 w_k1_FDX_s016
                                 + 0.797539 w_k1_IBM_s016
                                 + 1.22462 w_k1_INTC_s016
                                 + 1.20802 w_k1_KO_s016
                                 + 0.803666 w_k1_MAT_s016
                                 + 1.31279 w_k1_MSFT_s016 + 1.1476 w_k1_PG_s016
                                 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.890556 w_k1_ADM_s016
                                 + 0.674206 w_k1_AXP_s016
                                 + 1.04308 w_k1_DIS_s016
                                 + 0.939897 w_k1_FDX_s016
                                 + 0.865055 w_k1_IBM_s016
                                 + 1.05142 w_k1_INTC_s016
                                 + 1.00013 w_k1_KO_s016
                                 + 0.703186 w_k1_MAT_s016
                                 + 1.06337 w_k1_MSFT_s016
                                 + 0.920743 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.616055 w_k1_ADM_s016
                                 + 1.03141 w_k1_AXP_s016
                                 + 0.644432 w_k1_DIS_s016
                                 + 0.744802 w_k1_FDX_s016
                                 + 1.29033 w_k1_IBM_s016
                                 + 1.07323 w_k1_INTC_s016
                                 + 0.891567 w_k1_KO_s016
                                 + 1.01087 w_k1_MAT_s016
                                 + 0.798458 w_k1_MSFT_s016
                                 + 0.884361 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.915258 w_k1_ADM_s016
                                 + 1.07824 w_k1_AXP_s016
                                 + 0.997239 w_k1_DIS_s016
                                 + 0.953286 w_k1_FDX_s016
                                 + 1.02532 w_k1_IBM_s016
                                 + 1.13063 w_k1_INTC_s016 + 1.1247 w_k1_KO_s016
                                 + 1.08514 w_k1_MAT_s016
                                 + 1.13568 w_k1_MSFT_s016
                                 + 1.10732 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.955251 w_k1_ADM_s016
                                 + 0.823308 w_k1_AXP_s016
                                 + 1.11995 w_k1_DIS_s016
                                 + 0.911981 w_k1_FDX_s016
                                 + 1.36798 w_k1_IBM_s016
                                 + 0.918144 w_k1_INTC_s016
                                 + 1.24642 w_k1_KO_s016 + 1.27286 w_k1_MAT_s016
                                 + 1.06362 w_k1_MSFT_s016
                                 + 1.09552 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.844158 w_k1_ADM_s016
                                 + 0.696733 w_k1_AXP_s016
                                 + 0.640305 w_k1_DIS_s016
                                 + 0.729735 w_k1_FDX_s016
                                 + 0.803222 w_k1_IBM_s016
                                 + 0.879897 w_k1_INTC_s016
                                 + 0.774217 w_k1_KO_s016
                                 + 0.672153 w_k1_MAT_s016
                                 + 0.805334 w_k1_MSFT_s016
                                 + 0.852762 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.865138 w_k1_ADM_s016
                                 + 0.757296 w_k1_AXP_s016
                                 + 0.744539 w_k1_DIS_s016
                                 + 0.855345 w_k1_FDX_s016
                                 + 0.585816 w_k1_IBM_s016
                                 + 0.941567 w_k1_INTC_s016
                                 + 0.742118 w_k1_KO_s016
                                 + 0.558134 w_k1_MAT_s016
                                 + 0.922112 w_k1_MSFT_s016
                                 + 0.799295 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.892509 w_k1_ADM_s017
                                 + 0.775398 w_k1_AXP_s017
                                 + 0.671808 w_k1_DIS_s017
                                 + 0.82805 w_k1_FDX_s017
                                 + 0.808899 w_k1_IBM_s017
                                 + 0.893063 w_k1_INTC_s017
                                 + 0.890359 w_k1_KO_s017
                                 + 0.695079 w_k1_MAT_s017
                                 + 0.893858 w_k1_MSFT_s017
                                 + 0.883951 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.8632 w_k1_ADM_s017
                                 + 0.885145 w_k1_AXP_s017
                                 + 0.828491 w_k1_DIS_s017
                                 + 0.902011 w_k1_FDX_s017
                                 + 1.16566 w_k1_IBM_s017
                                 + 0.813342 w_k1_INTC_s017
                                 + 1.02591 w_k1_KO_s017 + 1.09056 w_k1_MAT_s017
                                 + 0.910225 w_k1_MSFT_s017
                                 + 1.02267 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.06722 w_k1_ADM_s017
                                 + 1.03186 w_k1_AXP_s017
                                 + 0.931488 w_k1_DIS_s017
                                 + 0.995063 w_k1_FDX_s017
                                 + 1.3014 w_k1_IBM_s017
                                 + 0.961659 w_k1_INTC_s017
                                 + 1.11357 w_k1_KO_s017 + 1.24634 w_k1_MAT_s017
                                 + 1.01437 w_k1_MSFT_s017
                                 + 1.15502 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.39524 w_k1_ADM_s017
                                 + 1.32411 w_k1_AXP_s017
                                 + 1.54708 w_k1_DIS_s017
                                 + 1.22513 w_k1_FDX_s017
                                 + 1.48489 w_k1_IBM_s017
                                 + 1.19074 w_k1_INTC_s017
                                 + 1.51549 w_k1_KO_s017 + 1.5942 w_k1_MAT_s017
                                 + 1.28915 w_k1_MSFT_s017
                                 + 1.17727 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.881032 w_k1_ADM_s017
                                 + 0.750155 w_k1_AXP_s017
                                 + 0.617008 w_k1_DIS_s017
                                 + 0.852373 w_k1_FDX_s017
                                 + 0.577436 w_k1_IBM_s017
                                 + 0.959929 w_k1_INTC_s017
                                 + 0.704605 w_k1_KO_s017
                                 + 0.620151 w_k1_MAT_s017
                                 + 0.91515 w_k1_MSFT_s017
                                 + 0.875717 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.3113 w_k1_ADM_s017
                                 + 1.05782 w_k1_AXP_s017
                                 + 1.29631 w_k1_DIS_s017
                                 + 1.12585 w_k1_FDX_s017
                                 + 1.13935 w_k1_IBM_s017
                                 + 1.21809 w_k1_INTC_s017
                                 + 1.22371 w_k1_KO_s017
                                 + 0.988074 w_k1_MAT_s017
                                 + 1.28181 w_k1_MSFT_s017
                                 + 1.15695 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.933252 w_k1_ADM_s017
                                 + 0.779045 w_k1_AXP_s017
                                 + 0.945111 w_k1_DIS_s017
                                 + 0.960625 w_k1_FDX_s017
                                 + 0.782703 w_k1_IBM_s017
                                 + 0.877113 w_k1_INTC_s017
                                 + 0.854131 w_k1_KO_s017
                                 + 0.887635 w_k1_MAT_s017
                                 + 1.00251 w_k1_MSFT_s017
                                 + 0.927588 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.13429 w_k1_ADM_s017
                                 + 1.18629 w_k1_AXP_s017
                                 + 1.28205 w_k1_DIS_s017
                                 + 1.28392 w_k1_FDX_s017
                                 + 0.96188 w_k1_IBM_s017
                                 + 1.2444 w_k1_INTC_s017 + 1.19107 w_k1_KO_s017
                                 + 1.08709 w_k1_MAT_s017
                                 + 1.37802 w_k1_MSFT_s017
                                 + 1.16934 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.847707 w_k1_ADM_s100
                                 + 0.780876 w_k1_AXP_s100
                                 + 0.773664 w_k1_DIS_s100
                                 + 0.83706 w_k1_FDX_s100
                                 + 0.618349 w_k1_IBM_s100
                                 + 1.11587 w_k1_INTC_s100
                                 + 0.73973 w_k1_KO_s100
                                 + 0.493308 w_k1_MAT_s100
                                 + 1.02744 w_k1_MSFT_s100
                                 + 0.792211 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.06591 w_k1_ADM_s100
                                 + 1.07599 w_k1_AXP_s100
                                 + 1.09667 w_k1_DIS_s100
                                 + 1.04583 w_k1_FDX_s100
                                 + 0.70132 w_k1_IBM_s100
                                 + 1.34231 w_k1_INTC_s100
                                 + 0.913574 w_k1_KO_s100
                                 + 0.891699 w_k1_MAT_s100
                                 + 1.31181 w_k1_MSFT_s100 + 1.0533 w_k1_PG_s100
                                 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.990344 w_k1_ADM_s100
                                 + 1.3904 w_k1_AXP_s100 + 1.15347 w_k1_DIS_s100
                                 + 1.27471 w_k1_FDX_s100
                                 + 0.688089 w_k1_IBM_s100
                                 + 1.32228 w_k1_INTC_s100
                                 + 0.918212 w_k1_KO_s100
                                 + 0.989441 w_k1_MAT_s100
                                 + 1.26825 w_k1_MSFT_s100
                                 + 0.944538 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.833084 w_k1_ADM_s100
                                 + 1.18383 w_k1_AXP_s100
                                 + 1.09274 w_k1_DIS_s100
                                 + 1.10229 w_k1_FDX_s100
                                 + 0.858627 w_k1_IBM_s100
                                 + 1.48284 w_k1_INTC_s100
                                 + 1.05733 w_k1_KO_s100
                                 + 0.834443 w_k1_MAT_s100
                                 + 1.22262 w_k1_MSFT_s100
                                 + 0.979891 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.5632 w_k1_ADM_s100
                                 + 1.62144 w_k1_AXP_s100
                                 + 1.49654 w_k1_DIS_s100
                                 + 1.41425 w_k1_FDX_s100
                                 + 1.09538 w_k1_IBM_s100
                                 + 1.42185 w_k1_INTC_s100
                                 + 1.25306 w_k1_KO_s100 + 1.39383 w_k1_MAT_s100
                                 + 1.38219 w_k1_MSFT_s100
                                 + 1.27001 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.89885 w_k1_ADM_s100
                                 + 1.05497 w_k1_AXP_s100
                                 + 1.11148 w_k1_DIS_s100
                                 + 0.958864 w_k1_FDX_s100
                                 + 1.05459 w_k1_IBM_s100
                                 + 1.02756 w_k1_INTC_s100
                                 + 1.05668 w_k1_KO_s100
                                 + 0.955293 w_k1_MAT_s100
                                 + 1.01258 w_k1_MSFT_s100
                                 + 1.02407 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.05728 w_k1_ADM_s100
                                 + 1.50565 w_k1_AXP_s100
                                 + 1.42504 w_k1_DIS_s100
                                 + 1.15315 w_k1_FDX_s100
                                 + 1.62659 w_k1_IBM_s100
                                 + 1.0857 w_k1_INTC_s100 + 1.35716 w_k1_KO_s100
                                 + 1.9273 w_k1_MAT_s100
                                 + 1.11163 w_k1_MSFT_s100
                                 + 1.25874 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.696707 w_k1_ADM_s100
                                 + 0.772881 w_k1_AXP_s100
                                 + 0.675252 w_k1_DIS_s100
                                 + 0.735062 w_k1_FDX_s100
                                 + 0.938072 w_k1_IBM_s100
                                 + 1.05171 w_k1_INTC_s100
                                 + 0.81059 w_k1_KO_s100
                                 + 0.930032 w_k1_MAT_s100
                                 + 0.844939 w_k1_MSFT_s100
                                 + 0.829701 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.868199 w_k1_ADM_s101
                                 + 1.02649 w_k1_AXP_s101
                                 + 1.00918 w_k1_DIS_s101
                                 + 0.912839 w_k1_FDX_s101
                                 + 1.5681 w_k1_IBM_s101
                                 + 0.999878 w_k1_INTC_s101
                                 + 1.26841 w_k1_KO_s101 + 1.58715 w_k1_MAT_s101
                                 + 1.03354 w_k1_MSFT_s101
                                 + 1.00646 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.13251 w_k1_ADM_s101
                                 + 1.03456 w_k1_AXP_s101
                                 + 1.17409 w_k1_DIS_s101
                                 + 1.13788 w_k1_FDX_s101
                                 + 0.945465 w_k1_IBM_s101
                                 + 0.939477 w_k1_INTC_s101
                                 + 0.998858 w_k1_KO_s101
                                 + 0.881347 w_k1_MAT_s101
                                 + 1.14242 w_k1_MSFT_s101
                                 + 0.96975 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.895809 w_k1_ADM_s101
                                 + 0.798077 w_k1_AXP_s101
                                 + 0.780902 w_k1_DIS_s101
                                 + 0.800322 w_k1_FDX_s101
                                 + 0.992747 w_k1_IBM_s101
                                 + 0.801544 w_k1_INTC_s101
                                 + 0.874631 w_k1_KO_s101
                                 + 0.933685 w_k1_MAT_s101
                                 + 0.872188 w_k1_MSFT_s101
                                 + 0.924373 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.942682 w_k1_ADM_s101
                                 + 0.911046 w_k1_AXP_s101
                                 + 0.710751 w_k1_DIS_s101
                                 + 0.860681 w_k1_FDX_s101
                                 + 0.695003 w_k1_IBM_s101
                                 + 0.950305 w_k1_INTC_s101
                                 + 0.762421 w_k1_KO_s101
                                 + 0.778411 w_k1_MAT_s101
                                 + 0.87644 w_k1_MSFT_s101
                                 + 0.872343 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.10686 w_k1_ADM_s101
                                 + 1.31111 w_k1_AXP_s101
                                 + 1.37732 w_k1_DIS_s101
                                 + 1.34328 w_k1_FDX_s101
                                 + 0.903697 w_k1_IBM_s101
                                 + 0.908196 w_k1_INTC_s101
                                 + 0.974548 w_k1_KO_s101
                                 + 0.905794 w_k1_MAT_s101
                                 + 1.21482 w_k1_MSFT_s101 + 1.0864 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.973983 w_k1_ADM_s101
                                 + 1.01781 w_k1_AXP_s101
                                 + 0.900871 w_k1_DIS_s101
                                 + 0.896386 w_k1_FDX_s101
                                 + 1.20742 w_k1_IBM_s101
                                 + 0.760129 w_k1_INTC_s101
                                 + 0.917614 w_k1_KO_s101 + 1.2879 w_k1_MAT_s101
                                 + 0.879551 w_k1_MSFT_s101
                                 + 0.995561 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.59895 w_k1_ADM_s101
                                 + 1.55251 w_k1_AXP_s101
                                 + 1.66314 w_k1_DIS_s101
                                 + 1.38966 w_k1_FDX_s101
                                 + 1.42449 w_k1_IBM_s101
                                 + 1.10124 w_k1_INTC_s101
                                 + 1.45587 w_k1_KO_s101 + 1.64866 w_k1_MAT_s101
                                 + 1.49037 w_k1_MSFT_s101
                                 + 1.40216 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.851638 w_k1_ADM_s101
                                 + 1.20017 w_k1_AXP_s101
                                 + 1.34307 w_k1_DIS_s101
                                 + 1.10444 w_k1_FDX_s101
                                 + 1.23934 w_k1_IBM_s101
                                 + 1.13408 w_k1_INTC_s101 + 1.2816 w_k1_KO_s101
                                 + 1.4143 w_k1_MAT_s101
                                 + 1.24507 w_k1_MSFT_s101
                                 + 1.06298 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.128 w_k1_ADM_s102
                                 + 1.80241 w_k1_AXP_s102
                                 + 1.31987 w_k1_DIS_s102
                                 + 1.26417 w_k1_FDX_s102
                                 + 1.19662 w_k1_IBM_s102
                                 + 1.65118 w_k1_INTC_s102
                                 + 1.41474 w_k1_KO_s102 + 1.61689 w_k1_MAT_s102
                                 + 1.3014 w_k1_MSFT_s102 + 1.15593 w_k1_PG_s102
                                 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.13482 w_k1_ADM_s102
                                 + 1.34098 w_k1_AXP_s102
                                 + 1.70526 w_k1_DIS_s102
                                 + 1.32409 w_k1_FDX_s102
                                 + 1.11419 w_k1_IBM_s102
                                 + 1.51182 w_k1_INTC_s102
                                 + 1.59401 w_k1_KO_s102 + 1.53315 w_k1_MAT_s102
                                 + 1.55355 w_k1_MSFT_s102
                                 + 1.25695 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.990774 w_k1_ADM_s102
                                 + 0.904707 w_k1_AXP_s102
                                 + 0.743955 w_k1_DIS_s102
                                 + 0.882875 w_k1_FDX_s102
                                 + 0.989482 w_k1_IBM_s102
                                 + 0.858096 w_k1_INTC_s102
                                 + 0.994289 w_k1_KO_s102
                                 + 1.22452 w_k1_MAT_s102
                                 + 0.870492 w_k1_MSFT_s102
                                 + 0.88534 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.05722 w_k1_ADM_s102
                                 + 1.38255 w_k1_AXP_s102
                                 + 1.80862 w_k1_DIS_s102
                                 + 1.23232 w_k1_FDX_s102
                                 + 1.57554 w_k1_IBM_s102
                                 + 1.29486 w_k1_INTC_s102
                                 + 1.89174 w_k1_KO_s102 + 2.16643 w_k1_MAT_s102
                                 + 1.42959 w_k1_MSFT_s102
                                 + 1.10405 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.674078 w_k1_ADM_s102
                                 + 0.544118 w_k1_AXP_s102
                                 + 0.686146 w_k1_DIS_s102
                                 + 0.759959 w_k1_FDX_s102
                                 + 0.645201 w_k1_IBM_s102
                                 + 0.993805 w_k1_INTC_s102
                                 + 0.861409 w_k1_KO_s102
                                 + 0.407376 w_k1_MAT_s102
                                 + 0.942606 w_k1_MSFT_s102
                                 + 0.811243 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.94416 w_k1_ADM_s102
                                 + 0.905029 w_k1_AXP_s102
                                 + 0.831325 w_k1_DIS_s102
                                 + 1.05845 w_k1_FDX_s102
                                 + 0.636375 w_k1_IBM_s102
                                 + 1.00566 w_k1_INTC_s102
                                 + 0.946843 w_k1_KO_s102
                                 + 0.910315 w_k1_MAT_s102
                                 + 1.1271 w_k1_MSFT_s102
                                 + 0.990598 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.02033 w_k1_ADM_s102
                                 + 0.895199 w_k1_AXP_s102
                                 + 1.35589 w_k1_DIS_s102
                                 + 1.17768 w_k1_FDX_s102
                                 + 0.897103 w_k1_IBM_s102
                                 + 1.0564 w_k1_INTC_s102 + 1.25439 w_k1_KO_s102
                                 + 0.964782 w_k1_MAT_s102
                                 + 1.32187 w_k1_MSFT_s102
                                 + 1.04637 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.948538 w_k1_ADM_s102
                                 + 0.973073 w_k1_AXP_s102
                                 + 1.0199 w_k1_DIS_s102 + 0.90457 w_k1_FDX_s102
                                 + 1.32212 w_k1_IBM_s102
                                 + 1.09096 w_k1_INTC_s102
                                 + 1.23665 w_k1_KO_s102 + 1.28285 w_k1_MAT_s102
                                 + 1.11615 w_k1_MSFT_s102
                                 + 0.931077 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.07017 w_k1_ADM_s103
                                 + 0.911408 w_k1_AXP_s103
                                 + 1.01058 w_k1_DIS_s103
                                 + 0.985183 w_k1_FDX_s103
                                 + 0.969526 w_k1_IBM_s103
                                 + 1.03741 w_k1_INTC_s103
                                 + 1.07149 w_k1_KO_s103
                                 + 0.618586 w_k1_MAT_s103
                                 + 0.937619 w_k1_MSFT_s103
                                 + 0.99511 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.62029 w_k1_ADM_s103
                                 + 1.73108 w_k1_AXP_s103
                                 + 1.57524 w_k1_DIS_s103
                                 + 1.42278 w_k1_FDX_s103
                                 + 1.03574 w_k1_IBM_s103
                                 + 1.31302 w_k1_INTC_s103
                                 + 1.28047 w_k1_KO_s103 + 1.52793 w_k1_MAT_s103
                                 + 1.26586 w_k1_MSFT_s103
                                 + 1.24078 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.773344 w_k1_ADM_s103
                                 + 0.924499 w_k1_AXP_s103
                                 + 0.792836 w_k1_DIS_s103
                                 + 0.861885 w_k1_FDX_s103
                                 + 1.15651 w_k1_IBM_s103
                                 + 0.901834 w_k1_INTC_s103
                                 + 0.984135 w_k1_KO_s103
                                 + 1.04169 w_k1_MAT_s103
                                 + 0.865204 w_k1_MSFT_s103
                                 + 0.921943 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.878991 w_k1_ADM_s103
                                 + 0.892063 w_k1_AXP_s103
                                 + 0.593861 w_k1_DIS_s103
                                 + 0.772277 w_k1_FDX_s103
                                 + 0.818077 w_k1_IBM_s103
                                 + 0.841788 w_k1_INTC_s103
                                 + 0.797076 w_k1_KO_s103
                                 + 0.694074 w_k1_MAT_s103
                                 + 0.693548 w_k1_MSFT_s103
                                 + 0.821684 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.07281 w_k1_ADM_s103
                                 + 1.03049 w_k1_AXP_s103
                                 + 1.54644 w_k1_DIS_s103
                                 + 1.20294 w_k1_FDX_s103
                                 + 0.892268 w_k1_IBM_s103
                                 + 1.33712 w_k1_INTC_s103 + 1.0103 w_k1_KO_s103
                                 + 0.59939 w_k1_MAT_s103
                                 + 1.24499 w_k1_MSFT_s103
                                 + 0.955476 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.07582 w_k1_ADM_s103
                                 + 0.782002 w_k1_AXP_s103
                                 + 1.06564 w_k1_DIS_s103
                                 + 1.08595 w_k1_FDX_s103
                                 + 0.723733 w_k1_IBM_s103
                                 + 0.997851 w_k1_INTC_s103
                                 + 1.05484 w_k1_KO_s103
                                 + 0.832645 w_k1_MAT_s103
                                 + 1.14152 w_k1_MSFT_s103
                                 + 0.990285 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.11213 w_k1_ADM_s103
                                 + 1.57264 w_k1_AXP_s103
                                 + 1.78875 w_k1_DIS_s103
                                 + 1.24921 w_k1_FDX_s103
                                 + 1.47312 w_k1_IBM_s103
                                 + 1.11909 w_k1_INTC_s103
                                 + 1.51706 w_k1_KO_s103 + 1.80819 w_k1_MAT_s103
                                 + 1.33832 w_k1_MSFT_s103
                                 + 1.25517 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.14907 w_k1_ADM_s103
                                 + 1.25367 w_k1_AXP_s103
                                 + 1.25035 w_k1_DIS_s103
                                 + 1.15295 w_k1_FDX_s103
                                 + 1.36418 w_k1_IBM_s103
                                 + 0.92143 w_k1_INTC_s103
                                 + 1.13257 w_k1_KO_s103 + 1.3345 w_k1_MAT_s103
                                 + 1.09078 w_k1_MSFT_s103
                                 + 0.984091 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.24083 w_k1_ADM_s104
                                 + 1.61596 w_k1_AXP_s104
                                 + 1.50235 w_k1_DIS_s104
                                 + 1.44855 w_k1_FDX_s104
                                 + 1.15243 w_k1_IBM_s104
                                 + 1.2925 w_k1_INTC_s104 + 1.20405 w_k1_KO_s104
                                 + 1.73203 w_k1_MAT_s104 + 1.386 w_k1_MSFT_s104
                                 + 1.18826 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.2351 w_k1_ADM_s104
                                 + 1.06188 w_k1_AXP_s104
                                 + 1.19478 w_k1_DIS_s104
                                 + 1.18574 w_k1_FDX_s104
                                 + 0.763804 w_k1_IBM_s104
                                 + 1.02819 w_k1_INTC_s104
                                 + 0.97791 w_k1_KO_s104
                                 + 0.928305 w_k1_MAT_s104
                                 + 1.17189 w_k1_MSFT_s104
                                 + 1.05245 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.908714 w_k1_ADM_s104
                                 + 0.895924 w_k1_AXP_s104
                                 + 0.809442 w_k1_DIS_s104
                                 + 0.892209 w_k1_FDX_s104
                                 + 0.896612 w_k1_IBM_s104
                                 + 0.87621 w_k1_INTC_s104
                                 + 0.805761 w_k1_KO_s104
                                 + 0.763649 w_k1_MAT_s104
                                 + 0.959889 w_k1_MSFT_s104
                                 + 0.929666 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.778645 w_k1_ADM_s104
                                 + 0.784356 w_k1_AXP_s104
                                 + 0.743496 w_k1_DIS_s104
                                 + 0.874208 w_k1_FDX_s104
                                 + 0.635333 w_k1_IBM_s104
                                 + 0.807693 w_k1_INTC_s104
                                 + 0.770517 w_k1_KO_s104 + 0.7317 w_k1_MAT_s104
                                 + 0.964019 w_k1_MSFT_s104
                                 + 0.887292 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.12667 w_k1_ADM_s104
                                 + 0.998262 w_k1_AXP_s104
                                 + 1.15189 w_k1_DIS_s104
                                 + 1.04087 w_k1_FDX_s104
                                 + 1.02389 w_k1_IBM_s104
                                 + 0.985214 w_k1_INTC_s104
                                 + 1.04147 w_k1_KO_s104 + 1.35019 w_k1_MAT_s104
                                 + 1.16536 w_k1_MSFT_s104
                                 + 0.988705 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.23549 w_k1_ADM_s104
                                 + 1.34999 w_k1_AXP_s104
                                 + 0.943442 w_k1_DIS_s104
                                 + 1.05423 w_k1_FDX_s104
                                 + 0.740934 w_k1_IBM_s104
                                 + 0.956553 w_k1_INTC_s104
                                 + 0.817827 w_k1_KO_s104
                                 + 1.08168 w_k1_MAT_s104
                                 + 0.941916 w_k1_MSFT_s104
                                 + 0.912259 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.923835 w_k1_ADM_s104
                                 + 0.876524 w_k1_AXP_s104
                                 + 0.964126 w_k1_DIS_s104
                                 + 0.889938 w_k1_FDX_s104
                                 + 1.28317 w_k1_IBM_s104
                                 + 0.78736 w_k1_INTC_s104
                                 + 1.17026 w_k1_KO_s104 + 1.43623 w_k1_MAT_s104
                                 + 0.921182 w_k1_MSFT_s104
                                 + 0.903891 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.780016 w_k1_ADM_s104
                                 + 0.69703 w_k1_AXP_s104
                                 + 0.553769 w_k1_DIS_s104
                                 + 0.713643 w_k1_FDX_s104
                                 + 0.840483 w_k1_IBM_s104
                                 + 0.84387 w_k1_INTC_s104
                                 + 0.777551 w_k1_KO_s104
                                 + 0.956649 w_k1_MAT_s104
                                 + 0.870747 w_k1_MSFT_s104
                                 + 0.781725 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.18667 w_k1_ADM_s105
                                 + 1.1905 w_k1_AXP_s105 + 1.27004 w_k1_DIS_s105
                                 + 1.10633 w_k1_FDX_s105
                                 + 1.30257 w_k1_IBM_s105
                                 + 0.950535 w_k1_INTC_s105
                                 + 1.26689 w_k1_KO_s105 + 1.14167 w_k1_MAT_s105
                                 + 1.13709 w_k1_MSFT_s105
                                 + 1.06409 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.20924 w_k1_ADM_s105
                                 + 1.58026 w_k1_AXP_s105
                                 + 1.32356 w_k1_DIS_s105
                                 + 1.27648 w_k1_FDX_s105
                                 + 0.959439 w_k1_IBM_s105
                                 + 1.29261 w_k1_INTC_s105
                                 + 1.13411 w_k1_KO_s105
                                 + 0.895811 w_k1_MAT_s105
                                 + 1.30287 w_k1_MSFT_s105
                                 + 1.07855 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.945082 w_k1_ADM_s105
                                 + 0.745548 w_k1_AXP_s105
                                 + 1.00281 w_k1_DIS_s105
                                 + 0.921034 w_k1_FDX_s105
                                 + 1.04824 w_k1_IBM_s105
                                 + 0.84199 w_k1_INTC_s105
                                 + 1.05837 w_k1_KO_s105 + 1.06286 w_k1_MAT_s105
                                 + 0.965823 w_k1_MSFT_s105
                                 + 0.98086 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.39725 w_k1_ADM_s105
                                 + 1.5115 w_k1_AXP_s105 + 1.84461 w_k1_DIS_s105
                                 + 1.37218 w_k1_FDX_s105
                                 + 1.28611 w_k1_IBM_s105
                                 + 1.22096 w_k1_INTC_s105
                                 + 1.58813 w_k1_KO_s105 + 1.61804 w_k1_MAT_s105
                                 + 1.35848 w_k1_MSFT_s105
                                 + 1.29456 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.949384 w_k1_ADM_s105
                                 + 1.18114 w_k1_AXP_s105
                                 + 0.950728 w_k1_DIS_s105
                                 + 1.03734 w_k1_FDX_s105
                                 + 0.831606 w_k1_IBM_s105
                                 + 0.947393 w_k1_INTC_s105
                                 + 1.00172 w_k1_KO_s105
                                 + 0.936157 w_k1_MAT_s105
                                 + 0.985256 w_k1_MSFT_s105
                                 + 0.996053 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.838171 w_k1_ADM_s105
                                 + 0.842267 w_k1_AXP_s105
                                 + 0.702409 w_k1_DIS_s105
                                 + 0.794163 w_k1_FDX_s105
                                 + 0.765974 w_k1_IBM_s105
                                 + 0.862638 w_k1_INTC_s105
                                 + 0.768279 w_k1_KO_s105
                                 + 0.607369 w_k1_MAT_s105
                                 + 0.836284 w_k1_MSFT_s105
                                 + 0.82031 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.01089 w_k1_ADM_s105
                                 + 1.42863 w_k1_AXP_s105
                                 + 1.22967 w_k1_DIS_s105
                                 + 0.974359 w_k1_FDX_s105
                                 + 1.72786 w_k1_IBM_s105
                                 + 1.05685 w_k1_INTC_s105 + 1.4147 w_k1_KO_s105
                                 + 1.86729 w_k1_MAT_s105
                                 + 1.05697 w_k1_MSFT_s105
                                 + 1.27648 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.76816 w_k1_ADM_s105
                                 + 0.983306 w_k1_AXP_s105
                                 + 0.804876 w_k1_DIS_s105
                                 + 0.748842 w_k1_FDX_s105
                                 + 1.33179 w_k1_IBM_s105
                                 + 0.871657 w_k1_INTC_s105
                                 + 1.09825 w_k1_KO_s105 + 1.34336 w_k1_MAT_s105
                                 + 0.830348 w_k1_MSFT_s105
                                 + 0.890591 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.0866 w_k1_ADM_s106
                                 + 1.60527 w_k1_AXP_s106
                                 + 1.56838 w_k1_DIS_s106
                                 + 1.38933 w_k1_FDX_s106
                                 + 1.29805 w_k1_IBM_s106
                                 + 1.44604 w_k1_INTC_s106
                                 + 1.52405 w_k1_KO_s106 + 1.73938 w_k1_MAT_s106
                                 + 1.36527 w_k1_MSFT_s106
                                 + 1.36853 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.909538 w_k1_ADM_s106
                                 + 1.06943 w_k1_AXP_s106
                                 + 1.02533 w_k1_DIS_s106
                                 + 0.879235 w_k1_FDX_s106
                                 + 1.44411 w_k1_IBM_s106
                                 + 0.890177 w_k1_INTC_s106
                                 + 1.31186 w_k1_KO_s106 + 2.05183 w_k1_MAT_s106
                                 + 1.00928 w_k1_MSFT_s106
                                 + 1.05304 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.01205 w_k1_ADM_s106
                                 + 1.13574 w_k1_AXP_s106
                                 + 1.21482 w_k1_DIS_s106
                                 + 1.09804 w_k1_FDX_s106
                                 + 0.979979 w_k1_IBM_s106
                                 + 1.11356 w_k1_INTC_s106
                                 + 1.31503 w_k1_KO_s106 + 1.44447 w_k1_MAT_s106
                                 + 1.22895 w_k1_MSFT_s106
                                 + 1.13912 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.03294 w_k1_ADM_s106
                                 + 1.47107 w_k1_AXP_s106
                                 + 1.33192 w_k1_DIS_s106
                                 + 1.14077 w_k1_FDX_s106
                                 + 1.69279 w_k1_IBM_s106
                                 + 1.04981 w_k1_INTC_s106
                                 + 1.40133 w_k1_KO_s106 + 2.79936 w_k1_MAT_s106
                                 + 1.13098 w_k1_MSFT_s106
                                 + 1.20903 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.768356 w_k1_ADM_s106
                                 + 0.581913 w_k1_AXP_s106
                                 + 0.581165 w_k1_DIS_s106
                                 + 0.667022 w_k1_FDX_s106
                                 + 0.683514 w_k1_IBM_s106
                                 + 0.62191 w_k1_INTC_s106
                                 + 0.694808 w_k1_KO_s106
                                 + 0.973371 w_k1_MAT_s106
                                 + 0.756564 w_k1_MSFT_s106
                                 + 0.805729 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.01636 w_k1_ADM_s106
                                 + 1.02413 w_k1_AXP_s106
                                 + 1.08552 w_k1_DIS_s106
                                 + 1.07873 w_k1_FDX_s106
                                 + 0.647501 w_k1_IBM_s106
                                 + 0.755528 w_k1_INTC_s106
                                 + 0.91521 w_k1_KO_s106
                                 + 0.926167 w_k1_MAT_s106
                                 + 0.992534 w_k1_MSFT_s106
                                 + 1.0008 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.887576 w_k1_ADM_s106
                                 + 0.890702 w_k1_AXP_s106
                                 + 0.669375 w_k1_DIS_s106
                                 + 0.726926 w_k1_FDX_s106
                                 + 1.18351 w_k1_IBM_s106
                                 + 0.891029 w_k1_INTC_s106
                                 + 1.06866 w_k1_KO_s106 + 1.30376 w_k1_MAT_s106
                                 + 0.806411 w_k1_MSFT_s106
                                 + 0.953942 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.876333 w_k1_ADM_s106
                                 + 0.84245 w_k1_AXP_s106
                                 + 0.820617 w_k1_DIS_s106
                                 + 0.831518 w_k1_FDX_s106
                                 + 0.943997 w_k1_IBM_s106
                                 + 0.772978 w_k1_INTC_s106
                                 + 0.857956 w_k1_KO_s106
                                 + 0.749981 w_k1_MAT_s106
                                 + 0.89878 w_k1_MSFT_s106
                                 + 0.989661 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.17306 w_k1_ADM_s107
                                 + 1.68689 w_k1_AXP_s107
                                 + 1.48348 w_k1_DIS_s107
                                 + 1.31369 w_k1_FDX_s107
                                 + 1.06712 w_k1_IBM_s107
                                 + 1.31337 w_k1_INTC_s107
                                 + 1.16143 w_k1_KO_s107 + 1.37923 w_k1_MAT_s107
                                 + 1.39158 w_k1_MSFT_s107
                                 + 1.05762 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.297 w_k1_ADM_s107
                                 + 1.05245 w_k1_AXP_s107
                                 + 1.41683 w_k1_DIS_s107
                                 + 1.17196 w_k1_FDX_s107
                                 + 1.23308 w_k1_IBM_s107
                                 + 1.05923 w_k1_INTC_s107
                                 + 1.01627 w_k1_KO_s107 + 1.01021 w_k1_MAT_s107
                                 + 1.21348 w_k1_MSFT_s107
                                 + 1.09792 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.962339 w_k1_ADM_s107
                                 + 1.15668 w_k1_AXP_s107
                                 + 0.935445 w_k1_DIS_s107
                                 + 0.882469 w_k1_FDX_s107
                                 + 1.25789 w_k1_IBM_s107
                                 + 0.75932 w_k1_INTC_s107
                                 + 0.987215 w_k1_KO_s107
                                 + 1.74354 w_k1_MAT_s107
                                 + 0.839014 w_k1_MSFT_s107
                                 + 0.931304 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.23114 w_k1_ADM_s107
                                 + 1.08959 w_k1_AXP_s107
                                 + 0.892447 w_k1_DIS_s107
                                 + 1.07224 w_k1_FDX_s107
                                 + 0.859299 w_k1_IBM_s107
                                 + 0.841347 w_k1_INTC_s107
                                 + 0.790911 w_k1_KO_s107
                                 + 1.03423 w_k1_MAT_s107
                                 + 0.979136 w_k1_MSFT_s107
                                 + 1.00531 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.50651 w_k1_ADM_s107
                                 + 1.61802 w_k1_AXP_s107
                                 + 1.79736 w_k1_DIS_s107
                                 + 1.51711 w_k1_FDX_s107
                                 + 1.49928 w_k1_IBM_s107
                                 + 1.33864 w_k1_INTC_s107
                                 + 1.24191 w_k1_KO_s107 + 1.47318 w_k1_MAT_s107
                                 + 1.46442 w_k1_MSFT_s107
                                 + 1.24198 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.24274 w_k1_ADM_s107
                                 + 1.48359 w_k1_AXP_s107
                                 + 1.30851 w_k1_DIS_s107
                                 + 1.09397 w_k1_FDX_s107
                                 + 1.54736 w_k1_IBM_s107
                                 + 1.0669 w_k1_INTC_s107 + 1.32007 w_k1_KO_s107
                                 + 2.23565 w_k1_MAT_s107
                                 + 1.15931 w_k1_MSFT_s107
                                 + 1.22184 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.982429 w_k1_ADM_s107
                                 + 0.590455 w_k1_AXP_s107
                                 + 0.644404 w_k1_DIS_s107
                                 + 0.737736 w_k1_FDX_s107
                                 + 0.851969 w_k1_IBM_s107
                                 + 0.811069 w_k1_INTC_s107
                                 + 0.818083 w_k1_KO_s107
                                 + 0.646434 w_k1_MAT_s107
                                 + 0.896197 w_k1_MSFT_s107
                                 + 0.854597 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.00895 w_k1_ADM_s107
                                 + 0.758611 w_k1_AXP_s107
                                 + 1.04319 w_k1_DIS_s107
                                 + 1.02419 w_k1_FDX_s107
                                 + 0.883205 w_k1_IBM_s107
                                 + 0.97989 w_k1_INTC_s107
                                 + 0.81853 w_k1_KO_s107
                                 + 0.668844 w_k1_MAT_s107
                                 + 1.04432 w_k1_MSFT_s107
                                 + 0.98752 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.687273 w_k1_ADM_s110
                                 + 0.748185 w_k1_AXP_s110
                                 + 0.723887 w_k1_DIS_s110
                                 + 0.708291 w_k1_FDX_s110
                                 + 1.28885 w_k1_IBM_s110
                                 + 0.86393 w_k1_INTC_s110
                                 + 1.09577 w_k1_KO_s110 + 1.4161 w_k1_MAT_s110
                                 + 0.871626 w_k1_MSFT_s110
                                 + 0.951145 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.930842 w_k1_ADM_s110
                                 + 0.94778 w_k1_AXP_s110
                                 + 0.756287 w_k1_DIS_s110
                                 + 0.867479 w_k1_FDX_s110
                                 + 0.989076 w_k1_IBM_s110
                                 + 0.797673 w_k1_INTC_s110
                                 + 0.806392 w_k1_KO_s110
                                 + 1.11033 w_k1_MAT_s110
                                 + 0.911313 w_k1_MSFT_s110
                                 + 0.88097 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.6345 w_k1_ADM_s110
                                 + 1.51477 w_k1_AXP_s110
                                 + 1.85849 w_k1_DIS_s110 + 1.4756 w_k1_FDX_s110
                                 + 1.10449 w_k1_IBM_s110
                                 + 1.20596 w_k1_INTC_s110
                                 + 1.41714 w_k1_KO_s110 + 1.58325 w_k1_MAT_s110
                                 + 1.45634 w_k1_MSFT_s110
                                 + 1.25263 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.979826 w_k1_ADM_s110
                                 + 1.00477 w_k1_AXP_s110
                                 + 1.08962 w_k1_DIS_s110
                                 + 1.07683 w_k1_FDX_s110
                                 + 1.16081 w_k1_IBM_s110
                                 + 1.14104 w_k1_INTC_s110
                                 + 1.12469 w_k1_KO_s110 + 1.43738 w_k1_MAT_s110
                                 + 1.1492 w_k1_MSFT_s110 + 1.07121 w_k1_PG_s110
                                 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.39888 w_k1_ADM_s110
                                 + 1.21886 w_k1_AXP_s110
                                 + 1.47503 w_k1_DIS_s110
                                 + 1.10637 w_k1_FDX_s110
                                 + 1.64415 w_k1_IBM_s110
                                 + 0.987019 w_k1_INTC_s110
                                 + 1.50725 w_k1_KO_s110 + 1.75848 w_k1_MAT_s110
                                 + 1.07889 w_k1_MSFT_s110
                                 + 1.20188 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.951228 w_k1_ADM_s110
                                 + 0.817251 w_k1_AXP_s110
                                 + 0.986144 w_k1_DIS_s110
                                 + 0.90417 w_k1_FDX_s110
                                 + 0.797336 w_k1_IBM_s110
                                 + 0.860041 w_k1_INTC_s110
                                 + 0.960299 w_k1_KO_s110
                                 + 0.810314 w_k1_MAT_s110
                                 + 0.984084 w_k1_MSFT_s110
                                 + 0.906157 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.47335 w_k1_ADM_s110
                                 + 1.44913 w_k1_AXP_s110 + 1.139 w_k1_DIS_s110
                                 + 1.19344 w_k1_FDX_s110
                                 + 0.844338 w_k1_IBM_s110
                                 + 1.01825 w_k1_INTC_s110
                                 + 1.02268 w_k1_KO_s110
                                 + 0.902066 w_k1_MAT_s110
                                 + 1.13824 w_k1_MSFT_s110
                                 + 1.09824 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.04321 w_k1_ADM_s110
                                 + 1.18618 w_k1_AXP_s110 + 1.3286 w_k1_DIS_s110
                                 + 1.07296 w_k1_FDX_s110
                                 + 1.16756 w_k1_IBM_s110
                                 + 1.0806 w_k1_INTC_s110 + 1.20194 w_k1_KO_s110
                                 + 1.17367 w_k1_MAT_s110
                                 + 1.29375 w_k1_MSFT_s110
                                 + 1.20425 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.593597 w_k1_ADM_s111
                                 + 0.822947 w_k1_AXP_s111
                                 + 0.819103 w_k1_DIS_s111
                                 + 0.769301 w_k1_FDX_s111
                                 + 1.21034 w_k1_IBM_s111
                                 + 0.882937 w_k1_INTC_s111
                                 + 1.11722 w_k1_KO_s111 + 1.29145 w_k1_MAT_s111
                                 + 0.878489 w_k1_MSFT_s111
                                 + 0.934816 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.940876 w_k1_ADM_s111
                                 + 1.39604 w_k1_AXP_s111 + 1.3262 w_k1_DIS_s111
                                 + 1.28906 w_k1_FDX_s111 + 1.0894 w_k1_IBM_s111
                                 + 1.52194 w_k1_INTC_s111
                                 + 1.16534 w_k1_KO_s111 + 1.3454 w_k1_MAT_s111
                                 + 1.56113 w_k1_MSFT_s111 + 1.138 w_k1_PG_s111
                                 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.895099 w_k1_ADM_s111
                                 + 1.08528 w_k1_AXP_s111
                                 + 1.13565 w_k1_DIS_s111 + 1.1071 w_k1_FDX_s111
                                 + 1.24491 w_k1_IBM_s111
                                 + 1.04795 w_k1_INTC_s111
                                 + 1.35645 w_k1_KO_s111 + 1.48123 w_k1_MAT_s111
                                 + 1.35699 w_k1_MSFT_s111
                                 + 1.13394 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.902222 w_k1_ADM_s111
                                 + 1.05851 w_k1_AXP_s111
                                 + 0.82325 w_k1_DIS_s111
                                 + 0.930042 w_k1_FDX_s111
                                 + 1.02716 w_k1_IBM_s111
                                 + 0.9968 w_k1_INTC_s111
                                 + 0.873529 w_k1_KO_s111
                                 + 0.828735 w_k1_MAT_s111
                                 + 0.980814 w_k1_MSFT_s111
                                 + 0.859262 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.32717 w_k1_ADM_s111
                                 + 1.66183 w_k1_AXP_s111
                                 + 2.14273 w_k1_DIS_s111
                                 + 1.50212 w_k1_FDX_s111
                                 + 1.50743 w_k1_IBM_s111
                                 + 1.3665 w_k1_INTC_s111 + 1.62138 w_k1_KO_s111
                                 + 1.84689 w_k1_MAT_s111
                                 + 1.50125 w_k1_MSFT_s111 + 1.2242 w_k1_PG_s111
                                 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.607818 w_k1_ADM_s111
                                 + 0.623139 w_k1_AXP_s111
                                 + 0.714497 w_k1_DIS_s111
                                 + 0.746072 w_k1_FDX_s111
                                 + 0.976776 w_k1_IBM_s111
                                 + 0.821504 w_k1_INTC_s111
                                 + 0.899955 w_k1_KO_s111
                                 + 0.700654 w_k1_MAT_s111
                                 + 0.918654 w_k1_MSFT_s111
                                 + 0.856524 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.987469 w_k1_ADM_s111
                                 + 0.883951 w_k1_AXP_s111
                                 + 0.695782 w_k1_DIS_s111
                                 + 0.892099 w_k1_FDX_s111
                                 + 0.757732 w_k1_IBM_s111
                                 + 1.01414 w_k1_INTC_s111
                                 + 0.873868 w_k1_KO_s111
                                 + 0.578881 w_k1_MAT_s111
                                 + 0.983203 w_k1_MSFT_s111
                                 + 0.878756 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.14531 w_k1_ADM_s111
                                 + 1.23785 w_k1_AXP_s111
                                 + 1.11161 w_k1_DIS_s111
                                 + 0.996315 w_k1_FDX_s111
                                 + 1.76478 w_k1_IBM_s111
                                 + 1.27093 w_k1_INTC_s111
                                 + 1.42741 w_k1_KO_s111 + 1.38237 w_k1_MAT_s111
                                 + 1.08761 w_k1_MSFT_s111
                                 + 1.09057 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.30285 w_k1_ADM_s112
                                 + 1.53312 w_k1_AXP_s112
                                 + 1.23935 w_k1_DIS_s112
                                 + 1.15493 w_k1_FDX_s112
                                 + 1.16464 w_k1_IBM_s112
                                 + 1.06618 w_k1_INTC_s112
                                 + 1.16425 w_k1_KO_s112 + 1.75364 w_k1_MAT_s112
                                 + 0.902922 w_k1_MSFT_s112
                                 + 1.05636 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.34911 w_k1_ADM_s112
                                 + 1.60726 w_k1_AXP_s112
                                 + 1.56228 w_k1_DIS_s112
                                 + 1.36916 w_k1_FDX_s112
                                 + 0.973387 w_k1_IBM_s112
                                 + 0.994257 w_k1_INTC_s112
                                 + 1.12504 w_k1_KO_s112 + 1.17825 w_k1_MAT_s112
                                 + 1.18428 w_k1_MSFT_s112 + 1.0943 w_k1_PG_s112
                                 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.93262 w_k1_ADM_s112
                                 + 0.981748 w_k1_AXP_s112
                                 + 0.807416 w_k1_DIS_s112
                                 + 0.973406 w_k1_FDX_s112
                                 + 0.699557 w_k1_IBM_s112
                                 + 0.807332 w_k1_INTC_s112
                                 + 0.69773 w_k1_KO_s112
                                 + 0.741463 w_k1_MAT_s112
                                 + 0.840139 w_k1_MSFT_s112
                                 + 0.795006 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.904794 w_k1_ADM_s112
                                 + 1.39821 w_k1_AXP_s112
                                 + 1.23588 w_k1_DIS_s112
                                 + 1.13713 w_k1_FDX_s112
                                 + 1.37132 w_k1_IBM_s112
                                 + 1.03108 w_k1_INTC_s112
                                 + 1.20155 w_k1_KO_s112 + 1.70564 w_k1_MAT_s112
                                 + 1.08393 w_k1_MSFT_s112
                                 + 1.09456 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.88566 w_k1_ADM_s112
                                 + 1.07314 w_k1_AXP_s112
                                 + 0.938546 w_k1_DIS_s112
                                 + 0.908224 w_k1_FDX_s112
                                 + 1.12223 w_k1_IBM_s112
                                 + 0.947404 w_k1_INTC_s112
                                 + 1.03457 w_k1_KO_s112 + 1.25192 w_k1_MAT_s112
                                 + 0.870046 w_k1_MSFT_s112
                                 + 0.977864 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.1713 w_k1_ADM_s112
                                 + 1.07369 w_k1_AXP_s112
                                 + 0.96326 w_k1_DIS_s112
                                 + 1.12164 w_k1_FDX_s112
                                 + 0.763436 w_k1_IBM_s112
                                 + 0.950437 w_k1_INTC_s112
                                 + 0.880633 w_k1_KO_s112
                                 + 0.676584 w_k1_MAT_s112
                                 + 0.992769 w_k1_MSFT_s112
                                 + 0.907891 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.787309 w_k1_ADM_s112
                                 + 0.708488 w_k1_AXP_s112
                                 + 0.68123 w_k1_DIS_s112
                                 + 0.82789 w_k1_FDX_s112
                                 + 0.820864 w_k1_IBM_s112
                                 + 0.795475 w_k1_INTC_s112
                                 + 0.882587 w_k1_KO_s112
                                 + 0.84535 w_k1_MAT_s112
                                 + 0.776207 w_k1_MSFT_s112
                                 + 0.823099 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.12116 w_k1_ADM_s112
                                 + 1.09168 w_k1_AXP_s112
                                 + 1.42074 w_k1_DIS_s112
                                 + 1.15639 w_k1_FDX_s112
                                 + 1.05615 w_k1_IBM_s112
                                 + 1.07922 w_k1_INTC_s112
                                 + 1.27142 w_k1_KO_s112 + 1.36661 w_k1_MAT_s112
                                 + 1.15954 w_k1_MSFT_s112
                                 + 1.02719 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.36558 w_k1_ADM_s113
                                 + 1.30966 w_k1_AXP_s113
                                 + 1.13779 w_k1_DIS_s113
                                 + 1.20511 w_k1_FDX_s113
                                 + 0.758675 w_k1_IBM_s113
                                 + 1.17398 w_k1_INTC_s113
                                 + 0.799928 w_k1_KO_s113
                                 + 0.692898 w_k1_MAT_s113
                                 + 1.20399 w_k1_MSFT_s113
                                 + 1.04501 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.00717 w_k1_ADM_s113
                                 + 1.15471 w_k1_AXP_s113
                                 + 1.24638 w_k1_DIS_s113
                                 + 1.07593 w_k1_FDX_s113
                                 + 1.16432 w_k1_IBM_s113
                                 + 1.20319 w_k1_INTC_s113 + 1.2561 w_k1_KO_s113
                                 + 1.20724 w_k1_MAT_s113
                                 + 1.20325 w_k1_MSFT_s113
                                 + 1.10427 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.42959 w_k1_ADM_s113
                                 + 1.65898 w_k1_AXP_s113
                                 + 1.61206 w_k1_DIS_s113
                                 + 1.43959 w_k1_FDX_s113
                                 + 1.09162 w_k1_IBM_s113
                                 + 1.27014 w_k1_INTC_s113
                                 + 1.06739 w_k1_KO_s113
                                 + 0.945593 w_k1_MAT_s113
                                 + 1.27031 w_k1_MSFT_s113
                                 + 1.19417 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.975674 w_k1_ADM_s113
                                 + 0.929394 w_k1_AXP_s113
                                 + 0.968597 w_k1_DIS_s113
                                 + 1.01007 w_k1_FDX_s113
                                 + 0.847265 w_k1_IBM_s113
                                 + 0.969616 w_k1_INTC_s113
                                 + 0.762404 w_k1_KO_s113
                                 + 0.561461 w_k1_MAT_s113
                                 + 1.06129 w_k1_MSFT_s113
                                 + 0.890836 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.767432 w_k1_ADM_s113
                                 + 0.967497 w_k1_AXP_s113
                                 + 0.769745 w_k1_DIS_s113
                                 + 0.839263 w_k1_FDX_s113
                                 + 0.997949 w_k1_IBM_s113
                                 + 0.994231 w_k1_INTC_s113
                                 + 0.842018 w_k1_KO_s113
                                 + 0.829226 w_k1_MAT_s113
                                 + 0.883107 w_k1_MSFT_s113
                                 + 0.836459 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.676097 w_k1_ADM_s113
                                 + 0.548459 w_k1_AXP_s113
                                 + 0.703042 w_k1_DIS_s113
                                 + 0.723726 w_k1_FDX_s113
                                 + 0.776621 w_k1_IBM_s113
                                 + 0.841623 w_k1_INTC_s113
                                 + 0.806226 w_k1_KO_s113
                                 + 0.631693 w_k1_MAT_s113
                                 + 0.853607 w_k1_MSFT_s113
                                 + 0.84719 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.2945 w_k1_ADM_s113
                                 + 1.28544 w_k1_AXP_s113
                                 + 1.73317 w_k1_DIS_s113
                                 + 1.25643 w_k1_FDX_s113
                                 + 1.47611 w_k1_IBM_s113
                                 + 1.19565 w_k1_INTC_s113 + 1.4601 w_k1_KO_s113
                                 + 1.66768 w_k1_MAT_s113
                                 + 1.3364 w_k1_MSFT_s113 + 1.22548 w_k1_PG_s113
                                 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.740922 w_k1_ADM_s113
                                 + 0.653457 w_k1_AXP_s113
                                 + 0.660171 w_k1_DIS_s113
                                 + 0.827251 w_k1_FDX_s113
                                 + 0.535338 w_k1_IBM_s113
                                 + 0.870411 w_k1_INTC_s113
                                 + 0.612153 w_k1_KO_s113
                                 + 0.476157 w_k1_MAT_s113
                                 + 0.907856 w_k1_MSFT_s113
                                 + 0.794038 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.09377 w_k1_ADM_s114
                                 + 1.27365 w_k1_AXP_s114
                                 + 1.08398 w_k1_DIS_s114 + 1.1779 w_k1_FDX_s114
                                 + 1.08731 w_k1_IBM_s114
                                 + 1.18239 w_k1_INTC_s114
                                 + 1.00762 w_k1_KO_s114 + 1.40502 w_k1_MAT_s114
                                 + 1.12623 w_k1_MSFT_s114
                                 + 1.23623 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.02317 w_k1_ADM_s114
                                 + 0.935345 w_k1_AXP_s114
                                 + 0.702299 w_k1_DIS_s114
                                 + 0.934274 w_k1_FDX_s114
                                 + 0.693402 w_k1_IBM_s114
                                 + 0.999066 w_k1_INTC_s114
                                 + 0.686852 w_k1_KO_s114
                                 + 0.861681 w_k1_MAT_s114
                                 + 0.938212 w_k1_MSFT_s114
                                 + 0.903946 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.16016 w_k1_ADM_s114
                                 + 1.09038 w_k1_AXP_s114
                                 + 1.17235 w_k1_DIS_s114
                                 + 1.15054 w_k1_FDX_s114
                                 + 0.81132 w_k1_IBM_s114
                                 + 0.978795 w_k1_INTC_s114
                                 + 1.00516 w_k1_KO_s114
                                 + 0.840181 w_k1_MAT_s114
                                 + 1.28458 w_k1_MSFT_s114
                                 + 1.12853 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.8453 w_k1_ADM_s114
                                 + 0.78114 w_k1_AXP_s114
                                 + 0.684757 w_k1_DIS_s114
                                 + 0.796237 w_k1_FDX_s114
                                 + 0.949912 w_k1_IBM_s114
                                 + 0.860982 w_k1_INTC_s114
                                 + 0.816705 w_k1_KO_s114
                                 + 1.06537 w_k1_MAT_s114
                                 + 0.884872 w_k1_MSFT_s114
                                 + 0.959624 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.15556 w_k1_ADM_s114
                                 + 1.08874 w_k1_AXP_s114
                                 + 1.03602 w_k1_DIS_s114 + 1.0437 w_k1_FDX_s114
                                 + 0.955799 w_k1_IBM_s114
                                 + 1.0122 w_k1_INTC_s114
                                 + 0.928508 w_k1_KO_s114
                                 + 0.782517 w_k1_MAT_s114
                                 + 1.013 w_k1_MSFT_s114 + 1.03952 w_k1_PG_s114
                                 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.06509 w_k1_ADM_s114
                                 + 1.09573 w_k1_AXP_s114
                                 + 1.10601 w_k1_DIS_s114
                                 + 0.985165 w_k1_FDX_s114
                                 + 1.53785 w_k1_IBM_s114
                                 + 1.14242 w_k1_INTC_s114
                                 + 1.21496 w_k1_KO_s114 + 1.67606 w_k1_MAT_s114
                                 + 1.09802 w_k1_MSFT_s114
                                 + 1.13647 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.11563 w_k1_ADM_s114
                                 + 0.896129 w_k1_AXP_s114
                                 + 1.29498 w_k1_DIS_s114
                                 + 0.98879 w_k1_FDX_s114
                                 + 1.18516 w_k1_IBM_s114
                                 + 0.973217 w_k1_INTC_s114
                                 + 1.14491 w_k1_KO_s114
                                 + 0.749516 w_k1_MAT_s114
                                 + 1.13311 w_k1_MSFT_s114
                                 + 1.06041 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.22103 w_k1_ADM_s114
                                 + 1.43887 w_k1_AXP_s114
                                 + 1.67452 w_k1_DIS_s114
                                 + 1.45776 w_k1_FDX_s114
                                 + 1.14949 w_k1_IBM_s114
                                 + 1.35876 w_k1_INTC_s114
                                 + 1.27915 w_k1_KO_s114 + 1.14233 w_k1_MAT_s114
                                 + 1.41838 w_k1_MSFT_s114
                                 + 1.36478 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.18619 w_k1_ADM_s115
                                 + 1.35371 w_k1_AXP_s115 + 1.3106 w_k1_DIS_s115
                                 + 1.22379 w_k1_FDX_s115
                                 + 1.20488 w_k1_IBM_s115
                                 + 0.949816 w_k1_INTC_s115
                                 + 1.19046 w_k1_KO_s115 + 1.67204 w_k1_MAT_s115
                                 + 1.14401 w_k1_MSFT_s115
                                 + 1.08292 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.841174 w_k1_ADM_s115
                                 + 0.605971 w_k1_AXP_s115
                                 + 0.847859 w_k1_DIS_s115
                                 + 0.824647 w_k1_FDX_s115
                                 + 0.735742 w_k1_IBM_s115
                                 + 0.846145 w_k1_INTC_s115
                                 + 0.913543 w_k1_KO_s115
                                 + 0.529096 w_k1_MAT_s115
                                 + 0.882578 w_k1_MSFT_s115
                                 + 0.826339 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.1653 w_k1_ADM_s115
                                 + 1.40392 w_k1_AXP_s115
                                 + 1.57985 w_k1_DIS_s115
                                 + 1.22515 w_k1_FDX_s115
                                 + 1.67465 w_k1_IBM_s115
                                 + 0.914834 w_k1_INTC_s115
                                 + 1.2205 w_k1_KO_s115 + 1.87638 w_k1_MAT_s115
                                 + 1.18334 w_k1_MSFT_s115
                                 + 1.06478 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.35649 w_k1_ADM_s115
                                 + 1.16331 w_k1_AXP_s115
                                 + 1.31858 w_k1_DIS_s115
                                 + 1.26094 w_k1_FDX_s115
                                 + 0.865692 w_k1_IBM_s115
                                 + 0.891349 w_k1_INTC_s115
                                 + 1.11245 w_k1_KO_s115
                                 + 0.816647 w_k1_MAT_s115
                                 + 1.07193 w_k1_MSFT_s115
                                 + 1.05354 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.0176 w_k1_ADM_s115
                                 + 0.894208 w_k1_AXP_s115
                                 + 1.18444 w_k1_DIS_s115
                                 + 0.94337 w_k1_FDX_s115
                                 + 1.19709 w_k1_IBM_s115
                                 + 0.833846 w_k1_INTC_s115
                                 + 1.25951 w_k1_KO_s115 + 1.14752 w_k1_MAT_s115
                                 + 0.970818 w_k1_MSFT_s115
                                 + 0.939224 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.05605 w_k1_ADM_s115
                                 + 1.06845 w_k1_AXP_s115
                                 + 0.954673 w_k1_DIS_s115
                                 + 1.01056 w_k1_FDX_s115
                                 + 0.893133 w_k1_IBM_s115
                                 + 0.892521 w_k1_INTC_s115
                                 + 0.950867 w_k1_KO_s115 + 1.0882 w_k1_MAT_s115
                                 + 0.975416 w_k1_MSFT_s115
                                 + 0.905483 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.924362 w_k1_ADM_s115
                                 + 0.623602 w_k1_AXP_s115
                                 + 0.629105 w_k1_DIS_s115
                                 + 0.717669 w_k1_FDX_s115
                                 + 0.90265 w_k1_IBM_s115
                                 + 0.644971 w_k1_INTC_s115
                                 + 0.78573 w_k1_KO_s115 + 0.88511 w_k1_MAT_s115
                                 + 0.783227 w_k1_MSFT_s115
                                 + 0.734026 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.94603 w_k1_ADM_s115
                                 + 0.908201 w_k1_AXP_s115
                                 + 0.720308 w_k1_DIS_s115
                                 + 0.872718 w_k1_FDX_s115
                                 + 0.916196 w_k1_IBM_s115
                                 + 0.795077 w_k1_INTC_s115
                                 + 0.928236 w_k1_KO_s115
                                 + 0.886419 w_k1_MAT_s115
                                 + 0.85351 w_k1_MSFT_s115
                                 + 0.886743 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.894387 w_k1_ADM_s116
                                 + 1.24981 w_k1_AXP_s116
                                 + 1.30916 w_k1_DIS_s116
                                 + 1.06598 w_k1_FDX_s116
                                 + 1.48694 w_k1_IBM_s116
                                 + 0.969189 w_k1_INTC_s116
                                 + 1.22685 w_k1_KO_s116 + 1.63863 w_k1_MAT_s116
                                 + 1.11524 w_k1_MSFT_s116
                                 + 1.13859 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.23463 w_k1_ADM_s116
                                 + 1.14326 w_k1_AXP_s116
                                 + 1.12257 w_k1_DIS_s116
                                 + 1.16371 w_k1_FDX_s116
                                 + 0.832811 w_k1_IBM_s116
                                 + 1.24148 w_k1_INTC_s116
                                 + 0.935226 w_k1_KO_s116
                                 + 0.804223 w_k1_MAT_s116
                                 + 1.17896 w_k1_MSFT_s116
                                 + 1.15009 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.00295 w_k1_ADM_s116
                                 + 1.06669 w_k1_AXP_s116
                                 + 0.97595 w_k1_DIS_s116
                                 + 0.942859 w_k1_FDX_s116
                                 + 0.90432 w_k1_IBM_s116
                                 + 0.990207 w_k1_INTC_s116
                                 + 0.965636 w_k1_KO_s116
                                 + 0.87527 w_k1_MAT_s116
                                 + 1.07405 w_k1_MSFT_s116
                                 + 1.10807 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.982832 w_k1_ADM_s116
                                 + 1.01097 w_k1_AXP_s116
                                 + 0.916895 w_k1_DIS_s116
                                 + 0.84026 w_k1_FDX_s116
                                 + 1.10817 w_k1_IBM_s116
                                 + 1.05364 w_k1_INTC_s116
                                 + 0.958005 w_k1_KO_s116
                                 + 1.09221 w_k1_MAT_s116
                                 + 0.997189 w_k1_MSFT_s116
                                 + 0.941216 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.14613 w_k1_ADM_s116
                                 + 1.59743 w_k1_AXP_s116
                                 + 1.03874 w_k1_DIS_s116
                                 + 1.08185 w_k1_FDX_s116 + 1.1937 w_k1_IBM_s116
                                 + 1.15928 w_k1_INTC_s116
                                 + 1.06741 w_k1_KO_s116 + 1.49831 w_k1_MAT_s116
                                 + 1.09934 w_k1_MSFT_s116 + 1.1789 w_k1_PG_s116
                                 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.849985 w_k1_ADM_s116
                                 + 0.811355 w_k1_AXP_s116
                                 + 0.701644 w_k1_DIS_s116
                                 + 0.820648 w_k1_FDX_s116
                                 + 0.745189 w_k1_IBM_s116
                                 + 0.783263 w_k1_INTC_s116
                                 + 0.736953 w_k1_KO_s116
                                 + 1.06335 w_k1_MAT_s116
                                 + 0.818912 w_k1_MSFT_s116
                                 + 0.89074 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.28758 w_k1_ADM_s116
                                 + 1.67407 w_k1_AXP_s116
                                 + 1.67774 w_k1_DIS_s116
                                 + 1.38077 w_k1_FDX_s116
                                 + 1.26998 w_k1_IBM_s116
                                 + 1.5153 w_k1_INTC_s116 + 1.3573 w_k1_KO_s116
                                 + 1.3371 w_k1_MAT_s116
                                 + 1.35006 w_k1_MSFT_s116
                                 + 1.21541 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.06016 w_k1_ADM_s116
                                 + 0.879899 w_k1_AXP_s116
                                 + 0.935782 w_k1_DIS_s116
                                 + 0.969837 w_k1_FDX_s116
                                 + 0.62035 w_k1_IBM_s116
                                 + 1.13785 w_k1_INTC_s116
                                 + 0.851603 w_k1_KO_s116
                                 + 0.57209 w_k1_MAT_s116
                                 + 1.10997 w_k1_MSFT_s116
                                 + 0.835106 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.00239 w_k1_ADM_s117
                                 + 0.927225 w_k1_AXP_s117
                                 + 0.937967 w_k1_DIS_s117
                                 + 0.887413 w_k1_FDX_s117
                                 + 1.26887 w_k1_IBM_s117
                                 + 1.14483 w_k1_INTC_s117
                                 + 1.04063 w_k1_KO_s117
                                 + 0.719927 w_k1_MAT_s117
                                 + 0.914732 w_k1_MSFT_s117
                                 + 1.06463 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.00078 w_k1_ADM_s117
                                 + 0.693743 w_k1_AXP_s117
                                 + 0.591819 w_k1_DIS_s117
                                 + 0.690285 w_k1_FDX_s117
                                 + 0.766319 w_k1_IBM_s117
                                 + 0.707303 w_k1_INTC_s117
                                 + 0.719879 w_k1_KO_s117
                                 + 0.46514 w_k1_MAT_s117
                                 + 0.662574 w_k1_MSFT_s117
                                 + 0.766568 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.15779 w_k1_ADM_s117
                                 + 1.4185 w_k1_AXP_s117 + 1.61706 w_k1_DIS_s117
                                 + 1.21859 w_k1_FDX_s117
                                 + 1.48297 w_k1_IBM_s117
                                 + 1.20151 w_k1_INTC_s117
                                 + 1.35692 w_k1_KO_s117 + 1.32363 w_k1_MAT_s117
                                 + 1.25236 w_k1_MSFT_s117
                                 + 1.22893 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.37601 w_k1_ADM_s117
                                 + 1.0386 w_k1_AXP_s117 + 1.11958 w_k1_DIS_s117
                                 + 1.05023 w_k1_FDX_s117
                                 + 1.28899 w_k1_IBM_s117
                                 + 1.02123 w_k1_INTC_s117
                                 + 1.14398 w_k1_KO_s117
                                 + 0.724046 w_k1_MAT_s117
                                 + 0.891333 w_k1_MSFT_s117
                                 + 0.994443 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.0426 w_k1_ADM_s117
                                 + 0.808948 w_k1_AXP_s117
                                 + 0.786931 w_k1_DIS_s117
                                 + 1.00642 w_k1_FDX_s117
                                 + 0.626328 w_k1_IBM_s117
                                 + 0.900772 w_k1_INTC_s117
                                 + 0.757089 w_k1_KO_s117
                                 + 0.626856 w_k1_MAT_s117
                                 + 0.91754 w_k1_MSFT_s117
                                 + 0.903914 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.49912 w_k1_ADM_s117
                                 + 1.4295 w_k1_AXP_s117 + 1.41456 w_k1_DIS_s117
                                 + 1.38762 w_k1_FDX_s117
                                 + 0.97398 w_k1_IBM_s117
                                 + 1.31141 w_k1_INTC_s117
                                 + 1.17889 w_k1_KO_s117
                                 + 0.985896 w_k1_MAT_s117
                                 + 1.30394 w_k1_MSFT_s117
                                 + 1.19217 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.09621 w_k1_ADM_s117
                                 + 0.742694 w_k1_AXP_s117
                                 + 0.826389 w_k1_DIS_s117
                                 + 0.839149 w_k1_FDX_s117
                                 + 0.902117 w_k1_IBM_s117
                                 + 1.04314 w_k1_INTC_s117
                                 + 0.991844 w_k1_KO_s117
                                 + 0.569461 w_k1_MAT_s117
                                 + 0.804318 w_k1_MSFT_s117
                                 + 0.910277 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.29865 w_k1_ADM_s117
                                 + 1.03864 w_k1_AXP_s117
                                 + 0.903469 w_k1_DIS_s117
                                 + 1.06485 w_k1_FDX_s117
                                 + 0.877113 w_k1_IBM_s117
                                 + 0.864089 w_k1_INTC_s117
                                 + 0.899702 w_k1_KO_s117
                                 + 0.82924 w_k1_MAT_s117
                                 + 0.842824 w_k1_MSFT_s117
                                 + 0.984159 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.00613279511872141 w_k1_ADM_s000
                                 + 0.0173713650969484 w_k1_ADM_s001
                                 + 0.00716912293960491 w_k1_ADM_s002
                                 + 0.0058214388489917 w_k1_ADM_s003
                                 + 0.00868783412850029 w_k1_ADM_s004
                                 + 0.0158548136310841 w_k1_ADM_s005
                                 + 0.0098241754738009 w_k1_ADM_s006
                                 + 0.00937692676811562 w_k1_ADM_s007
                                 + 0.0392377401692812 w_k1_ADM_s010
                                 + 0.0399177849276988 w_k1_ADM_s011
                                 + 0.014780572814313 w_k1_ADM_s012
                                 + 0.0538658049769288 w_k1_ADM_s013
                                 + 0.0135058316666726 w_k1_ADM_s014
                                 + 0.0582167918986744 w_k1_ADM_s015
                                 + 0.0190700159193875 w_k1_ADM_s016
                                 + 0.0213485423588329 w_k1_ADM_s017
                                 + 0.063720935845543 w_k1_ADM_s100
                                 + 0.0407873980497466 w_k1_ADM_s101
                                 + 0.0130891607197005 w_k1_ADM_s102
                                 + 0.0194566527132441 w_k1_ADM_s103
                                 + 0.0395066279997464 w_k1_ADM_s104
                                 + 0.0405313066750607 w_k1_ADM_s105
                                 + 0.0355272257022573 w_k1_ADM_s106
                                 + 0.0368133259640369 w_k1_ADM_s107
                                 + 0.0374464459671716 w_k1_ADM_s110
                                 + 0.017354629464557 w_k1_ADM_s111
                                 + 0.0312780709844002 w_k1_ADM_s112
                                 + 0.0491867910616588 w_k1_ADM_s113
                                 + 0.0395388550192593 w_k1_ADM_s114
                                 + 0.0961672285505495 w_k1_ADM_s115
                                 + 0.0598174033351225 w_k1_ADM_s116
                                 + 0.079672634170358 w_k1_ADM_s117
                                 + 0.00646094477352642 w_k1_AXP_s000
                                 + 0.0175723667374897 w_k1_AXP_s001
                                 + 0.00737556279827238 w_k1_AXP_s002
                                 + 0.00831585877786905 w_k1_AXP_s003
                                 + 0.00853597872665894 w_k1_AXP_s004
                                 + 0.0140679764077756 w_k1_AXP_s005
                                 + 0.00997394206663151 w_k1_AXP_s006
                                 + 0.0100109747882528 w_k1_AXP_s007
                                 + 0.0427897183986377 w_k1_AXP_s010
                                 + 0.0390919384015788 w_k1_AXP_s011
                                 + 0.0157506059766898 w_k1_AXP_s012
                                 + 0.0492139003341309 w_k1_AXP_s013
                                 + 0.0124953689163436 w_k1_AXP_s014
                                 + 0.0491729087916392 w_k1_AXP_s015
                                 + 0.0198237382867201 w_k1_AXP_s016
                                 + 0.0198219534146018 w_k1_AXP_s017
                                 + 0.0758752571661078 w_k1_AXP_s100
                                 + 0.0417792433639665 w_k1_AXP_s101
                                 + 0.0137338350000694 w_k1_AXP_s102
                                 + 0.0199038743221061 w_k1_AXP_s103
                                 + 0.0405021659474421 w_k1_AXP_s104
                                 + 0.0456464644858741 w_k1_AXP_s105
                                 + 0.0390414376893986 w_k1_AXP_s106
                                 + 0.0344623676022496 w_k1_AXP_s107
                                 + 0.0360138541077928 w_k1_AXP_s110
                                 + 0.0197591333525674 w_k1_AXP_s111
                                 + 0.0341222983957677 w_k1_AXP_s112
                                 + 0.0517810471136011 w_k1_AXP_s113
                                 + 0.0391076783733287 w_k1_AXP_s114
                                 + 0.0931549174956226 w_k1_AXP_s115
                                 + 0.0659874779816343 w_k1_AXP_s116
                                 + 0.0682871446740439 w_k1_AXP_s117
                                 + 0.00640044654348936 w_k1_DIS_s000
                                 + 0.0159412864321249 w_k1_DIS_s001
                                 + 0.0070646505286116 w_k1_DIS_s002
                                 + 0.00818902514371585 w_k1_DIS_s003
                                 + 0.00824261091912663 w_k1_DIS_s004
                                 + 0.0137773400014192 w_k1_DIS_s005
                                 + 0.0103415863511892 w_k1_DIS_s006
                                 + 0.00923694481867794 w_k1_DIS_s007
                                 + 0.0458842288765554 w_k1_DIS_s010
                                 + 0.0391748200251185 w_k1_DIS_s011
                                 + 0.0160595920540527 w_k1_DIS_s012
                                 + 0.0540480339588489 w_k1_DIS_s013
                                 + 0.0133111196480694 w_k1_DIS_s014
                                 + 0.0527088785023519 w_k1_DIS_s015
                                 + 0.0217339331207628 w_k1_DIS_s016
                                 + 0.0207395563688338 w_k1_DIS_s017
                                 + 0.0723159529108328 w_k1_DIS_s100
                                 + 0.0417097803187937 w_k1_DIS_s101
                                 + 0.0146790254764584 w_k1_DIS_s102
                                 + 0.0195523574045699 w_k1_DIS_s103
                                 + 0.0395257024951232 w_k1_DIS_s104
                                 + 0.0426823170260809 w_k1_DIS_s105
                                 + 0.0388630014190842 w_k1_DIS_s106
                                 + 0.0339289734460076 w_k1_DIS_s107
                                 + 0.0374484349098808 w_k1_DIS_s110
                                 + 0.0190208451429845 w_k1_DIS_s111
                                 + 0.0318000479908888 w_k1_DIS_s112
                                 + 0.053162887956963 w_k1_DIS_s113
                                 + 0.0395105707784159 w_k1_DIS_s114
                                 + 0.0949334942600124 w_k1_DIS_s115
                                 + 0.0608451437611964 w_k1_DIS_s116
                                 + 0.0697222791045523 w_k1_DIS_s117
                                 + 0.00630973825192131 w_k1_FDX_s000
                                 + 0.0151720910825561 w_k1_FDX_s001
                                 + 0.00688516787034119 w_k1_FDX_s002
                                 + 0.00726335380844238 w_k1_FDX_s003
                                 + 0.00857043658985749 w_k1_FDX_s004
                                 + 0.0142440243195999 w_k1_FDX_s005
                                 + 0.00945769633199055 w_k1_FDX_s006
                                 + 0.00897647706129963 w_k1_FDX_s007
                                 + 0.0436458846356501 w_k1_FDX_s010
                                 + 0.0363498944530977 w_k1_FDX_s011
                                 + 0.0154091794626334 w_k1_FDX_s012
                                 + 0.0524072128673226 w_k1_FDX_s013
                                 + 0.013220559443462 w_k1_FDX_s014
                                 + 0.0523071446490044 w_k1_FDX_s015
                                 + 0.0205155428467898 w_k1_FDX_s016
                                 + 0.0211383481434585 w_k1_FDX_s017
                                 + 0.0679016092742342 w_k1_FDX_s100
                                 + 0.0405573131499648 w_k1_FDX_s101
                                 + 0.0138568896607018 w_k1_FDX_s102
                                 + 0.0191264353617203 w_k1_FDX_s103
                                 + 0.0395614927016874 w_k1_FDX_s104
                                 + 0.0405623184155297 w_k1_FDX_s105
                                 + 0.0366876143465405 w_k1_FDX_s106
                                 + 0.0335503568734389 w_k1_FDX_s107
                                 + 0.034502919274646 w_k1_FDX_s110
                                 + 0.0191757106845197 w_k1_FDX_s111
                                 + 0.0319305883861615 w_k1_FDX_s112
                                 + 0.0507164712855806 w_k1_FDX_s113
                                 + 0.039036364292499 w_k1_FDX_s114
                                 + 0.0919969444760226 w_k1_FDX_s115
                                 + 0.0579966371314962 w_k1_FDX_s116
                                 + 0.0686208053582855 w_k1_FDX_s117
                                 + 0.00611389932232233 w_k1_IBM_s000
                                 + 0.0156897023998712 w_k1_IBM_s001
                                 + 0.00709119231932184 w_k1_IBM_s002
                                 + 0.00714306412016041 w_k1_IBM_s003
                                 + 0.00874538351086084 w_k1_IBM_s004
                                 + 0.0146188680377139 w_k1_IBM_s005
                                 + 0.00902454596809957 w_k1_IBM_s006
                                 + 0.00964234425487327 w_k1_IBM_s007
                                 + 0.0447736293694303 w_k1_IBM_s010
                                 + 0.0368956252978784 w_k1_IBM_s011
                                 + 0.0152170521449016 w_k1_IBM_s012
                                 + 0.0531675444491396 w_k1_IBM_s013
                                 + 0.0129387278709753 w_k1_IBM_s014
                                 + 0.0511989139172043 w_k1_IBM_s015
                                 + 0.0217809461184141 w_k1_IBM_s016
                                 + 0.0207452874079219 w_k1_IBM_s017
                                 + 0.0687020055467858 w_k1_IBM_s100
                                 + 0.04158219815415 w_k1_IBM_s101
                                 + 0.0138438486586732 w_k1_IBM_s102
                                 + 0.01986610932513 w_k1_IBM_s103
                                 + 0.0384095328240318 w_k1_IBM_s104
                                 + 0.042675610930369 w_k1_IBM_s105
                                 + 0.0383250150697785 w_k1_IBM_s106
                                 + 0.0348172491632441 w_k1_IBM_s107
                                 + 0.0372280674952032 w_k1_IBM_s110
                                 + 0.021133585731559 w_k1_IBM_s111
                                 + 0.0292510046005575 w_k1_IBM_s112
                                 + 0.0484845809825889 w_k1_IBM_s113
                                 + 0.0387781563126178 w_k1_IBM_s114
                                 + 0.0928667181552873 w_k1_IBM_s115
                                 + 0.0598633666954212 w_k1_IBM_s116
                                 + 0.0685129524389618 w_k1_IBM_s117
                                 + 0.00726237548610044 w_k1_INTC_s000
                                 + 0.0141709223733471 w_k1_INTC_s001
                                 + 0.00763566874757738 w_k1_INTC_s002
                                 + 0.00685768442291427 w_k1_INTC_s003
                                 + 0.0104453716267214 w_k1_INTC_s004
                                 + 0.0134257806692784 w_k1_INTC_s005
                                 + 0.00847924857475497 w_k1_INTC_s006
                                 + 0.00905147701543639 w_k1_INTC_s007
                                 + 0.0470779772733319 w_k1_INTC_s010
                                 + 0.0311665386689114 w_k1_INTC_s011
                                 + 0.0150834547979356 w_k1_INTC_s012
                                 + 0.0517266708653224 w_k1_INTC_s013
                                 + 0.0115092673359907 w_k1_INTC_s014
                                 + 0.0554495093379337 w_k1_INTC_s015
                                 + 0.0221340731202992 w_k1_INTC_s016
                                 + 0.0207941607985352 w_k1_INTC_s017
                                 + 0.0783086126239266 w_k1_INTC_s100
                                 + 0.0368621693388787 w_k1_INTC_s101
                                 + 0.0151883620552195 w_k1_INTC_s102
                                 + 0.0186516028903736 w_k1_INTC_s103
                                 + 0.0367304765104846 w_k1_INTC_s104
                                 + 0.0399018081595394 w_k1_INTC_s105
                                 + 0.0349490192453256 w_k1_INTC_s106
                                 + 0.0308009847195675 w_k1_INTC_s107
                                 + 0.0326432798070155 w_k1_INTC_s110
                                 + 0.0209767182650075 w_k1_INTC_s111
                                 + 0.028491256549198 w_k1_INTC_s112
                                 + 0.0526347300792001 w_k1_INTC_s113
                                 + 0.039345394950994 w_k1_INTC_s114
                                 + 0.0768208812318092 w_k1_INTC_s115
                                 + 0.0621457907723116 w_k1_INTC_s116
                                 + 0.0702734062470857 w_k1_INTC_s117
                                 + 0.00706633816440291 w_k1_KO_s000
                                 + 0.0163359465380962 w_k1_KO_s001
                                 + 0.00690412407446918 w_k1_KO_s002
                                 + 0.00677987932334959 w_k1_KO_s003
                                 + 0.00850074620933557 w_k1_KO_s004
                                 + 0.0126236304454123 w_k1_KO_s005
                                 + 0.00858753341833481 w_k1_KO_s006
                                 + 0.00917848624816678 w_k1_KO_s007
                                 + 0.0455411534938454 w_k1_KO_s010
                                 + 0.0356421913700044 w_k1_KO_s011
                                 + 0.0130960451666662 w_k1_KO_s012
                                 + 0.0532276890470928 w_k1_KO_s013
                                 + 0.0127610627240713 w_k1_KO_s014
                                 + 0.0547272701857046 w_k1_KO_s015
                                 + 0.0222595788742277 w_k1_KO_s016
                                 + 0.0217101662037491 w_k1_KO_s017
                                 + 0.0688356924134609 w_k1_KO_s100
                                 + 0.039919097147317 w_k1_KO_s101
                                 + 0.016314340322408 w_k1_KO_s102
                                 + 0.0197823401841075 w_k1_KO_s103
                                 + 0.0383149572480198 w_k1_KO_s104
                                 + 0.0439525331388892 w_k1_KO_s105
                                 + 0.0416908222379232 w_k1_KO_s106
                                 + 0.0309198460963457 w_k1_KO_s107
                                 + 0.0374494139922797 w_k1_KO_s110
                                 + 0.0211102510995348 w_k1_KO_s111
                                 + 0.0304374947069032 w_k1_KO_s112
                                 + 0.0475257986954488 w_k1_KO_s113
                                 + 0.0370656062230884 w_k1_KO_s114
                                 + 0.0935270145237835 w_k1_KO_s115
                                 + 0.0578286402094443 w_k1_KO_s116
                                 + 0.0687735682080961 w_k1_KO_s117
                                 + 0.00700021198629338 w_k1_MAT_s000
                                 + 0.0180767930435985 w_k1_MAT_s001
                                 + 0.00697877259327077 w_k1_MAT_s002
                                 + 0.0069437663163606 w_k1_MAT_s003
                                 + 0.00929510589631784 w_k1_MAT_s004
                                 + 0.0141910981987442 w_k1_MAT_s005
                                 + 0.0090894946787753 w_k1_MAT_s006
                                 + 0.0117927596144776 w_k1_MAT_s007
                                 + 0.0418814413601203 w_k1_MAT_s010
                                 + 0.0422189596830993 w_k1_MAT_s011
                                 + 0.013357806522461 w_k1_MAT_s012
                                 + 0.0546260718486316 w_k1_MAT_s013
                                 + 0.0125426414991889 w_k1_MAT_s014
                                 + 0.05421227588531 w_k1_MAT_s015
                                 + 0.0227399058570682 w_k1_MAT_s016
                                 + 0.0207802752981864 w_k1_MAT_s017
                                 + 0.0733264489148307 w_k1_MAT_s100
                                 + 0.0429818093508301 w_k1_MAT_s101
                                 + 0.0159810397550522 w_k1_MAT_s102
                                 + 0.0185732956371442 w_k1_MAT_s103
                                 + 0.0465358253777806 w_k1_MAT_s104
                                 + 0.042949398574151 w_k1_MAT_s105
                                 + 0.0505980816826122 w_k1_MAT_s106
                                 + 0.038074770491816 w_k1_MAT_s107
                                 + 0.0416234316540544 w_k1_MAT_s110
                                 + 0.0201723510079079 w_k1_MAT_s111
                                 + 0.0331080671551843 w_k1_MAT_s112
                                 + 0.0433743116103769 w_k1_MAT_s113
                                 + 0.039854393801237 w_k1_MAT_s114
                                 + 0.101824736791742 w_k1_MAT_s115
                                 + 0.0654715529991616 w_k1_MAT_s116
                                 + 0.0524805223834398 w_k1_MAT_s117
                                 + 0.00691937204449463 w_k1_MSFT_s000
                                 + 0.0150245112616839 w_k1_MSFT_s001
                                 + 0.00676898039498031 w_k1_MSFT_s002
                                 + 0.00751156141685786 w_k1_MSFT_s003
                                 + 0.00952101858350565 w_k1_MSFT_s004
                                 + 0.0126906673181778 w_k1_MSFT_s005
                                 + 0.009302118230634 w_k1_MSFT_s006
                                 + 0.00849802940924385 w_k1_MSFT_s007
                                 + 0.0446801507959155 w_k1_MSFT_s010
                                 + 0.0375812361683225 w_k1_MSFT_s011
                                 + 0.0144132459862722 w_k1_MSFT_s012
                                 + 0.053740340826104 w_k1_MSFT_s013
                                 + 0.0128394374787708 w_k1_MSFT_s014
                                 + 0.0479759811121952 w_k1_MSFT_s015
                                 + 0.0220561580166801 w_k1_MSFT_s016
                                 + 0.022414240939141 w_k1_MSFT_s017
                                 + 0.0728333935646022 w_k1_MSFT_s100
                                 + 0.0417854044066593 w_k1_MSFT_s101
                                 + 0.0155295475931774 w_k1_MSFT_s102
                                 + 0.0185238645831382 w_k1_MSFT_s103
                                 + 0.0408547282735654 w_k1_MSFT_s104
                                 + 0.0415297074703302 w_k1_MSFT_s105
                                 + 0.0389027925540987 w_k1_MSFT_s106
                                 + 0.0340428831673237 w_k1_MSFT_s107
                                 + 0.0365741172006951 w_k1_MSFT_s110
                                 + 0.0219398889723964 w_k1_MSFT_s111
                                 + 0.0290476816298053 w_k1_MSFT_s112
                                 + 0.0535898899880834 w_k1_MSFT_s113
                                 + 0.0407531486363058 w_k1_MSFT_s114
                                 + 0.0887030386037736 w_k1_MSFT_s115
                                 + 0.061733820633708 w_k1_MSFT_s116
                                 + 0.064242848677438 w_k1_MSFT_s117
                                 + 0.00649257648785922 w_k1_PG_s000
                                 + 0.014991875476918 w_k1_PG_s001
                                 + 0.00681631590075115 w_k1_PG_s002
                                 + 0.00730400766079378 w_k1_PG_s003
                                 + 0.00868924373976148 w_k1_PG_s004
                                 + 0.0132284155166482 w_k1_PG_s005
                                 + 0.00917029323284124 w_k1_PG_s006
                                 + 0.00895723653548869 w_k1_PG_s007
                                 + 0.0391840271716573 w_k1_PG_s010
                                 + 0.0378458695041258 w_k1_PG_s011
                                 + 0.014427324256505 w_k1_PG_s012
                                 + 0.05260986164256 w_k1_PG_s013
                                 + 0.0120882666773507 w_k1_PG_s014
                                 + 0.0542314731520288 w_k1_PG_s015
                                 + 0.0214792073264227 w_k1_PG_s016
                                 + 0.0214801290768065 w_k1_PG_s017
                                 + 0.0680468264503418 w_k1_PG_s100
                                 + 0.0399114250526587 w_k1_PG_s101
                                 + 0.0133859069207868 w_k1_PG_s102
                                 + 0.0184394368259752 w_k1_PG_s103
                                 + 0.0372879098080856 w_k1_PG_s104
                                 + 0.0409968987563545 w_k1_PG_s105
                                 + 0.0396023378630723 w_k1_PG_s106
                                 + 0.0329759179030307 w_k1_PG_s107
                                 + 0.0353536583434499 w_k1_PG_s110
                                 + 0.0191336323573882 w_k1_PG_s111
                                 + 0.0290680410975592 w_k1_PG_s112
                                 + 0.0484316340536895 w_k1_PG_s113
                                 + 0.0408601939524622 w_k1_PG_s114
                                 + 0.0849568238102722 w_k1_PG_s115
                                 + 0.0606933032570902 w_k1_PG_s116
                                 + 0.0677634794650302 w_k1_PG_s117 >= 1.13
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
